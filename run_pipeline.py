#!/usr/bin/env python3
"""
Pipeline de limpieza y anonimización de trauma.csv para RAG clínico.

    python run_pipeline.py
    python run_pipeline.py --entrada data/trauma.csv --salida outputs

Determinista: mismas entradas -> mismas salidas. El hash del CSV de origen
queda registrado en outputs/manifest.json para trazabilidad.

Fases:
    0  congelar y hashear el original
    1  normalización estructural
    2  flags de calidad
    3  anonimización (capas A y B)
    4  enriquecimiento (abreviaturas + metadatos)
    5  ensamblado del corpus y export
    6  verificación (residuo + muestra manual + reporte)

La capa C (NER) se corre aparte:  python -m src.ner_review
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.anonymize import Anonimizador
from src.build_corpus import construir_corpus, exportar
from src.enrich import (
    ExpansorAbreviaturas,
    clasificar_desenlace,
    clasificar_mecanismo,
    clasificar_regiones,
)
from src.normalize import TEXT_COLS, cargar_csv, normalizar_dataframe
from src.quality import marcar_calidad, resumen_calidad
from src.verify import (
    detectar_residuo,
    muestra_estratificada,
    reporte_html,
    resumen_residuo,
)

RAIZ = Path(__file__).resolve().parent


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=None,
                    help="ruta del CSV; por defecto busca data/trauma.csv y luego trauma.csv")
    ap.add_argument("--salida", default="outputs")
    ap.add_argument("--reportes", default="reports")
    ap.add_argument("--config", default="config")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--muestra", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aviso-en-texto", action="store_true",
                    help="ABLACIÓN: mete la nota de truncamiento dentro del texto indexado "
                         "(reproduce el corpus v1). Por defecto va solo en metadatos.")
    ap.add_argument("--incluir-incompletos", action="store_true",
                    help="ABLACIÓN: exporta con el criterio anterior (`indexable`), que deja "
                         "entrar truncados y registros con campos vacíos. Por defecto solo "
                         "entra `apto_rag`.")
    args = ap.parse_args()

    t0 = time.time()
    filtro = "indexable" if args.incluir_incompletos else "apto_rag"
    if args.entrada:
        entrada = Path(args.entrada)
    else:
        candidatos = [RAIZ / "data" / "trauma.csv", RAIZ / "trauma.csv", Path("trauma.csv")]
        entrada = next((c for c in candidatos if c.exists()), candidatos[0])
    if not entrada.exists():
        raise SystemExit(f"No encuentro el CSV. Pásalo con --entrada. Buscado: {entrada}")
    salida = Path(args.salida) if Path(args.salida).is_absolute() else RAIZ / args.salida
    reportes = Path(args.reportes) if Path(args.reportes).is_absolute() else RAIZ / args.reportes
    config = Path(args.config) if Path(args.config).is_absolute() else RAIZ / args.config

    # --- Fase 0: congelar ---------------------------------------------------
    digest = sha256(entrada)
    print(f"[0] {entrada}  sha256={digest[:16]}…")

    df = cargar_csv(entrada)
    print(f"    {len(df):,} registros · {len(df.columns)} columnas")

    # --- Fase 1: normalizar -------------------------------------------------
    df = normalizar_dataframe(df)
    print(f"[1] normalización estructural ✓")

    # --- Fase 2: calidad ----------------------------------------------------
    df = marcar_calidad(df, min_chars=args.min_chars)
    cal = resumen_calidad(df)
    print("[2] calidad:")
    for _, r in cal.iterrows():
        print(f"      {r['metrica']:22s} {int(r['n']):>7,}  {r['pct']:>6.2f}%")

    # --- Fase 3: anonimizar -------------------------------------------------
    anon = Anonimizador(config / "gazetteers.yaml")
    log_total: list = []
    for col in TEXT_COLS:
        destino = []
        for rid, texto in zip(df["TraumaRegisterRecordID"], df[f"{col}_clean"]):
            res = anon.anonimizar(texto, record_id=str(rid), campo=col)
            destino.append(res.texto)
            log_total.extend(res.reemplazos)
        df[f"{col}_anon"] = destino

    log_df = pd.DataFrame([vars(r) for r in log_total])
    por_registro = (
        log_df.groupby("record_id").size() if not log_df.empty else pd.Series(dtype=int)
    )
    tipos = (
        log_df.groupby("record_id")["regla"].agg(lambda s: sorted(set(s)))
        if not log_df.empty else pd.Series(dtype=object)
    )
    rid_str = df["TraumaRegisterRecordID"].astype(str)
    df["n_reemplazos_phi"] = rid_str.map(por_registro).fillna(0).astype(int)
    df["tipos_phi"] = rid_str.map(tipos).apply(lambda v: v if isinstance(v, list) else [])

    conteo_phi = (
        log_df.groupby("regla")
        .agg(n_reemplazos=("original", "size"), n_registros=("record_id", "nunique"))
        .reset_index().sort_values("n_reemplazos", ascending=False)
        if not log_df.empty else pd.DataFrame()
    )
    print(f"[3] anonimización: {len(log_df):,} reemplazos en "
          f"{df['n_reemplazos_phi'].gt(0).sum():,} registros")
    for _, r in conteo_phi.iterrows():
        print(f"      {r['regla']:22s} {int(r['n_reemplazos']):>7,}")

    # --- Fase 4: enriquecer -------------------------------------------------
    expansor = ExpansorAbreviaturas(config / "abreviaturas.yaml")
    for col in TEXT_COLS:
        textos, abrevs = [], []
        for t in df[f"{col}_anon"]:
            nuevo, encontradas = expansor.expandir(t)
            textos.append(nuevo)
            abrevs.append(encontradas)
        df[f"{col}_anon"] = textos
        df[f"abrev::{col}"] = abrevs
    df["abreviaturas"] = df[[f"abrev::{c}" for c in TEXT_COLS]].apply(
        lambda r: sorted({a for lst in r for a in lst}), axis=1
    )
    df = df.drop(columns=[f"abrev::{c}" for c in TEXT_COLS])

    motivo = df[f"{TEXT_COLS[0]}_anon"].fillna("")
    cuerpo = (df[f"{TEXT_COLS[1]}_anon"].fillna("") + " " + df[f"{TEXT_COLS[2]}_anon"].fillna(""))
    todo = motivo + " " + cuerpo
    df["mecanismo_lesion"] = [clasificar_mecanismo(m, c) for m, c in zip(motivo, cuerpo)]
    df["region_anatomica"] = [clasificar_regiones(t) for t in todo]
    df["desenlace"] = [clasificar_desenlace(t) for t in cuerpo]

    print("[4] enriquecimiento:")
    print("      mecanismo_lesion:")
    for k, v in df.loc[df[filtro], "mecanismo_lesion"].value_counts().items():
        print(f"        {k:22s} {v:>7,}")
    print("      desenlace:")
    for k, v in df.loc[df[filtro], "desenlace"].value_counts().items():
        print(f"        {k:22s} {v:>7,}")

    # --- Fase 5: corpus -----------------------------------------------------
    corpus = construir_corpus(df, incluir_aviso=args.aviso_en_texto)
    rutas = exportar(corpus, salida, columna_filtro=filtro)
    indexables = int(corpus[filtro].sum())
    excluidos = {
        "truncado": int((corpus["indexable"] & corpus["truncado"]).sum()),
        "incompleto": int((corpus["indexable"] & ~corpus["truncado"]
                           & corpus["quality_flags"].map(lambda f: "incompleto" in f)).sum()),
    }
    print(f"[5] corpus: {indexables:,} documentos al índice (criterio `{filtro}`)")
    if filtro == "apto_rag":
        print(f"      fuera por truncado {excluidos['truncado']:,} · "
              f"por campo vacío {excluidos['incompleto']:,}")
    for k, v in rutas.items():
        print(f"      {k:8s} {v}")

    # log de reemplazos, para auditoría
    if not log_df.empty:
        log_df.to_csv(salida / "log_reemplazos.csv", index=False, encoding="utf-8")

    # --- Fase 6: verificar --------------------------------------------------
    idx = corpus[corpus[filtro]]
    residuo = detectar_residuo(idx["texto"], idx["record_id"], config / "gazetteers.yaml")
    res_residuo = resumen_residuo(residuo, indexables)
    print("[6] residuo tras anonimizar (a_revisar = residuo real):")
    if res_residuo.empty:
        print("      ninguno")
    else:
        for _, r in res_residuo.iterrows():
            print(f"      {r['detector']:22s} {int(r['n_matches']):>6,} matches · "
                  f"a revisar {int(r['a_revisar']):>5,} en {int(r['n_docs']):>5,} docs "
                  f"({r['pct_docs_revisar']}%)")

    reportes.mkdir(parents=True, exist_ok=True)
    if not residuo.empty:
        residuo.to_csv(reportes / "residuo_phi.csv", index=False, encoding="utf-8")

    muestra = muestra_estratificada(corpus, n=args.muestra, seed=args.seed, columna=filtro)
    muestra[["record_id", "n_chars_doc", "mecanismo_lesion", "desenlace", "truncado", "texto"]] \
        .to_csv(reportes / "muestra_revision_manual.csv", index=False, encoding="utf-8")

    desc = pd.DataFrame(
        sorted(anon.medicos_desconocidos.items(), key=lambda x: -x[1]),
        columns=["token_tras_dr", "n"],
    )
    if not desc.empty:
        desc.to_csv(reportes / "medicos_no_whitelist.csv", index=False, encoding="utf-8")

    # control de SOBRE-anonimización: qué se reemplazó tras 'hospital'
    hosp = pd.DataFrame(
        sorted(anon.hospitales_desconocidos.items(), key=lambda x: -x[1]),
        columns=["token_tras_hospital", "n"],
    )
    if not hosp.empty:
        hosp.to_csv(reportes / "hospital_tokens_anonimizados.csv", index=False, encoding="utf-8")
        print(f"      control  {reportes / 'hospital_tokens_anonimizados.csv'} "
              f"({len(hosp)} tokens distintos)")

    revisar = residuo[residuo["veredicto"] == "revisar"] if not residuo.empty else residuo
    ejemplos = revisar.head(40) if not revisar.empty else pd.DataFrame()
    html = reporte_html(cal, conteo_phi, res_residuo, ejemplos, desc.head(40),
                        indexables, reportes / "reporte_anonimizacion.html")
    print(f"      reporte  {html}")
    print(f"      muestra  {reportes / 'muestra_revision_manual.csv'}")

    # --- manifest -----------------------------------------------------------
    manifest = {
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entrada": str(entrada),
        "sha256_entrada": digest,
        "n_registros": int(len(df)),
        "n_indexables": indexables,
        "criterio_indice": filtro,
        "n_excluidos_indexables": excluidos if filtro == "apto_rag" else {},
        "parametros": {
            "min_chars": args.min_chars,
            "seed": args.seed,
            "aviso_en_texto": args.aviso_en_texto,
            "incluir_incompletos": args.incluir_incompletos,
        },
        "calidad": {r["metrica"]: int(r["n"]) for _, r in cal.iterrows()},
        "reemplazos_por_regla": (
            {r["regla"]: int(r["n_reemplazos"]) for _, r in conteo_phi.iterrows()}
            if not conteo_phi.empty else {}
        ),
        "residuo_por_detector": (
            {r["detector"]: {"matches": int(r["n_matches"]), "a_revisar": int(r["a_revisar"])}
             for _, r in res_residuo.iterrows()}
            if not res_residuo.empty else {}
        ),
        "duracion_s": round(time.time() - t0, 1),
    }
    (salida / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nListo en {manifest['duracion_s']}s → {salida / 'manifest.json'}")


if __name__ == "__main__":
    main()
