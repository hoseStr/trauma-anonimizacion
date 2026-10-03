"""
Fase 2 — Flags de calidad.

Marca, nunca borra. Cada registro se etiqueta y la decisión de qué entra al
índice se toma al final (y se puede cambiar sin volver a procesar).

Flags:
  empty      -> los tres campos de texto vacíos
  too_short  -> menos de MIN_CHARS caracteres útiles en total
  truncated  -> campo cortado. Dos señales:
                · el export cortó examen físico y hospitalización a 500
                  caracteres; todo campo de esos dos a >= UMBRAL_TOPE se trata
                  como cortado (hay un pico en 495-499 que no llega a 500).
                  El motivo de consulta NO tiene tope: no aplica.
                · termina en palabra funcional ('...AL RECUPERARLA SE'): la
                  frase quedó a medias, sea quien sea el que la cortó.
                Terminar sin punto NO es señal: '...SE DA SALIDA' está completo.
  incompleto -> algún campo de texto vacío.
  duplicate  -> texto normalizado idéntico a otro registro (se conserva el
                de menor record_id como canónico)

`apto_rag` = indexable y ni truncado ni incompleto. Es lo único que llega al
índice: un caso sin evolución o cortado es justo lo que el generador tiende a
completar inventando, y en un asistente de diagnóstico eso no se tolera.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from .normalize import TEXT_COLS, clave_norm

# El export original cortó estos dos campos a 500 caracteres. El motivo de
# consulta no tiene tope (llega a más de 5.000).
LONGITUD_TRUNCAMIENTO = 500
CAMPOS_CON_TOPE = {"Descripción del examen físico", "Detalles de hospitalización"}
# Margen bajo el tope: hay un pico anómalo en 495-499 caracteres (cortes a 500
# a los que el export les quitó un carácter final), y una nota que llega a 480
# de 500 posibles no deja saber si seguía.
UMBRAL_TOPE = 480
MIN_CHARS = 40

# Una frase no termina en estas palabras (comparadas sin tildes, en minúsculas).
PALABRAS_FUNCIONALES = frozenset(
    "de del con y e o u el la los las en a al por para se que un una sin su sus "
    "le lo les mas ni como hacia desde entre sobre tras".split()
)


def _es_truncado(original: object, limpio: str, col: str) -> bool:
    """El tope se mide sobre el valor ORIGINAL; la palabra final, sobre el limpio."""
    if not limpio:
        return False
    if col in CAMPOS_CON_TOPE and len(str(original or "")) >= UMBRAL_TOPE:
        return True
    ultima = clave_norm(limpio.split()[-1]).strip(",;:-")
    return ultima in PALABRAS_FUNCIONALES


def hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def marcar_calidad(df: pd.DataFrame, min_chars: int = MIN_CHARS) -> pd.DataFrame:
    out = df.copy()
    clean_cols = [f"{c}_clean" for c in TEXT_COLS]

    # --- vacío / muy corto ---
    longitudes = out["texto_concat"].str.len()
    out["n_chars"] = longitudes
    out["flag_empty"] = longitudes.eq(0)
    out["flag_too_short"] = (~out["flag_empty"]) & longitudes.lt(min_chars)

    # --- truncamiento, por campo y agregado ---
    for col, clean in zip(TEXT_COLS, clean_cols):
        out[f"trunc::{col}"] = [
            _es_truncado(o, c, col) for o, c in zip(out[col], out[clean])
        ]
    out["flag_truncated"] = out[[f"trunc::{c}" for c in TEXT_COLS]].any(axis=1)
    out["campos_truncados"] = out[[f"trunc::{c}" for c in TEXT_COLS]].apply(
        lambda r: [c for c, v in zip(TEXT_COLS, r) if v], axis=1
    )
    out = out.drop(columns=[f"trunc::{c}" for c in TEXT_COLS])

    # --- incompleto: algún campo vacío (sin evolución no hay desenlace) ---
    vacios = pd.concat([out[c].eq("") for c in clean_cols], axis=1)
    vacios.columns = TEXT_COLS
    out["flag_incompleto"] = (~out["flag_empty"]) & vacios.any(axis=1)
    out["campos_vacios"] = vacios.apply(lambda r: [c for c in TEXT_COLS if r[c]], axis=1)

    # --- duplicados: hash del texto normalizado ---
    out["hash_texto"] = out["texto_norm"].map(hash_texto)
    con_texto = ~out["flag_empty"]
    orden = out.assign(_id=out["TraumaRegisterRecordID"].astype(int)).sort_values("_id")
    primeros = orden[con_texto.loc[orden.index]].drop_duplicates("hash_texto").index
    out["flag_duplicate"] = con_texto & ~out.index.isin(primeros)

    # --- decisión de indexado ---
    out["indexable"] = ~(
        out["flag_empty"] | out["flag_too_short"] | out["flag_duplicate"]
    )
    out["apto_rag"] = out["indexable"] & ~out["flag_truncated"] & ~out["flag_incompleto"]
    out["quality_flags"] = out.apply(
        lambda r: [
            n
            for n, v in (
                ("empty", r["flag_empty"]),
                ("too_short", r["flag_too_short"]),
                ("truncated", r["flag_truncated"]),
                ("incompleto", r["flag_incompleto"]),
                ("duplicate", r["flag_duplicate"]),
            )
            if v
        ],
        axis=1,
    )
    return out


def resumen_calidad(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    filas = [
        ("total", total),
        ("empty", int(df["flag_empty"].sum())),
        ("too_short", int(df["flag_too_short"].sum())),
        ("duplicate", int(df["flag_duplicate"].sum())),
        ("truncated", int(df["flag_truncated"].sum())),
        ("incompleto", int(df["flag_incompleto"].sum())),
        ("indexable", int(df["indexable"].sum())),
        ("excluido_truncado", int((df["indexable"] & df["flag_truncated"]).sum())),
        ("excluido_incompleto",
         int((df["indexable"] & ~df["flag_truncated"] & df["flag_incompleto"]).sum())),
        ("apto_rag", int(df["apto_rag"].sum())),
    ]
    res = pd.DataFrame(filas, columns=["metrica", "n"])
    res["pct"] = (res["n"] / total * 100).round(2)
    return res
