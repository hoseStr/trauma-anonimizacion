"""
Fase 5 — Ensamblado del corpus y export.

Un registro = un documento. Con mediana de 163 caracteres por registro no hay
nada que partir por ventana deslizante: el chunking por caracteres solo
rompería frases. Los pocos documentos largos (>1500 chars) se marcan para que
el indexador decida si los parte por sección.

El documento se ensambla como narrativa con encabezados explícitos, porque el
encabezado es contexto que el embedding aprovecha y que el generador necesita
para saber si está leyendo un motivo de consulta o una nota de evolución.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ENCABEZADOS = {
    "Motivo de Consulta": "Motivo de consulta",
    "Descripción del examen físico": "Examen físico",
    "Detalles de hospitalización": "Evolución y hospitalización",
}
UMBRAL_DOC_LARGO = 1500


def ensamblar_documento(fila: pd.Series, sufijo: str = "_anon") -> str:
    partes = []
    for col, titulo in ENCABEZADOS.items():
        valor = (fila.get(f"{col}{sufijo}") or "").strip()
        if valor:
            partes.append(f"{titulo}: {valor}")
    return "\n\n".join(partes)


def aviso_truncado(fila: pd.Series | dict) -> str:
    """Advertencia para el GENERADOR: la fuente está incompleta.

    Con el filtro `apto_rag` los truncados ya no llegan al índice: esto solo
    se usa en la ablación (`--incluir-incompletos`, `--aviso-en-texto`).

    NO va dentro del texto indexado. Son 135 caracteres idénticos en 4.865
    documentos (22 % del índice) — el 17 % del documento en la mediana y hasta
    el 66 % en los más cortos. Embebido, acerca artificialmente entre sí a los
    casos truncados; en BM25 mete 'truncado', 'corpus' y 'desenlace' al
    vocabulario con frecuencia documental alta.

    Vive en los metadatos (`truncado`, `campos_truncados`) y se inyecta en el
    prompt en tiempo de generación. Ver rag/generate.py.
    """
    get = fila.get
    if not get("truncado", get("flag_truncated", False)):
        return ""
    campos = get("campos_truncados") or []
    nombres = ", ".join(ENCABEZADOS.get(c, c).lower() for c in campos)
    return (
        f"[Registro truncado en {nombres or 'algún campo'}: "
        f"el desenlace puede no estar documentado.]"
    )


def construir_corpus(df: pd.DataFrame, incluir_aviso: bool = False) -> pd.DataFrame:
    """`incluir_aviso` existe solo para reproducir el corpus v1 (ablación).

    El valor correcto es False: la advertencia va en metadatos, no en el texto.
    """
    docs = df.apply(ensamblar_documento, axis=1)
    if incluir_aviso:
        docs = docs + "\n\n" + df.apply(aviso_truncado, axis=1)

    out = pd.DataFrame(
        {
            "record_id": df["TraumaRegisterRecordID"],
            "texto": docs.str.strip(),
            "n_chars_doc": docs.str.len(),
            "campos_truncados": df["campos_truncados"],
            "campos_vacios": df["campos_vacios"],
            "mecanismo_lesion": df["mecanismo_lesion"],
            "region_anatomica": df["region_anatomica"],
            "desenlace": df["desenlace"],
            "es_pediatrico": df["es_pediatrico"],
            "unidad_edad": df["unidad_edad"],
            "abreviaturas": df["abreviaturas"],
            "quality_flags": df["quality_flags"],
            "truncado": df["flag_truncated"],
            "indexable": df["indexable"],
            "apto_rag": df["apto_rag"],
            "n_reemplazos_phi": df["n_reemplazos_phi"],
            "tipos_phi": df["tipos_phi"],
            "hash_texto": df["hash_texto"],
        }
    )
    out["doc_largo"] = out["n_chars_doc"] > UMBRAL_DOC_LARGO
    return out


def motivo_exclusion(fila: pd.Series) -> str:
    """Por qué un registro no llegó al índice. El primero que aplica."""
    for flag in ("empty", "too_short", "duplicate", "truncated", "incompleto"):
        if flag in fila["quality_flags"]:
            return flag
    return ""


def exportar(
    corpus: pd.DataFrame,
    salida: Path,
    nombre: str = "corpus",
    columna_filtro: str = "apto_rag",
) -> dict[str, Path]:
    """`columna_filtro` decide qué entra al JSONL. `indexable` reproduce el
    criterio anterior (truncados e incompletos incluidos) para la ablación."""
    salida.mkdir(parents=True, exist_ok=True)
    rutas: dict[str, Path] = {}

    parquet = salida / f"{nombre}.parquet"
    corpus.to_parquet(parquet, index=False)
    rutas["parquet"] = parquet

    # JSONL solo con lo apto: es lo que consume el pipeline de ingesta.
    # El nombre se conserva por compatibilidad con el prototipo RAG (repo trauma-rag).
    jsonl = salida / f"{nombre}_indexable.jsonl"
    idx = corpus[corpus[columna_filtro]]
    with jsonl.open("w", encoding="utf-8") as f:
        for _, r in idx.iterrows():
            f.write(
                json.dumps(
                    {
                        "id": str(r["record_id"]),
                        "text": r["texto"],
                        "metadata": {
                            "mecanismo_lesion": r["mecanismo_lesion"],
                            "region_anatomica": list(r["region_anatomica"]),
                            "desenlace": r["desenlace"],
                            "es_pediatrico": bool(r["es_pediatrico"]),
                            "truncado": bool(r["truncado"]),
                            "doc_largo": bool(r["doc_largo"]),
                            "n_chars": int(r["n_chars_doc"]),
                            "abreviaturas": list(r["abreviaturas"] or []),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    rutas["jsonl"] = jsonl

    # auditoría: qué quedó fuera del índice y por qué
    fuera = corpus[~corpus[columna_filtro]]
    excluidos = pd.DataFrame(
        {
            "record_id": fuera["record_id"],
            "motivo": [motivo_exclusion(r) for _, r in fuera.iterrows()],
            "campos_truncados": fuera["campos_truncados"].map(lambda l: "|".join(l or [])),
            "campos_vacios": fuera["campos_vacios"].map(lambda l: "|".join(l or [])),
            "n_chars_doc": fuera["n_chars_doc"],
        }
    )
    csv = salida / "excluidos_rag.csv"
    excluidos.to_csv(csv, index=False, encoding="utf-8")
    rutas["excluidos"] = csv
    return rutas
