"""
Fase 1 — Normalización estructural.

No toca la semántica del texto: arregla artefactos del export (comillas de
escape, espacios, saltos de línea, formas Unicode) y produce una columna
paralela `*_norm` (minúsculas, sin tildes) usada SOLO para deduplicar y para
hacer match contra gazetteers y diccionarios.

El texto que va al índice conserva casing y tildes: los embeddings
multilingües los aprovechan y quitarlos pierde señal.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

TEXT_COLS = [
    "Motivo de Consulta",
    "Descripción del examen físico",
    "Detalles de hospitalización",
]
ID_COL = "TraumaRegisterRecordID"

# --- artefactos de export detectados en el perfilado -------------------------
_RE_COMILLAS_MULTI = re.compile(r'"{2,}')
_RE_COMILLA_HUERFANA = re.compile(r'(?<![\w])"(?![\w])')
_RE_PREFIJO_BASURA = re.compile(r'^\s*[\d"\']+(?=[A-ZÁÉÍÓÚÑ]{3,})')  # '2HERIDA POR ARMA...'
_RE_ESPACIOS = re.compile(r"[ \t   ]+")
_RE_NL_MULTI = re.compile(r"\n{3,}")
_RE_NL_ESPACIO = re.compile(r"[ \t]*\n[ \t]*")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_PUNTO_REPETIDO = re.compile(r"\.{4,}")
_RE_ACENTO = re.compile(r"[̀-ͯ]")


def quitar_tildes(texto: str) -> str:
    """NFD -> elimina diacríticos -> NFC. 'REGIÓN' y 'REGION' colapsan."""
    return unicodedata.normalize(
        "NFC", _RE_ACENTO.sub("", unicodedata.normalize("NFD", texto))
    )


def normalizar_texto(valor: object) -> str:
    """Limpieza conservadora de un campo de texto libre."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    texto = str(valor)

    texto = unicodedata.normalize("NFC", texto)
    texto = _RE_CTRL.sub(" ", texto)

    # '"""SE CAYO"""' -> 'SE CAYO'   |   '2HERIDA POR ARMA...""' -> 'HERIDA POR ARMA...'
    texto = _RE_COMILLAS_MULTI.sub('"', texto)
    texto = _RE_PREFIJO_BASURA.sub("", texto)
    texto = texto.strip().strip('"').strip()
    texto = _RE_COMILLA_HUERFANA.sub("", texto)

    texto = _RE_PUNTO_REPETIDO.sub("...", texto)
    texto = _RE_NL_ESPACIO.sub("\n", texto)
    texto = _RE_NL_MULTI.sub("\n\n", texto)
    texto = _RE_ESPACIOS.sub(" ", texto)

    # espacio antes de puntuación y falta de espacio después
    texto = re.sub(r"\s+([,;:.])", r"\1", texto)
    texto = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", texto)

    return texto.strip()


def clave_norm(texto: str) -> str:
    """Clave de matching: minúsculas, sin tildes, espacios colapsados."""
    return _RE_ESPACIOS.sub(" ", quitar_tildes(texto).lower()).strip()


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Añade `<col>_clean` por cada columna de texto y `texto_norm` global."""
    out = df.copy()
    for col in TEXT_COLS:
        out[f"{col}_clean"] = out[col].map(normalizar_texto)

    out["texto_concat"] = out[[f"{c}_clean" for c in TEXT_COLS]].agg(
        lambda r: " \n".join(x for x in r if x), axis=1
    )
    out["texto_norm"] = out["texto_concat"].map(clave_norm)

    # `Unidad de edad`: única señal de edad que sobrevivió al export original
    unidad = out["Unidad de edad"].fillna("").map(clave_norm)
    out["es_pediatrico"] = unidad.eq("meses")
    out["unidad_edad"] = unidad.replace({"": None})
    return out


def cargar_csv(ruta) -> pd.DataFrame:
    """Lectura estricta: separador ';', todo como string, sin inferencia de tipos."""
    df = pd.read_csv(ruta, sep=";", dtype=str, engine="python", keep_default_na=False)
    df = df.replace({"": None})
    faltantes = [c for c in [ID_COL, *TEXT_COLS] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas ausentes en el CSV: {faltantes}")
    return df
