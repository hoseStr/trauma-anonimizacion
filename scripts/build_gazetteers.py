"""
Extrae del propio dataset las listas cerradas (gazetteers) que usará la Capa B
de anonimización: instituciones, apellidos de médicos y municipios.

Salida: config/gazetteers.yaml  (REVISAR A MANO antes de usar en producción)
"""
from __future__ import annotations
import re, csv, sys, collections, unicodedata
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data" / "trauma.csv"
OUT = ROOT / "config" / "gazetteers.yaml"

TEXT_COLS = ["Motivo de Consulta", "Descripción del examen físico", "Detalles de hospitalización"]

# Municipios del Valle del Cauca + capitales vecinas (semilla fija, no dependiente de los datos)
MUNICIPIOS_SEMILLA = [
    "cali", "santiago de cali", "palmira", "buenaventura", "tulua", "buga", "guadalajara de buga",
    "cartago", "jamundi", "yumbo", "candelaria", "florida", "pradera", "dagua", "zarzal",
    "sevilla", "caicedonia", "roldanillo", "la union", "el cerrito", "ginebra", "guacari",
    "restrepo", "calima", "darien", "vijes", "la cumbre", "yotoco", "san pedro", "andalucia",
    "bugalagrande", "trujillo", "riofrio", "bolivar", "el dovio", "versalles", "el aguila",
    "el cairo", "argelia", "ansermanuevo", "toro", "la victoria", "obando", "ulloa", "alcala",
    "popayan", "pasto", "bogota", "medellin", "puerto tejada", "santander de quilichao",
    "villa rica", "guapi", "timbio", "corinto", "miranda", "caloto", "padilla", "suarez",
    "buenos aires", "tumaco", "ipiales", "quibdo", "istmina", "tado", "condoto",
]

# Palabras que siguen a hospital/clínica pero NO son nombres propios de institución.
# Sin esta lista, "evolución clínica satisfactoria" se anonimizaría como institución.
STOP_TRAS_TIPO = {
    "satisfactoria", "satisfactorio", "estable", "favorable", "adecuada", "adecuado",
    "estricta", "estricto", "se", "por", "en", "de", "del", "la", "el", "y", "que",
    "con", "sin", "para", "donde", "nivel", "local", "general", "actual", "similar",
    "no", "es", "sera", "fue", "presenta", "ordena", "decide", "da", "dan", "queda",
    "cual", "lo", "los", "las", "un", "una", "al", "sus", "su", "mas", "menos",
    "compatible", "sugestiva", "sugestivo", "aceptable", "buena", "bueno", "mala",
    "regular", "critica", "critico", "leve", "moderada", "severa", "referida",
    "manifestada", "observada", "descrita",
}

def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", strip_acc(s).lower()).strip()

def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", dtype=str, engine="python")

def extraer_instituciones(blob: pd.Series) -> list[str]:
    """Captura 'hospital|clínica|fundación|IPS + <1-4 tokens capitalizados>' y filtra adjetivos."""
    tipo = r"(?:hospital|cl[ií]nica|fundaci[oó]n|centro\s+m[eé]dico|ips|clinica)"
    rx = re.compile(rf"\b{tipo}\s+((?:[A-Za-zÁÉÍÓÚÑáéíóúñ]+\s*){{1,4}})", re.I)
    cnt = collections.Counter()
    for t in blob:
        for m in rx.finditer(t):
            toks = [w for w in re.split(r"\s+", m.group(1).strip()) if w]
            # recortar en la primera palabra que sea "stopword tras tipo"
            keep = []
            for w in toks:
                if norm(w) in STOP_TRAS_TIPO:
                    break
                keep.append(w)
            if not keep:
                continue
            # descartar los que no parecen nombre propio (todo minúscula en texto mixto no basta;
            # exigimos >=1 token de >=4 letras y que no sea vocabulario clínico común)
            nombre = norm(" ".join(keep))
            if len(nombre) < 4:
                continue
            cnt[nombre] += 1
    # nos quedamos con los que aparecen >=3 veces: los hápax suelen ser ruido de parseo
    return sorted({k for k, v in cnt.items() if v >= 3})

def extraer_medicos(blob: pd.Series) -> list[str]:
    """Solo con trigger explícito Dr/Dra/Doctor(a): evita DRENADO, DRENAJE, etc."""
    rx = re.compile(r"\b(?:dr|dra|doctor|doctora)\.?\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]{3,})", re.I)
    # verbos/sustantivos que pueden seguir a "doctor" sin ser apellido
    falsos = {"ordena", "tratante", "de", "que", "en", "quien", "del", "la", "el",
              "general", "especialista", "encargado", "responsable", "y", "se", "por"}
    cnt = collections.Counter()
    for t in blob:
        for m in rx.finditer(t):
            ap = norm(m.group(1))
            if ap in falsos:
                continue
            cnt[ap] += 1
    return sorted(cnt)

def extraer_municipios(blob: pd.Series) -> list[str]:
    presentes = []
    joined = norm(" || ".join(blob.head(30000)))
    for m in MUNICIPIOS_SEMILLA:
        if re.search(rf"\b{re.escape(m)}\b", joined):
            presentes.append(m)
    return sorted(set(presentes))

def main() -> None:
    df = load(CSV_IN)
    blob = df[TEXT_COLS].fillna("").agg(" || ".join, axis=1)

    inst = extraer_instituciones(blob)
    med = extraer_medicos(blob)
    mun = extraer_municipios(blob)

    data = {
        "_nota": (
            "Generado por scripts/build_gazetteers.py a partir del propio corpus. "
            "REVISAR A MANO: añadir lo que falte, quitar falsos positivos. "
            "Las claves están normalizadas (minúsculas, sin tildes)."
        ),
        "instituciones": inst,
        "medicos_apellidos": med,
        "municipios": mun,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=100)

    print(f"instituciones: {len(inst)}")
    print(f"medicos:       {len(med)}")
    print(f"municipios:    {len(mun)}")
    print(f"-> {OUT}")

if __name__ == "__main__":
    main()
