"""
Fase 6 — Verificación.

No se confía en la anonimización: se mide. Dos cosas:

1. Segundo pase de detección de PHI sobre la salida YA anonimizada, con
   patrones independientes de los que hicieron el reemplazo. Lo que aparezca
   aquí es residuo real.
2. Muestreo estratificado para revisión manual, sesgado hacia los documentos
   largos: son los que acumulan más PHI por registro.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Detectores independientes de los de anonymize.py. Si estos encuentran algo
# en la salida, es residuo real (o un falso positivo que hay que documentar).
DETECTORES: dict[str, re.Pattern] = {
    "fecha_numerica": re.compile(r"(?<![\d/])\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}(?![\d/])"),
    "fecha_textual": re.compile(
        r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|setiembre|octubre|noviembre|diciembre)\b", re.I),
    "anio": re.compile(r"(?<![\d/.-])(?:19[5-9]\d|20[0-3]\d)(?![\d/.-])"),
    "hora": re.compile(r"(?<![\d:])(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:])"),
    "num_largo": re.compile(r"(?<![\d.,:/-])\d{7,12}(?![\d.,:/-])"),
    "documento_kw": re.compile(
        r"\b(?:c\.?c\.?|t\.?i\.?|h\.?c\.?|c[eé]dula|documento)\s*[:#nº.]*\s*\d{4,}", re.I),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "telefono": re.compile(r"(?<!\d)3\d{9}(?!\d)"),
    "edad_explicita": re.compile(r"\b(?:de\s+)?\d{1,3}\s*a[nñ]os\s+de\s+edad\b", re.I),
    "trigger_medico": re.compile(r"\b(?:dr|dra|doctor|doctora)\s*\.?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}"),
    "trigger_institucion": re.compile(
        r"\b(?:hospital|cl[ií]nica|fundaci[oó]n)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}", re.I),
}


# ---------------------------------------------------------------------------
# Clasificación del residuo
#
# Los detectores de arriba se mantienen deliberadamente "tontos": no importan
# nada de anonymize.py, así que no heredan sus puntos ciegos. Lo que sí se lee
# es config/gazetteers.yaml, que es la DECLARACIÓN AUDITABLE de qué se decidió
# no anonimizar. Un revisor puede abrir ese archivo y contrastar.
#
#   veredicto = "esperado"  -> exclusión declarada en la configuración
#   veredicto = "revisar"   -> residuo real, alguien tiene que mirarlo
# ---------------------------------------------------------------------------
_UNIDAD_TRAS = re.compile(r"^\s*(?:cc|ml|mg|mcg|gr?|kg|cm|mm|mts?|ui?|meq|mmhg)\b", re.I)
_TRIGGER_TOKEN = re.compile(
    r"^\W*(?:hospital(?:es)?|cl[ií]nicas?|fundaci[oó]n|e\.?s\.?e\.?|ips)\W+(.+?)\W*$", re.I
)
_DR_TOKEN = re.compile(r"^\W*(?:dr|dra|doctor|doctora)\W+(.+?)\W*$", re.I)


def _sin_tildes(t: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


class ClasificadorResiduo:
    def __init__(self, ruta_config: Path | str | None = None):
        self.genericos: set[str] = set()
        self.medicos: set[str] = set()
        if ruta_config:
            import yaml
            with open(ruta_config, encoding="utf-8") as f:
                g = yaml.safe_load(f) or {}
            self.genericos = {
                _sin_tildes(str(t)).lower()
                for t in (g.get("tipos_genericos") or []) + (g.get("hospital_genericos") or [])
            }
            self.medicos = {_sin_tildes(str(m)).lower() for m in (g.get("medicos") or [])}

    def __call__(self, detector: str, match: str, contexto: str) -> str:
        if detector == "anio":
            pos = contexto.find(match)
            if pos >= 0 and _UNIDAD_TRAS.match(contexto[pos + len(match):]):
                return "esperado"
            return "revisar"

        if detector == "trigger_institucion":
            m = _TRIGGER_TOKEN.match(match)
            if m and _sin_tildes(m.group(1)).lower() in self.genericos:
                return "esperado"
            return "revisar"

        if detector == "trigger_medico":
            m = _DR_TOKEN.match(match)
            # token que NO está en el whitelist de médicos -> se dejó a propósito
            if m and _sin_tildes(m.group(1)).lower() not in self.medicos:
                return "esperado"
            return "revisar"

        return "revisar"


def detectar_residuo(
    textos: pd.Series, ids: pd.Series, ruta_config: Path | str | None = None
) -> pd.DataFrame:
    clasificar = ClasificadorResiduo(ruta_config)
    filas = []
    for rid, t in zip(ids, textos):
        if not isinstance(t, str) or not t:
            continue
        for nombre, rx in DETECTORES.items():
            for m in rx.finditer(t):
                ctx = t[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
                filas.append(
                    {
                        "record_id": rid,
                        "detector": nombre,
                        "match": m.group(0),
                        "veredicto": clasificar(nombre, m.group(0), ctx),
                        "contexto": ctx,
                    }
                )
    return pd.DataFrame(
        filas, columns=["record_id", "detector", "match", "veredicto", "contexto"]
    )


def resumen_residuo(residuo: pd.DataFrame, n_docs: int) -> pd.DataFrame:
    cols = ["detector", "n_matches", "a_revisar", "n_docs", "pct_docs_revisar"]
    if residuo.empty:
        return pd.DataFrame(columns=cols)
    rev = residuo[residuo["veredicto"] == "revisar"]
    r = (
        residuo.groupby("detector")
        .agg(n_matches=("match", "size"))
        .reset_index()
    )
    agg_rev = (
        rev.groupby("detector")
        .agg(a_revisar=("match", "size"), n_docs=("record_id", "nunique"))
        .reset_index()
    )
    r = r.merge(agg_rev, on="detector", how="left").fillna({"a_revisar": 0, "n_docs": 0})
    r["a_revisar"] = r["a_revisar"].astype(int)
    r["n_docs"] = r["n_docs"].astype(int)
    r["pct_docs_revisar"] = (r["n_docs"] / n_docs * 100).round(3)
    return r[cols].sort_values("a_revisar", ascending=False)


def muestra_estratificada(
    corpus: pd.DataFrame, n: int = 250, seed: int = 42, columna: str = "apto_rag"
) -> pd.DataFrame:
    """Sesgada a documentos largos: concentran el PHI y son los que más pesan.

    Se muestrea sobre `columna` (lo que llega al índice), no sobre todo el corpus.
    """
    idx = corpus[corpus[columna]].copy()
    if idx.empty:
        return idx
    idx["estrato"] = pd.qcut(
        idx["n_chars_doc"].rank(method="first"), q=4,
        labels=["q1_corto", "q2", "q3", "q4_largo"],
    )
    pesos = {"q1_corto": 0.10, "q2": 0.15, "q3": 0.25, "q4_largo": 0.50}
    partes = []
    for estrato, peso in pesos.items():
        sub = idx[idx["estrato"] == estrato]
        k = min(len(sub), max(1, int(round(n * peso))))
        partes.append(sub.sample(k, random_state=seed))
    return pd.concat(partes).sample(frac=1, random_state=seed)


def reporte_html(
    resumen_cal: pd.DataFrame,
    conteo_phi: pd.DataFrame,
    res_residuo: pd.DataFrame,
    ejemplos: pd.DataFrame,
    medicos_desconocidos: pd.DataFrame,
    n_docs: int,
    salida: Path,
) -> Path:
    def tabla(df: pd.DataFrame, vacio: str) -> str:
        if df is None or df.empty:
            return f'<p class="ok">{vacio}</p>'
        return df.to_html(index=False, border=0, classes="t", escape=True)

    css = """
    :root{--bg:#fbfbfa;--fg:#1c1c1a;--mut:#6b6b66;--line:#e3e2df;--acc:#7a4a2b}
    @media(prefers-color-scheme:dark){:root{--bg:#191917;--fg:#eceae6;--mut:#a0a09a;--line:#33322e;--acc:#d99a6c}}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
    .wrap{max-width:1000px;margin:0 auto;padding-block:48px;padding-left:20px;padding-right:20px}
    h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
    .sub{color:var(--mut);margin:0 0 32px;font-size:14px}
    .grid{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
    .kpi{flex:1 1 150px;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
    .kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
    .kpi span{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
    .scroll{overflow-x:auto}
    table.t{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
    table.t th{text-align:left;background:transparent;color:var(--mut);font-weight:600;
      border-bottom:1px solid var(--line);padding:7px 10px;white-space:nowrap}
    table.t td{border-bottom:1px solid var(--line);padding:7px 10px;vertical-align:top}
    table.t tr:hover td{background:color-mix(in srgb,var(--acc) 7%,transparent)}
    .ok{color:var(--mut);font-style:italic}
    code{background:color-mix(in srgb,var(--fg) 8%,transparent);padding:1px 5px;border-radius:4px;font-size:12px}
    """
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reporte de anonimización — trauma.csv</title><style>{css}</style></head><body><div class="wrap">
<h1>Reporte de anonimización</h1>
<p class="sub">Corpus <code>trauma.csv</code> · {n_docs:,} documentos indexables · generado por <code>run_pipeline.py</code></p>

<h2>1. Calidad del corpus</h2>
<div class="scroll">{tabla(resumen_cal, "sin datos")}</div>

<h2>2. Reemplazos aplicados (capas A y B)</h2>
<div class="scroll">{tabla(conteo_phi, "no se aplicó ningún reemplazo")}</div>

<h2>3. Residuo tras la anonimización</h2>
<p class="sub">Detectores independientes de los que hicieron el reemplazo. Lo que aparece aquí es
residuo real o un falso positivo que hay que documentar.</p>
<div class="scroll">{tabla(res_residuo, "sin residuo detectado")}</div>

<h2>4. Ejemplos de residuo</h2>
<div class="scroll">{tabla(ejemplos, "sin ejemplos")}</div>

<h2>5. Tokens tras Dr/Dra fuera del whitelist</h2>
<p class="sub">Revisar: si alguno es un apellido real, añadirlo a <code>config/gazetteers.yaml</code>
en la clave <code>medicos</code> y volver a correr el pipeline.</p>
<div class="scroll">{tabla(medicos_desconocidos, "ninguno")}</div>
</div></body></html>"""
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")
    return salida
