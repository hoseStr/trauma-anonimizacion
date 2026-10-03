"""
Fase 4 — Enriquecimiento para RAG.

Dos cosas, y las dos importan más que afinar el embedding:

1. Expansión AUMENTATIVA de abreviaturas.
   'HPAF en abdomen' -> 'HPAF (herida por arma de fuego) en abdomen'.
   El chunk indexado matchea tanto la consulta abreviada como la expandida.
   Solo la primera aparición por documento, para no inflar el texto.

2. Metadatos estructurados derivados del texto, para filtrado híbrido
   pre-retrieval: mecanismo de lesión, región anatómica, desenlace.
   En un RAG clínico el filtrado por metadatos sube la precisión más que
   cualquier tuning del modelo de embeddings.

Las reglas se apoyan en que el campo `Motivo de Consulta` está muy
estereotipado: 'ACCIDENTE DE TRANSITO' (1.975 + variantes), 'ME CAI' (868),
'ME DISPARARON' (466), 'ME APUÑALARON' (156).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml


def _sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", _sin_tildes(t).lower()).strip()


# ---------------------------------------------------------------------------
# 1. Expansión de abreviaturas
# ---------------------------------------------------------------------------
class ExpansorAbreviaturas:
    def __init__(self, ruta: str | Path):
        with open(ruta, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.mapa: dict[str, str] = {
            k.lower(): v for k, v in (cfg.get("abreviaturas") or {}).items()
        }
        if not self.mapa:
            self.rx = None
            return
        alt = "|".join(re.escape(k) for k in sorted(self.mapa, key=len, reverse=True))
        self.rx = re.compile(rf"\b({alt})\b", re.IGNORECASE)

    def expandir(self, texto: str) -> tuple[str, list[str]]:
        """Devuelve (texto_expandido, abreviaturas_encontradas)."""
        if not texto or self.rx is None:
            return texto, []
        vistas: set[str] = set()
        encontradas: list[str] = []

        def _f(m: re.Match) -> str:
            clave = m.group(1).lower()
            if clave in vistas:
                return m.group(0)
            vistas.add(clave)
            encontradas.append(clave)
            return f"{m.group(0)} ({self.mapa[clave]})"

        return self.rx.sub(_f, texto), encontradas


# ---------------------------------------------------------------------------
# 2. Metadatos derivados
# ---------------------------------------------------------------------------
MECANISMOS: dict[str, list[str]] = {
    "arma_de_fuego": [
        r"hpaf", r"\bhaf\b", r"arma\s+de\s+fuego", r"proyectil", r"disparar", r"disparo",
        r"me\s+pegaron\s+un\s+tiro", r"balea", r"pafm?\b", r"impacto\s+de\s+bala",
    ],
    "cortopunzante": [
        r"hpacp", r"\bhacp\b", r"cortopunzante", r"apu[nñ]al", r"pu[nñ]al", r"cuchill",
        r"machete", r"navaja", r"vidrio", r"me\s+cort", r"herida\s+por\s+arma\s+blanca",
    ],
    "accidente_transito": [
        r"accidente\s+de\s+tr[aá]nsito", r"\bavt\b", r"colisi[oó]n", r"volcamiento",
        r"atropell", r"moto\s+vs", r"veh[ií]culo", r"me\s+accident", r"choque",
    ],
    "caida": [
        r"\bca[ií]da\b", r"me\s+ca[ií]", r"se\s+cay[oó]", r"propia\s+altura",
        r"cay[oó]\s+de", r"precipitaci[oó]n", r"desde\s+su\s+altura",
    ],
    "contundente": [
        r"contundente", r"me\s+golpearon", r"le\s+golpearon", r"me\s+pegaron(?!\s+un\s+tiro)",
        r"pu[nñ]o", r"patada", r"palo", r"piedra", r"ri[nñ]a", r"pelea", r"agresi[oó]n\s+f[ií]sica",
    ],
    "quemadura": [r"quemadur", r"\bscq\b", r"escaldadura", r"[aá]cido", r"electrocuci[oó]n"],
    "mordedura": [r"mordedur", r"mordi[oó]", r"mordieron", r"canino", r"serpiente", r"ofidic"],
    "cuerpo_extrano": [r"cuerpo\s+extra[nñ]o", r"\bce\s+en\b"],
    "intoxicacion": [r"intoxica", r"envenenamiento", r"sobredosis", r"ingesta\s+de"],
    "autoinfligido": [r"autol[ií]tic", r"autoinfligid", r"intento\s+suicid"],
}

REGIONES: dict[str, list[str]] = {
    "craneo_encefalo": [r"\btce\b", r"\btec\b", r"craneo", r"cr[aá]neo", r"cefal",
                        r"encef[aá]lic", r"cuero\s+cabelludo", r"parietal", r"occipital",
                        r"frontal", r"temporal", r"glasgow"],
    "cara": [r"\bcara\b", r"facial", r"nasal", r"pir[aá]mide", r"mandibul", r"maxilar",
             r"labio", r"mentón", r"ment[oó]n", r"[oó]rbit"],
    "ojo": [r"\bocular\b", r"\bojo\b", r"\bod\b", r"\boi\b", r"c[oó]rnea", r"conjuntiv",
            r"escler", r"pupil"],
    "cuello": [r"cuello", r"cervical", r"tr[aá]quea", r"laring", r"zona\s+i{1,3}\s+de\s+cuello"],
    "torax": [r"t[oó]rax", r"tor[aá]cic", r"costal", r"pulmon", r"hemot[oó]rax",
              r"neumot[oó]rax", r"precordial", r"esternal", r"intercostal", r"\beic\b"],
    "abdomen": [r"abdomen", r"abdominal", r"\babd\b", r"h[ií]gado", r"hep[aá]tic", r"bazo",
                r"espl[eé]nic", r"intestin", r"laparotom", r"epig[aá]stric", r"flanco"],
    "pelvis": [r"pelvis", r"p[eé]lvic", r"cadera", r"isqui", r"pubi", r"il[ií]ac",
               r"genital", r"perin"],
    "columna": [r"columna", r"vertebr", r"lumbar", r"dorsal", r"toracolumbar", r"m[eé]dula"],
    "miembro_superior": [r"miembro\s+superior", r"\bmss?[di]\b", r"\bmmss\b", r"hombro",
                         r"brazo", r"antebrazo", r"codo", r"mu[nñ]eca", r"\bmano\b",
                         r"dedo(?!\s+del\s+pie)", r"clav[ií]cula", r"h[uú]mero", r"radi[oa]l",
                         r"c[uú]bit"],
    "miembro_inferior": [r"miembro\s+inferior", r"\bm[mi]?i{1,2}\b", r"\bmmii\b", r"muslo",
                         r"rodilla", r"pierna", r"tobillo", r"\bpie\b", r"f[eé]mur",
                         r"tibia", r"peron", r"calc[aá]neo"],
}

DESENLACES: list[tuple[str, list[str]]] = [
    ("fallecimiento", [r"fallec", r"[oó]bito", r"muerte", r"deceso", r"se\s+declara\s+muerte",
                       r"paro\s+cardiorrespiratorio\s+irreversible", r"no\s+hay\s+respuesta"]),
    ("cirugia", [r"procedimiento\s+quir[uú]rgico", r"llevado\s+a\s+cirug[ií]a", r"quir[oó]fano",
                 r"laparotom", r"toracotom", r"osteos[ií]ntesis", r"craniectom",
                 r"lavado\s+quir[uú]rgico", r"reducci[oó]n\s+abierta"]),
    ("uci", [r"\buci\b", r"cuidado\s+intensivo", r"ventilaci[oó]n\s+mec[aá]nica", r"\biot\b"]),
    ("hospitalizacion", [r"se\s+hospitaliza", r"hospitalizaci[oó]n", r"ingresa\s+a\s+piso",
                         r"observaci[oó]n", r"se\s+deja\s+en"]),
    ("remision", [r"se\s+remite", r"remitid", r"traslad", r"contrarrefer"]),
    ("alta", [r"se\s+da\s+salida", r"dar\s+salida", r"alta\s+m[eé]dica", r"manejo\s+ambulatorio",
              r"signos\s+de\s+alarma", r"egreso", r"se\s+ordena\s+salida"]),
]

_MEC = {k: [re.compile(p) for p in v] for k, v in MECANISMOS.items()}
_REG = {k: [re.compile(p) for p in v] for k, v in REGIONES.items()}
_DES = [(k, [re.compile(p) for p in v]) for k, v in DESENLACES]


def _cualquiera(patrones, texto: str) -> bool:
    return any(p.search(texto) for p in patrones)


def clasificar_mecanismo(motivo: str, cuerpo: str) -> str:
    """El motivo de consulta manda; el resto del texto es respaldo."""
    m, c = _norm(motivo), _norm(cuerpo)
    for fuente in (m, f"{m} {c}"):
        for etiqueta, pats in _MEC.items():
            if _cualquiera(pats, fuente):
                return etiqueta
    return "no_determinado"


def clasificar_regiones(texto: str) -> list[str]:
    t = _norm(texto)
    return [k for k, pats in _REG.items() if _cualquiera(pats, t)]


def clasificar_desenlace(texto: str) -> str:
    """Prioridad por gravedad: fallecimiento > cirugía > UCI > ... > alta."""
    t = _norm(texto)
    for etiqueta, pats in _DES:
        if _cualquiera(pats, t):
            return etiqueta
    return "no_determinado"
