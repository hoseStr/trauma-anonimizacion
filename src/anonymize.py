"""
Fase 3 — Anonimización (de-identificación).

Tres capas, en este orden:

  Capa A  reglas deterministas sobre lo que tiene formato reconocible
          (documentos, historias clínicas, fechas, horas, edades)
  Capa B  gazetteers con desambiguación por contexto
          (instituciones, médicos, municipios)
  Capa C  NER estadístico -> ver src/ner_review.py (NO reemplaza, solo reporta)

Principios de diseño:

* Se GENERALIZA con placeholders tipados, no se suprime. `[FECHA]` conserva la
  estructura sintáctica de la frase (mejor embedding) y le dice al generador
  que ahí había un dato, en vez de dejar un hueco que tiende a rellenar.
* Las edades no se borran: se llevan a rango (`[EDAD:80-89]`, `[EDAD:90+]`
  siguiendo la regla de Safe Harbor para ≥90). Se conserva la señal clínica
  geriátrico/pediátrico sin el identificador.
* Todo reemplazo se registra con su span y su texto original en un log
  auditable. Sin log no hay verificación posible.
* Cada regla se escribió contra los falsos positivos que este corpus produce
  realmente: "clínica satisfactoria", "antebrazo dr.", Glasgow "13/15",
  adrenalina "1:1000", "hace 3 días". Ver tests/test_anonymize.py.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

# ---------------------------------------------------------------------------
# Utilidades de matching tolerante a tildes
# ---------------------------------------------------------------------------
_EQUIV = {
    "a": "aá", "e": "eé", "i": "ií", "o": "oó", "u": "uúü",
    "n": "nñ", "c": "cç",
}


def _sin_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def rx_tolerante(frase: str) -> str:
    """'almendros' -> '[aá]lm[eé]ndr[oó]s' (y espacios flexibles)."""
    partes = []
    for ch in frase.lower():
        if ch == " ":
            partes.append(r"\s+")
        elif ch in _EQUIV:
            partes.append(f"[{_EQUIV[ch]}]")
        elif ch.isalnum():
            partes.append(re.escape(ch))
        else:
            partes.append(re.escape(ch))
    return "".join(partes)


# ---------------------------------------------------------------------------
# Log de reemplazos
# ---------------------------------------------------------------------------
@dataclass
class Reemplazo:
    record_id: str
    campo: str
    regla: str
    original: str
    placeholder: str
    inicio: int


@dataclass
class ResultadoAnon:
    texto: str
    reemplazos: list[Reemplazo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Capa A — reglas deterministas
# ---------------------------------------------------------------------------

# Identificadores. Lo más grave del corpus aunque sea lo más raro (3 casos).
RX_DOCUMENTO = re.compile(
    r"\b(?:c\.?c\.?|t\.?i\.?|h\.?c\.?|nuip|r\.?c\.?|historia\s+cl[ií]nica|"
    r"documento|c[eé]dula|identificaci[oó]n)\s*(?:n[oº°]?\.?|#|:)?\s*"
    r"(\d[\d\.\s]{3,15}\d)",
    re.IGNORECASE,
)
# Entero suelto largo. Las magnitudes clínicas siempre llevan unidad
# ('4000cc', '3 cm'), así que un entero pelado de 7+ dígitos es un identificador.
RX_NUM_LARGO = re.compile(
    r"(?<![\d.,:/-])\d{7,12}(?![\d.,:/-])"
    r"(?!\s*(?:cc|ml|mg|mcg|g|gr|kg|cm|mm|mts?|ui?|meq|mmhg))",
    re.IGNORECASE,
)
RX_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RX_TELEFONO = re.compile(r"(?<!\d)(?:\+?57[\s-]?)?3\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")

_MESES = (
    r"ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|"
    r"jul(?:io)?|ago(?:sto)?|sep(?:t(?:iembre)?)?|set(?:iembre)?|oct(?:ubre)?|"
    r"nov(?:iembre)?|dic(?:iembre)?"
)
# Fecha numérica completa. Exige TRES componentes: no colisiona con Glasgow
# 13/15 ni con tensión arterial 120/80, que tienen dos.
# Se aceptan hasta 4 dígitos por componente porque el corpus trae erratas
# reales de digitación ('15/058/2019').
RX_FECHA_NUM = re.compile(
    r"(?<![\d/])\d{1,4}\s*[/\-.]\s*\d{1,4}\s*[/\-.]\s*\d{2,4}(?![\d/])"
)
# El separador antes del año es opcional: el corpus trae '11-jun2014' pegado.
# El lookahead (?=\b|\d) evita que 'jun' matchee dentro de 'junta'.
RX_FECHA_TXT = re.compile(
    rf"\b\d{{1,2}}\s*(?:de\s+|[-/. ])\s*(?:{_MESES})(?=\b|\d)\.?"
    rf"(?:\s*(?:de\s+|[-/ ])?\s*\d{{2,4}}\b)?",
    re.IGNORECASE,
)
RX_MES_ANIO = re.compile(rf"\b(?:{_MESES})\s+(?:de\s+)?(?:19|20)\d{{2}}\b", re.IGNORECASE)

# Año suelto. Se excluye si le sigue una unidad ('2000 cc' de hemotórax).
RX_ANIO = re.compile(
    r"(?<![\d/.-])(?:19[5-9]\d|20[0-3]\d)(?![\d/.-])"
    r"(?!\s*(?:cc|ml|mg|mcg|g|gr|kg|cm|mm|mts?|ui?|meq|mmhg))",
    re.IGNORECASE,
)

# Hora. Los lookarounds evitan adrenalina 1:1000 y relaciones 1:1.
# El lookahead NO lleva \s*: con él, '16:40 26 nov' quedaba sin anonimizar
# porque el dígito del día siguiente lo vetaba.
RX_HORA = re.compile(
    r"(?<![\d:])(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d(?![:\d])"
    r"(?:\s*(?:a\.?m\.?|p\.?m\.?|hrs?|horas))?",
    re.IGNORECASE,
)

# Edad. Exige un disparador de persona o el sufijo 'de edad'; excluye
# explícitamente los tiempos relativos ('hace 3 días', 'en 2 meses').
_PERSONA = (
    r"paciente|pcte|pte|masculino|femenino|hombre|mujer|menor|adulto|anciano|"
    r"lactante|neonato|escolar|adolescente|joven|se[nñ]or(?:a)?|sr\.?|sra\.?|"
    r"ni[nñ][oa]|beb[eé]|var[oó]n"
)
RX_EDAD_PERSONA = re.compile(
    rf"(?:(?:{_PERSONA})\s+(?:de\s+)?)(\d{{1,3}})\s*(a[nñ]os?|meses|mes|d[ií]as?)",
    re.IGNORECASE,
)
RX_EDAD_SUFIJO = re.compile(
    r"(?<!hace\s)(?<!en\s)\b(\d{1,3})\s*(a[nñ]os?|meses|mes)\s+de\s+edad\b",
    re.IGNORECASE,
)


def _rango_edad(valor: int, unidad: str) -> str:
    u = _sin_tildes(unidad.lower())
    if u.startswith("dia"):
        return "[EDAD:<1 mes]"
    if u.startswith("mes"):
        return "[EDAD:<1 año]"
    if valor >= 90:                      # regla Safe Harbor
        return "[EDAD:90+]"
    if valor < 1:
        return "[EDAD:<1 año]"
    if valor < 18:
        base = (valor // 5) * 5
        return f"[EDAD:{base}-{base + 4}]"
    base = (valor // 10) * 10
    return f"[EDAD:{base}-{base + 9}]"


# ---------------------------------------------------------------------------
# Anonimizador
# ---------------------------------------------------------------------------
class Anonimizador:
    def __init__(self, ruta_gazetteers: str | Path):
        with open(ruta_gazetteers, encoding="utf-8") as f:
            g = yaml.safe_load(f)

        self.tipos_genericos = {t.lower() for t in g.get("tipos_genericos", [])}
        self.hospital_genericos = {t.lower() for t in g.get("hospital_genericos", [])}
        self.medicos = {m.lower() for m in g.get("medicos", [])}

        # instituciones: más largas primero para que gane el match más específico
        inst = sorted(g.get("instituciones", []), key=len, reverse=True)
        self.rx_institucion = self._compilar_instituciones(inst)
        self.rx_generico_tras_tipo = self._compilar_genericos()

        # Regla estructural: 'hospital/clínica + San|Santa|Santo + Nombre' es
        # casi siempre una IPS aunque el nombre no esté en el gazetteer.
        # Cubre San Rafael, San Jorge, Santa Catalina, Santa Margarita… y los
        # errores de digitación ('hospital sann juan de dios').
        self.rx_institucion_santoral = re.compile(
            rf"{self._tipo_institucion()}{self._relleno()}\s+"
            r"s[aá]n{1,2}[a-z]{0,3}\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]{3,}"
            r"(?:\s+(?:de|del)\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]{3,})?",
            re.IGNORECASE,
        )

        # Regla abierta solo para 'hospital'/'ESE'/'IPS': ahí el token siguiente
        # es un nombre propio salvo que esté en hospital_genericos. No se aplica
        # a 'clínica', que en este corpus es adjetivo la mitad de las veces.
        self.rx_hospital_abierto = re.compile(
            r"\b(?:hospital(?:es)?|e\.?s\.?e\.?|ips)"
            + self._relleno()
            + r"\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]{3,})"
            r"(?:\s+(?:de|del)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]{3,}))?",
            re.IGNORECASE,
        )
        self.hospitales_desconocidos: dict[str, int] = {}

        siglas = g.get("instituciones_siglas", [])
        self.rx_siglas = (
            re.compile(r"\b(?:" + "|".join(re.escape(s) for s in siglas) + r")\b")
            if siglas else None
        )

        sin_trigger = sorted(g.get("instituciones_sin_trigger", []), key=len, reverse=True)
        self.rx_sin_trigger = (
            re.compile(r"\b(?:" + "|".join(rx_tolerante(n) for n in sin_trigger) + r")\b",
                       re.IGNORECASE)
            if sin_trigger else None
        )

        muni = sorted(g.get("municipios", []), key=len, reverse=True)
        self.rx_municipio = re.compile(
            r"\b(?:" + "|".join(rx_tolerante(m) for m in muni) + r")\b", re.IGNORECASE
        ) if muni else None
        self.rx_excepciones_muni = [
            re.compile(p, re.IGNORECASE) for p in g.get("municipios_excepciones", [])
        ]

        self.rx_medico = re.compile(
            r"\b(?:dr|dra|doctor|doctora)\s*\.?\s*"
            r"([A-Za-zÁÉÍÓÚÑáéíóúñ]{3,})",
            re.IGNORECASE,
        )
        # médicos vistos tras el trigger pero fuera del whitelist -> a revisión
        self.medicos_desconocidos: dict[str, int] = {}

    # -- construcción de patrones -------------------------------------------
    @staticmethod
    def _tipo_institucion() -> str:
        return r"(?:hospital(?:es)?|cl[ií]nicas?|fundaci[oó]n|centro\s+m[eé]dico|ips|e\.?s\.?e\.?)"

    @staticmethod
    def _relleno() -> str:
        """Palabras que pueden ir entre el tipo y el nombre propio.

        'hospital INFANTIL rayito de sol', 'hospital DEPARTAMENTAL de versalles',
        'clínica DE la aurora'. Sin esto se perdían instituciones reales.
        """
        return (
            r"(?:\s+(?:de|del|la|el|los|las|infantil|universitario|"
            r"departamental|regional|local|municipal|nivel)){0,3}"
        )

    def _compilar_instituciones(self, nombres: Iterable[str]) -> re.Pattern | None:
        nombres = list(nombres)
        if not nombres:
            return None
        alt = "|".join(rx_tolerante(n) for n in nombres)
        return re.compile(
            rf"{self._tipo_institucion()}{self._relleno()}\s+(?:{alt})\b",
            re.IGNORECASE,
        )

    def _compilar_genericos(self) -> re.Pattern | None:
        if not self.tipos_genericos:
            return None
        alt = "|".join(rx_tolerante(t) for t in sorted(self.tipos_genericos, key=len, reverse=True))
        return re.compile(rf"{self._tipo_institucion()}\s+(?:{alt})\s*", re.IGNORECASE)

    # -- aplicación ----------------------------------------------------------
    def anonimizar(self, texto: str, record_id: str = "", campo: str = "") -> ResultadoAnon:
        if not texto:
            return ResultadoAnon("")

        log: list[Reemplazo] = []

        def sub(rx: re.Pattern, regla: str, repl, s: str) -> str:
            def _f(m: re.Match) -> str:
                nuevo = repl(m) if callable(repl) else repl
                if nuevo is None:                # la regla decidió no tocar
                    return m.group(0)
                log.append(
                    Reemplazo(record_id, campo, regla, m.group(0), nuevo, m.start())
                )
                return nuevo
            return rx.sub(_f, s)

        t = texto

        # --- Capa A: identificadores directos (lo más sensible primero) ---
        t = sub(RX_EMAIL, "email", "[EMAIL]", t)
        t = sub(RX_DOCUMENTO, "documento", "[DOCUMENTO]", t)
        t = sub(RX_TELEFONO, "telefono", "[TELEFONO]", t)
        t = sub(RX_NUM_LARGO, "numero_id", "[NUMERO_ID]", t)

        # --- Capa B: instituciones antes que municipios ---
        # Los nombres completos e inequívocos van primero, con o sin trigger.
        if self.rx_sin_trigger is not None:
            t = sub(self.rx_sin_trigger, "institucion", "[INSTITUCION]", t)
        if self.rx_institucion is not None:
            t = sub(self.rx_institucion, "institucion", self._repl_institucion, t)
        t = sub(self.rx_institucion_santoral, "institucion_santoral", "[INSTITUCION]", t)
        t = sub(self.rx_hospital_abierto, "institucion_hospital", self._repl_hospital, t)
        if self.rx_siglas is not None:
            t = sub(self.rx_siglas, "institucion_sigla", "[INSTITUCION]", t)

        t = sub(self.rx_medico, "medico", self._repl_medico, t)

        # --- Capa A: temporales ---
        t = sub(RX_FECHA_NUM, "fecha", "[FECHA]", t)
        t = sub(RX_FECHA_TXT, "fecha", "[FECHA]", t)
        t = sub(RX_MES_ANIO, "fecha", "[FECHA]", t)
        t = sub(RX_HORA, "hora", "[HORA]", t)
        t = sub(RX_ANIO, "anio", "[ANIO]", t)

        # --- Capa A: edades a rango (después de fechas, para no comerse años) ---
        t = sub(RX_EDAD_SUFIJO, "edad", self._repl_edad_sufijo, t)
        t = sub(RX_EDAD_PERSONA, "edad", self._repl_edad_persona, t)

        # --- Capa B: municipios al final ---
        if self.rx_municipio is not None:
            t = self._sub_municipios(t, log, record_id, campo)

        return ResultadoAnon(t, log)

    # -- reemplazos con lógica --------------------------------------------
    def _repl_institucion(self, m: re.Match) -> str | None:
        """Descarta 'clínica satisfactoria', 'hospital de periferia', etc.

        Se exige FULLMATCH, no match de prefijo: 'hospital buena esperanza' es
        una IPS real y no puede quedar vetado porque 'buena' esté en la lista
        de adjetivos genéricos.
        """
        if self.rx_generico_tras_tipo and self.rx_generico_tras_tipo.fullmatch(m.group(0).strip()):
            return None
        return "[INSTITUCION]"

    def _repl_hospital(self, m: re.Match) -> str | None:
        """'hospital <token>' -> [INSTITUCION] salvo que el token sea genérico."""
        token = _sin_tildes(m.group(1)).lower()
        if token in self.hospital_genericos:
            return None
        self.hospitales_desconocidos[token] = self.hospitales_desconocidos.get(token, 0) + 1
        return "[INSTITUCION]"

    def _repl_medico(self, m: re.Match) -> str | None:
        """Solo reemplaza si el token está en el whitelist curado.

        'antebrazo dr.\\n sangrado activo' y 'sin dr pero con disco óptico'
        quedan intactos; el token desconocido se acumula para el reporte.
        """
        token = _sin_tildes(m.group(1)).lower()
        if token in self.medicos:
            return "[MEDICO]"
        self.medicos_desconocidos[token] = self.medicos_desconocidos.get(token, 0) + 1
        return None

    @staticmethod
    def _repl_edad_persona(m: re.Match) -> str:
        return m.group(0)[: m.start(1) - m.start(0)] + _rango_edad(
            int(m.group(1)), m.group(2)
        )

    @staticmethod
    def _repl_edad_sufijo(m: re.Match) -> str:
        return _rango_edad(int(m.group(1)), m.group(2))

    def _sub_municipios(self, texto: str, log: list, rid: str, campo: str) -> str:
        excl: list[tuple[int, int]] = []
        for rx in self.rx_excepciones_muni:
            excl.extend((mm.start(), mm.end()) for mm in rx.finditer(texto))

        def _f(m: re.Match) -> str:
            if any(a <= m.start() < b for a, b in excl):
                return m.group(0)
            log.append(Reemplazo(rid, campo, "municipio", m.group(0), "[MUNICIPIO]", m.start()))
            return "[MUNICIPIO]"

        return self.rx_municipio.sub(_f, texto)
