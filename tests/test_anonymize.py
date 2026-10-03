"""
Tests de la anonimización, escritos contra los falsos positivos que este
corpus produce realmente. Correr con:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anonymize import Anonimizador  # noqa: E402
from src.normalize import normalizar_texto  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config" / "gazetteers.yaml"


@pytest.fixture(scope="module")
def anon() -> Anonimizador:
    return Anonimizador(CONFIG)


def ano(anon: Anonimizador, texto: str) -> str:
    return anon.anonimizar(texto).texto


# --- DEBE anonimizar --------------------------------------------------------

@pytest.mark.parametrize(
    "entrada,esperado",
    [
        ("HC: 2264097 del paciente", "[DOCUMENTO]"),
        ("DOCUMENTO 884710502", "[DOCUMENTO]"),
        ("identificado con CC 16789234", "[DOCUMENTO]"),
        ("registro 2246287 en el sistema", "[NUMERO_ID]"),
        ("remitido del HOSPITAL SAN JUAN DE DIOS", "[INSTITUCION]"),
        ("traslado a Hospital Ernesto Gil Trujillo", "[INSTITUCION]"),
        ("valorado en CLINICA AURORA", "[INSTITUCION]"),
        ("procede del Hospital Integrado de Jamundi", "[INSTITUCION]"),
        ("ingresa al HUC por trauma", "[INSTITUCION]"),
        ("comentado con dr. moncada quien considera", "[MEDICO]"),
        ("valorado por el Dr. Villalobos (neurocirujano)", "[MEDICO]"),
        ("toracotomia 09 dic 2014 sin respuesta", "[FECHA]"),
        ("cirugia el 06/12/2014 sin complicaciones", "[FECHA]"),
        ("control el 15 de septiembre", "[FECHA]"),
        ("ingreso a las 18:47 horas", "[HORA]"),
        ("accidente en el ano 2013 previo", "[ANIO]"),
        ("PACIENTE DE 93 ANOS que sufre caida", "[EDAD:90+]"),
        ("paciente de 45 anos con trauma", "[EDAD:40-49]"),
        ("menor de 7 anos con fractura", "[EDAD:5-9]"),
        ("lactante de 3 meses con TCE", "[EDAD:<1 año]"),
        ("remitido de Jamundi por trauma", "[MUNICIPIO]"),
        ("procedente de Puerto Tejada", "[MUNICIPIO]"),
        # erratas de digitación del tipo que aparece en el corpus
        ("intervenido el 15/058/2019 en quirofano", "[FECHA]"),
        ("11-jun2014 hemocultivo positivo", "[FECHA]"),
        ("remitido del hospital sann Juan de Dios", "[INSTITUCION]"),
        ("consulta al HOSPITAL SAN RAFAEL por dolor", "[INSTITUCION]"),
        # institución sin trigger
        ("es llevado al Rayito de Sol donde le realizan curacion", "[INSTITUCION]"),
        ("remitido de Los Almendros", "[INSTITUCION]"),
        # regla abierta solo para 'hospital'
        ("remitido del hospital Hernando Villota", "[INSTITUCION]"),
    ],
)
def test_anonimiza(anon, entrada, esperado):
    assert esperado in ano(anon, entrada), f"no anonimizó: {entrada!r}"


# --- NO debe tocar (falsos positivos observados en el corpus) ---------------------

@pytest.mark.parametrize(
    "entrada,debe_conservar",
    [
        # 'clínica' como adjetivo: 100+ apariciones en el corpus
        ("evolucion clinica satisfactoria, se da salida", "clinica satisfactoria"),
        ("paciente con clinica estable", "clinica estable"),
        ("sin evidencia clinica ni radiologica de fractura", "clinica ni radiologica"),
        ("remitido de hospital de periferia", "periferia"),
        # 'dr' = derecho, no doctor
        ("orificio de salida en antebrazo dr. sangrado activo", "antebrazo dr"),
        ("sin dr pero con disco optico grande", "sin dr pero"),
        # cifras clínicas que parecen identificadores o fechas
        ("Glasgow 13/15 al ingreso", "13/15"),
        ("TA 120/80 mmHg", "120/80"),
        ("adrenalina 1:1000 subcutanea", "1:1000"),
        ("relacion I:E de 1:2", "1:2"),
        ("hemotorax izquierdo de 2000 cc", "2000 cc"),
        ("herida de 3 cm en region parietal", "3 cm"),
        ("se administran 1000 ml de cristaloide", "1000 ml"),
        # tiempo relativo != edad
        ("hace 3 dias sufre caida de su propia altura", "hace 3 dias"),
        ("control en 8 dias por consulta externa", "en 8 dias"),
        ("retirar puntos en 5 dias", "en 5 dias"),
        # institución que contiene un municipio: no debe partirse
        ("remitido de HOSPITAL ERNESTO GIL TRUJILLO", "[INSTITUCION]"),
        # 'hospital + genérico' no es una IPS nombrada
        ("remitido de hospital de menor complejidad", "menor complejidad"),
        ("trasladado a hospital de mayor nivel", "mayor"),
        ("consulta al hospital mas cercano", "cercano"),
        # epónimos quirúrgicos: apellidos reales que no son PHI
        ("se realiza incision de Kocher", "Kocher"),
        ("signo de Blumberg positivo", "Blumberg"),
    ],
)
def test_no_toca(anon, entrada, debe_conservar):
    salida = ano(anon, entrada)
    assert debe_conservar.lower() in salida.lower(), (
        f"falso positivo: {entrada!r} -> {salida!r}"
    )


def test_trujillo_institucion_no_es_municipio(anon):
    salida = ano(anon, "remitido de HOSPITAL ERNESTO GIL TRUJILLO por trauma")
    assert "[MUNICIPIO]" not in salida
    assert "[INSTITUCION]" in salida


def test_trujillo_solo_si_es_municipio(anon):
    salida = ano(anon, "paciente procedente del municipio de Trujillo")
    assert "[MUNICIPIO]" in salida


def test_log_registra_reemplazos(anon):
    r = anon.anonimizar("cirugia el 06/12/2014 en CLINICA AURORA", record_id="1", campo="x")
    reglas = {x.regla for x in r.reemplazos}
    assert {"fecha", "institucion"} <= reglas
    assert all(x.record_id == "1" for x in r.reemplazos)


def test_tolerancia_a_tildes(anon):
    assert "[INSTITUCION]" in ano(anon, "remitido de CLÍNICA AURORA")
    assert "[MUNICIPIO]" in ano(anon, "procedente de Jamundí")


def test_normalizacion_comillas():
    assert normalizar_texto('"""SE CAYO CORRIENDO"""') == "SE CAYO CORRIENDO"
    assert normalizar_texto('"2HERIDA POR ARMA CONTUDENTE""') == "HERIDA POR ARMA CONTUDENTE"
    assert normalizar_texto("dolor   en    region\n\n\n\nlumbar") == "dolor en region\n\nlumbar"


def test_idempotencia(anon):
    t = "paciente de 45 anos remitido de Jamundi el 06/12/2014 por el dr. Palacios"
    una = ano(anon, t)
    dos = ano(anon, una)
    assert una == dos, "anonimizar dos veces debe dar el mismo resultado"
