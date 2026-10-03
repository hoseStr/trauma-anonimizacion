"""
Tests del filtro de calidad: qué casos llegan al índice del RAG.

Solo entran casos completos (los tres campos llenos) y sin truncar. Un caso
cortado o sin evolución es justo el que el generador tiende a completar
inventando. Correr con:  python -m pytest tests/ -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_corpus import construir_corpus, exportar  # noqa: E402
from src.normalize import ID_COL, TEXT_COLS, normalizar_dataframe  # noqa: E402
from src.quality import UMBRAL_TOPE, _es_truncado, marcar_calidad  # noqa: E402

MOTIVO, EXAMEN, HOSP = TEXT_COLS

COMPLETO = {
    MOTIVO: "SE CAYO DE SU PROPIA ALTURA",
    EXAMEN: "HERIDA DE 3 CM EN REGION FRONTAL, SIN SANGRADO ACTIVO.",
    HOSP: "SE REALIZA SUTURA Y SE DA SALIDA CON RECOMENDACIONES Y SIGNOS DE ALARMA",
}


def _relleno(n: int, final: str = ".") -> str:
    base = "PACIENTE CON EVOLUCION ESTABLE SIN COMPLICACIONES "
    return (base * (n // len(base) + 1))[: n - len(final)] + final


def _calidad(filas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{ID_COL: str(i + 1), "Unidad de edad": "Años", **f} for i, f in enumerate(filas)]
    )
    return marcar_calidad(normalizar_dataframe(df))


# --- detector de truncamiento ----------------------------------------------

@pytest.mark.parametrize("n", [UMBRAL_TOPE, 499, 500])
@pytest.mark.parametrize("col", [EXAMEN, HOSP])
def test_campo_con_tope_cerca_de_500_es_truncado(col, n):
    texto = _relleno(n)
    assert _es_truncado(texto, texto, col)


@pytest.mark.parametrize("col", [EXAMEN, HOSP])
def test_campo_con_tope_bajo_el_umbral_no_es_truncado(col):
    texto = _relleno(UMBRAL_TOPE - 1)
    assert not _es_truncado(texto, texto, col)


def test_motivo_largo_no_es_truncado():
    """El motivo de consulta no tiene tope en el export: llega a >5.000 caracteres."""
    texto = _relleno(600)
    assert not _es_truncado(texto, texto, MOTIVO)


@pytest.mark.parametrize(
    "texto",
    [
        "PACIENTE ESTABLE, SE DA SALIDA",
        "RECOMENDACIONES Y SIGNOS DE ALARMA",
        "PULSOS DISTALES PRESENTES SIN SIGNOS DE LESION VASCULAR",
        "tras rcp sin respuesta, fallece",
    ],
)
def test_final_sin_punto_no_es_truncado(texto):
    """Es la forma habitual de cerrar una nota: la heurística anterior las marcaba."""
    assert not _es_truncado(texto, texto, HOSP)


@pytest.mark.parametrize(
    "texto",
    [
        "PRESENTO PERDIDA DE CONOCIMIENTO Y AL RECUPERARLA SE",
        "HERIDAS SUTURADAS SIN SIGNOS DE INFECCIÓN O",
        "paciente con dolor en region lumbar con",
        "SE TRASLADA A LA UNIDAD DE",
        "evolucion satisfactoria, se remite para",
    ],
)
def test_final_en_palabra_funcional_es_truncado(texto):
    assert _es_truncado(texto, texto, EXAMEN)


# --- apto_rag ---------------------------------------------------------------

def test_caso_completo_es_apto():
    df = _calidad([COMPLETO])
    assert df["apto_rag"].iloc[0]
    assert df["quality_flags"].iloc[0] == []


@pytest.mark.parametrize("vacio", TEXT_COLS)
def test_campo_vacio_excluye_del_rag(vacio):
    df = _calidad([{**COMPLETO, vacio: None}])
    fila = df.iloc[0]
    assert fila["indexable"]
    assert fila["flag_incompleto"]
    assert fila["campos_vacios"] == [vacio]
    assert not fila["apto_rag"]
    assert "incompleto" in fila["quality_flags"]


def test_truncado_excluye_del_rag():
    df = _calidad([{**COMPLETO, HOSP: _relleno(500, final="PACIENTE REQU")}])
    fila = df.iloc[0]
    assert fila["indexable"]
    assert fila["flag_truncated"]
    assert fila["campos_truncados"] == [HOSP]
    assert not fila["apto_rag"]


def test_vacio_total_no_cuenta_como_incompleto():
    """Un registro sin texto ya es `empty`; `incompleto` es para los parciales."""
    df = _calidad([{MOTIVO: None, EXAMEN: None, HOSP: None}])
    assert df["flag_empty"].iloc[0]
    assert not df["flag_incompleto"].iloc[0]
    assert not df["apto_rag"].iloc[0]


# --- export -----------------------------------------------------------------

def _corpus(filas: list[dict]) -> pd.DataFrame:
    df = _calidad(filas)
    for col in TEXT_COLS:
        df[f"{col}_anon"] = df[f"{col}_clean"]
    df["mecanismo_lesion"] = "caida"
    df["region_anatomica"] = [["cara"]] * len(df)
    df["desenlace"] = "alta"
    df["abreviaturas"] = [[]] * len(df)
    df["n_reemplazos_phi"] = 0
    df["tipos_phi"] = [[]] * len(df)
    return construir_corpus(df)


def test_jsonl_solo_contiene_aptos(tmp_path):
    corpus = _corpus([
        COMPLETO,                                                  # 1: apto
        {**COMPLETO, HOSP: None},                                  # 2: incompleto
        {**COMPLETO, EXAMEN: _relleno(499)},                       # 3: truncado
        {**COMPLETO, MOTIVO: "ME CAI JUGANDO FUTBOL EN LA CANCHA"},  # 4: apto
    ])
    rutas = exportar(corpus, tmp_path)

    docs = [json.loads(l) for l in rutas["jsonl"].read_text(encoding="utf-8").splitlines()]
    assert [d["id"] for d in docs] == ["1", "4"]
    for d in docs:
        assert d["metadata"]["truncado"] is False
        for encabezado in ("Motivo de consulta:", "Examen físico:", "Evolución y hospitalización:"):
            assert encabezado in d["text"]

    excluidos = pd.read_csv(rutas["excluidos"], dtype=str)
    assert dict(zip(excluidos["record_id"], excluidos["motivo"])) == {
        "2": "incompleto", "3": "truncated",
    }


def test_ablacion_indexable_incluye_truncados_e_incompletos(tmp_path):
    corpus = _corpus([COMPLETO, {**COMPLETO, HOSP: None}, {**COMPLETO, EXAMEN: _relleno(499)}])
    rutas = exportar(corpus, tmp_path, columna_filtro="indexable")
    assert len(rutas["jsonl"].read_text(encoding="utf-8").splitlines()) == 3
