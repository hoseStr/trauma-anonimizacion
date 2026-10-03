# Diccionario de datos — `outputs/corpus.parquet`

Una fila por registro del CSV original. 27.157 filas.

| Columna | Tipo | Descripción |
|---|---|---|
| `record_id` | str | `TraumaRegisterRecordID` del CSV original. Único, sin nulos. |
| `texto` | str | Documento ensamblado y anonimizado. Es lo que se indexa. |
| `n_chars_doc` | int | Longitud del documento ensamblado. |
| `mecanismo_lesion` | str | `arma_de_fuego` · `cortopunzante` · `accidente_transito` · `caida` · `contundente` · `quemadura` · `mordedura` · `cuerpo_extrano` · `intoxicacion` · `autoinfligido` · `no_determinado` |
| `region_anatomica` | list[str] | Cero o más de: `craneo_encefalo`, `cara`, `ojo`, `cuello`, `torax`, `abdomen`, `pelvis`, `columna`, `miembro_superior`, `miembro_inferior` |
| `desenlace` | str | Por gravedad descendente: `fallecimiento` · `cirugia` · `uci` · `hospitalizacion` · `remision` · `alta` · `no_determinado` |
| `es_pediatrico` | bool | Derivado de `Unidad de edad == "Meses"`. |
| `unidad_edad` | str? | `anos` · `meses` · nulo. Única señal de edad que sobrevivió al export original. |
| `abreviaturas` | list[str] | Abreviaturas expandidas en este documento. |
| `quality_flags` | list[str] | `empty` · `too_short` · `truncated` · `incompleto` · `duplicate` |
| `truncado` | bool | Algún campo está cortado: examen físico u hospitalización con ≥480 caracteres (el export los cortó a 500), o un campo que termina en palabra funcional (`de`, `con`, `se`…). |
| `campos_truncados` | list[str] | Qué campos están truncados. |
| `campos_vacios` | list[str] | Qué campos de texto están vacíos (bandera `incompleto`). |
| `doc_largo` | bool | `n_chars_doc > 1500`; candidato a partirse por sección. |
| `indexable` | bool | `not (empty or too_short or duplicate)`. Incluye truncados e incompletos. |
| `apto_rag` | bool | `indexable and not truncado and not incompleto`. **Es lo que llega al índice.** |
| `n_reemplazos_phi` | int | Cuántos reemplazos de anonimización recibió el registro. |
| `tipos_phi` | list[str] | Reglas que dispararon: `fecha`, `hora`, `edad`, `institucion`, `municipio`, `medico`, `anio`, `documento`, … |
| `hash_texto` | str | SHA-256 (16 hex) del texto normalizado. Base de la deduplicación. |

## Placeholders en `texto`

| Placeholder | Reemplaza |
|---|---|
| `[FECHA]` | fecha absoluta en cualquier formato |
| `[HORA]` | hora del día |
| `[ANIO]` | año suelto sin unidad detrás |
| `[EDAD:n-m]`, `[EDAD:90+]`, `[EDAD:<1 año]`, `[EDAD:<1 mes]` | edad generalizada a rango |
| `[INSTITUCION]` | nombre de IPS |
| `[MUNICIPIO]` | municipio o ciudad |
| `[MEDICO]` | nombre o apellido de profesional |
| `[DOCUMENTO]` | cédula, tarjeta de identidad, número de historia clínica |
| `[NUMERO_ID]` | entero suelto de 7–12 dígitos sin unidad |
| `[TELEFONO]`, `[EMAIL]` | reglas definidas; cero apariciones en este corpus |

Los tiempos **relativos** ("hace 3 días", "control en 8 días", "a las 48 horas")
se conservan intactos: son clínicamente relevantes y no identifican a nadie.

## `outputs/corpus_indexable.jsonl`

Solo `apto_rag == True`. Un objeto por línea:

```json
{"id": "12345", "text": "Motivo de consulta: …", "metadata": {
  "mecanismo_lesion": "caida", "region_anatomica": ["pelvis"],
  "desenlace": "alta", "es_pediatrico": false,
  "truncado": false, "doc_largo": false, "n_chars": 612,
  "abreviaturas": []}}
```

`truncado` siempre es `false` en el corpus apto; se conserva para que los filtros
del RAG sigan funcionando igual.

## `outputs/excluidos_rag.csv`

Una fila por registro que no llegó al índice: `record_id`, `motivo` (`empty`,
`too_short`, `duplicate`, `truncated` o `incompleto`: el primero que aplica),
`campos_truncados` y `campos_vacios` (separados por `|`), `n_chars_doc`.

## `outputs/log_reemplazos.csv`

Auditoría completa. Una fila por reemplazo: `record_id`, `campo`, `regla`,
`original` (el texto que se reemplazó), `placeholder`, `inicio` (offset).
