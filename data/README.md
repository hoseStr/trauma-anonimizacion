# data/

Aquí va el CSV de entrada, `trauma.csv`. Git ignora todo lo que hay en esta
carpeta salvo este archivo.

Formato esperado: separado por `;`, UTF-8, con estas columnas:

| Columna | Contenido |
|---|---|
| `TraumaRegisterRecordID` | identificador único del registro |
| `Unidad de edad` | `Años`, `Meses` o vacío |
| `Motivo de Consulta` | texto libre |
| `Descripción del examen físico` | texto libre |
| `Detalles de hospitalización` | texto libre (evolución, manejo y desenlace) |

También se puede pasar otra ruta: `python run_pipeline.py --entrada ruta/al/archivo.csv`.
