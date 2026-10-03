"""
Capa C — NER estadístico como red de seguridad.

NO reemplaza nada. Corre spaCy `es_core_news_lg` sobre el texto YA anonimizado
y reporta entidades PER / LOC / ORG que las capas A y B dejaron pasar, para
revisión manual.

Por qué no reemplaza — medido sobre este corpus, sin filtros:
    3.000 documentos -> 74.581 entidades, 26.078 únicas.
    'HERIDA' como ORG, 'TRAUMA' como LOC, 'TRANSITO' como PER.
El NER en español está entrenado sobre prosa periodística; el texto clínico en
MAYÚSCULAS lo descalabra. Automatizar el reemplazo aquí destruiría más corpus
del que protege.

Tres filtros lo vuelven revisable a mano:

 1. Enmascarado de placeholders. `[FECHA]`, `[INSTITUCION]` y la nota de
    truncamiento se sustituyen por un token neutro antes del NER; si no,
    spaCy los parte y los reporta como entidades.

 2. Frecuencia en el corpus. Un nombre propio real es RARO: aparece en un
    puñado de documentos. El vocabulario clínico aparece en cientos. Se
    descarta toda entidad cuyo token más común supere `--max-df` documentos.
    Este filtro se calcula del propio corpus, no de una lista escrita a mano,
    así que se adapta si cambia el dataset.

 3. Casing. En documentos íntegramente en MAYÚSCULAS el NER pierde su señal
    principal (la capitalización) y su precisión se desploma. Por defecto solo
    se analizan los documentos con casing mixto o minúscula; `--incluir-mayus`
    los añade, asumiendo mucho más ruido.

Uso:
    python -m src.ner_review
    python -m src.ner_review --top 3000 --max-df 20
    python -m src.ner_review --incluir-mayus
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import pandas as pd

MODELO = "es_core_news_lg"
ETIQUETAS = {"PER", "LOC", "ORG"}

# Placeholders y nota de truncamiento: fuera antes del NER.
RX_PLACEHOLDER = re.compile(r"\[[^\]]{0,120}\]")
RX_TOKEN_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{3,}")

# Epónimos y escalas: apellidos reales pero que aquí NO son PHI.
EPONIMOS = {
    "glasgow", "blumberg", "murphy", "babinski", "romberg", "chvostek", "trousseau",
    "kernig", "brudzinski", "tinel", "phalen", "lachman", "mcmurray", "finkelstein",
    "colles", "smith", "monteggia", "galeazzi", "bennett", "jones", "salter", "harris",
    "weber", "schatzker", "gustilo", "anderson", "hinchey", "ranson", "child", "pugh",
    "apgar", "silverman", "tanner", "rankin", "lund", "browder", "wallace", "morel",
    "lavallee", "tillaux", "maisonneuve", "pilon", "chance", "jefferson", "hangman",
    "fort", "lefort", "waters", "caldwell", "towne", "asia", "denver", "marshall",
    "rotterdam", "fisher", "hunt", "hess", "mallampati", "cormack", "lehane",
    "kocher", "hartmann", "pfannenstiel", "mcburney", "kehr", "penrose", "foley",
    "pratt", "jackson", "gigli", "esmarch", "sengstaken", "blakemore", "seldinger",
    "allen", "adson", "kelly", "mayo", "metzenbaum", "satinsky", "pringle",
}


def _cargar_modelo():
    try:
        import spacy
    except ImportError:
        sys.exit(
            "spaCy no está instalado.\n"
            "  pip install spacy\n"
            f"  python -m spacy download {MODELO}"
        )
    try:
        return spacy.load(MODELO, exclude=["lemmatizer", "tagger", "attribute_ruler", "parser"])
    except OSError:
        sys.exit(f"Falta el modelo. Ejecuta:  python -m spacy download {MODELO}")


def enmascarar(texto: str) -> str:
    """Sustituye placeholders por un token neutro que el NER ignora."""
    return RX_PLACEHOLDER.sub(" XX ", texto)


def es_mayusculas(texto: str, umbral: float = 0.8) -> bool:
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return False
    return sum(1 for c in letras if c.isupper()) / len(letras) > umbral


def frecuencia_documental(textos: pd.Series) -> collections.Counter:
    """En cuántos documentos aparece cada token. Base del filtro de rareza."""
    df = collections.Counter()
    for t in textos:
        df.update({w.lower() for w in RX_TOKEN_PALABRA.findall(t)})
    return df


def revisar(
    textos: pd.Series,
    ids: pd.Series,
    df_tokens: collections.Counter,
    max_df: int = 20,
    batch_size: int = 64,
    n_process: int = 1,
) -> pd.DataFrame:
    nlp = _cargar_modelo()
    filas: list[dict] = []
    enmascarados = [enmascarar(t) for t in textos]

    for rid, doc in zip(ids.tolist(), nlp.pipe(enmascarados, batch_size=batch_size,
                                               n_process=n_process)):
        for ent in doc.ents:
            if ent.label_ not in ETIQUETAS:
                continue
            txt = ent.text.strip(" .,:;()[]\n")
            if len(txt) < 3 or txt.isdigit() or txt.upper() == "XX":
                continue

            tokens = [w.lower() for w in RX_TOKEN_PALABRA.findall(txt)]
            if not tokens:
                continue
            if any(w in EPONIMOS for w in tokens):
                continue

            # filtro de rareza: el vocabulario clínico es frecuente, un nombre no
            df_max = max(df_tokens.get(w, 0) for w in tokens)
            if df_max > max_df:
                continue

            filas.append(
                {
                    "record_id": rid,
                    "label": ent.label_,
                    "entidad": txt,
                    "df_corpus": df_max,
                    "contexto": doc.text[max(0, ent.start_char - 70): ent.end_char + 70]
                    .replace("\n", " "),
                }
            )

    cols = ["record_id", "label", "entidad", "df_corpus", "contexto"]
    df = pd.DataFrame(filas, columns=cols)
    return df.sort_values("df_corpus") if not df.empty else df


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["entidad", "label"], as_index=False)
        .agg(n=("record_id", "size"), df_corpus=("df_corpus", "first"),
             ejemplo=("contexto", "first"))
        .sort_values(["n", "entidad"], ascending=[False, True])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Capa C: NER de revisión (no reemplaza)")
    ap.add_argument("--corpus", default="outputs/corpus.parquet")
    ap.add_argument("--salida", default="reports/ner_revision.csv")
    ap.add_argument("--top", type=int, default=0,
                    help="revisar solo los N documentos más largos (0 = todos)")
    ap.add_argument("--max-df", type=int, default=20,
                    help="descartar entidades cuyo token aparezca en más de N documentos")
    ap.add_argument("--incluir-mayus", action="store_true",
                    help="incluir documentos íntegramente en MAYÚSCULAS (mucho más ruido)")
    ap.add_argument("--n-process", type=int, default=1)
    args = ap.parse_args()

    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["indexable"]]
    df_tokens = frecuencia_documental(corpus["texto"])   # se calcula sobre TODO el corpus

    objetivo = corpus
    if not args.incluir_mayus:
        mixto = ~objetivo["texto"].map(es_mayusculas)
        print(f"Excluyendo {(~mixto).sum():,} documentos en MAYÚSCULAS "
              f"(el NER pierde precisión sin capitalización; usa --incluir-mayus para añadirlos)")
        objetivo = objetivo[mixto]
    if args.top:
        objetivo = objetivo.nlargest(args.top, "n_chars_doc")

    print(f"Analizando {len(objetivo):,} documentos con {MODELO} (max_df={args.max_df})...")
    det = revisar(objetivo["texto"], objetivo["record_id"], df_tokens,
                  max_df=args.max_df, n_process=args.n_process)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(salida, index=False, encoding="utf-8")
    res = resumen(det)
    res.to_csv(salida.with_name(salida.stem + "_resumen.csv"), index=False, encoding="utf-8")

    print(f"Entidades candidatas: {len(det):,}  |  únicas: {len(res):,}")
    print(f"-> {salida}")
    if not res.empty:
        print("\nTop 30 para revisión manual:")
        print(res.head(30)[["entidad", "label", "n", "df_corpus"]].to_string(index=False))


if __name__ == "__main__":
    main()
