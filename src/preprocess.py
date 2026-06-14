import re
import unicodedata
import pandas as pd


def clean_text(text: str) -> str:
    """
    Light normalisation only — BERT's own tokenizer handles subword splitting.
    Do NOT lemmatize or strip stopwords: BERT needs full natural sentences.
    """
    if not text or pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Unicode normalisation
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Collapse excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text