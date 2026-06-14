import os
import sys
import re
import argparse
import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from preprocess import clean_text

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "distilbert_sentiment")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Lazy-load model once per process ─────────────────────────────────────────
_tokenizer = None
_model     = None
_le        = None

def _load_artifacts():
    global _tokenizer, _model, _le
    if _model is not None:
        return

    if not os.path.isdir(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Trained artifacts missing. Please run train.py first.")

    _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    _model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.to(DEVICE)
    _model.eval()
    _le = joblib.load(ENCODER_PATH)


# ── Sub-sentiment (clause-level) ──────────────────────────────────────────────
def _predict_single(text: str):
    """Return (label_str, prob_dict) for a single piece of text."""
    _load_artifacts()
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None, {}

    enc = _tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits

    probs      = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx   = int(torch.argmax(logits, dim=1).item())
    pred_label = _le.inverse_transform([pred_idx])[0]
    prob_dict  = {_le.inverse_transform([i])[0]: round(p, 4) for i, p in enumerate(probs)}

    return pred_label, prob_dict


def extract_sub_sentiments(text: str):
    clauses    = re.split(r'\b(?:but|however|yet|although|though)\b|[.!,;]', text, flags=re.IGNORECASE)
    highlights = {"positive_parts": [], "negative_parts": []}

    for clause in clauses:
        clause = clause.strip()
        if len(clause.split()) < 2:
            continue
        label, _ = _predict_single(clause)
        if label == "positive":
            highlights["positive_parts"].append(clause)
        elif label == "negative":
            highlights["negative_parts"].append(clause)

    return highlights


# ── Public API ────────────────────────────────────────────────────────────────
def get_sentiment_prediction(text: str) -> dict:
    _load_artifacts()
    label, prob_dict = _predict_single(text)
    confidence       = f"{round(prob_dict.get(label, 0) * 100, 2)}%"
    highlights       = extract_sub_sentiments(text)

    return {
        "label":         label,
        "confidence":    confidence,
        "probabilities": prob_dict,
        "highlights":    highlights,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BERT Sentiment Inference")
    parser.add_argument("--text", type=str, required=True, help="Raw text to classify")
    args = parser.parse_args()

    try:
        res = get_sentiment_prediction(args.text)
        print("\n" + "=" * 60)
        print(f"👉 INPUT TEXT   : \"{args.text}\"")
        print(f"📊 SENTIMENT    : {res['label'].upper()}  (Confidence: {res['confidence']})")
        print(f"📈 PROBABILITIES: {res['probabilities']}")
        if res["highlights"]["positive_parts"]:
            print(f"✅ POSITIVE PARTS: {res['highlights']['positive_parts']}")
        if res["highlights"]["negative_parts"]:
            print(f"❌ NEGATIVE PARTS: {res['highlights']['negative_parts']}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    main()