import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from preprocess import clean_text

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "distilbert-base-uncased" # Base configuration ensures custom weights compile smoothly
MAX_LEN          = 128
BATCH_SIZE       = 32
EPOCHS           = 3    # Optimized execution matrix
MAX_STEPS        = 600  # Strict calculation ceiling to secure rapid compilation on Mac
LR               = 3e-5
WARMUP_STEPS     = 50
MODELS_DIR       = "models"
DATA_PATH        = "data/test_reviews.csv"
SAMPLE_PER_CLASS = 4000  # Streamlined sample size per class for faster training
DEVICE           = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Training/Evaluation Steps ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, current_step, max_steps):
    model.train()
    total_loss = 0
    steps_executed = 0
    
    for batch in loader:
        if current_step >= max_steps:
            break
            
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        current_step += 1
        steps_executed += 1
        
        if current_step % 50 == 0:
            logger.info(f"  Step {current_step}/{max_steps} | Current Loss = {loss.item():.4f}")
            
    return total_loss / max(1, steps_executed), current_step


def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_b       = batch["labels"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)


def balance_classes(df, sample_n):
    frames = []
    for label in df["sentiment"].unique():
        grp = df[df["sentiment"] == label]
        frames.append(grp.sample(min(len(grp), sample_n), random_state=42))
    return pd.concat(frames, ignore_index=True)


def load_external_neutral(n=4000):
    frames = []
    
    # Source 1: mteb/tweet_sentiment_multilingual
    try:
        logger.info("Trying mteb/tweet_sentiment_multilingual...")
        ds = load_dataset("mteb/tweet_sentiment_multilingual", "english", trust_remote_code=True)
        df_tmp = pd.concat([pd.DataFrame(ds[s]) for s in ds.keys()], ignore_index=True)
        neutral = df_tmp[df_tmp["label"] == 1][["text"]].copy()
        neutral["sentiment"] = "neutral"
        frames.append(neutral)
    except Exception as e:
        logger.warning(f"  mteb/tweet_sentiment_multilingual failed: {e}")

    # Source 2: SetFit/sst5
    try:
        logger.info("Trying SetFit/sst5...")
        ds2 = load_dataset("SetFit/sst5", trust_remote_code=True)
        df_tmp2 = pd.concat([pd.DataFrame(ds2[s]) for s in ds2.keys()], ignore_index=True)
        neutral2 = df_tmp2[df_tmp2["label"] == 2][["text"]].copy()
        neutral2["sentiment"] = "neutral"
        frames.append(neutral2)
    except Exception as e:
        logger.warning(f"  SetFit/sst5 failed: {e}")

    # Source 3: Synthetic fallback
    logger.info("Injecting synthetic baseline structural variants...")
    synthetic_texts = [
        "The item was received and placed on the shelf.",
        "The document was reviewed by the team on Tuesday.",
        "She completed the registration form and submitted it.",
        "The meeting lasted approximately one hour.",
        "He read the instruction manual before starting.",
        "The package contains three separate components.",
        "The store is located on the main street.",
        "The report was filed before the end of the month.",
        "The temperature remained steady throughout the day.",
        "She noted the details in her notebook.",
        "The system restarted after the update was applied.",
        "He checked the calendar before confirming the date.",
        "The delivery was made to the correct address.",
        "The invoice was sent to the billing department.",
        "The project timeline was updated last week.",
        "She forwarded the email to the relevant team.",
        "The file was saved in the default folder.",
        "He attended the briefing on Wednesday morning.",
        "The data was exported in CSV format.",
        "The office opens at eight thirty in the morning.",
    ]
    repeats = (n // len(synthetic_texts)) + 1
    synthetic = pd.DataFrame({
        "text":      (synthetic_texts * repeats)[:n],
        "sentiment": "neutral"
    })
    frames.append(synthetic)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    combined = combined.sample(min(n, len(combined)), random_state=42).reset_index(drop=True)
    return combined[["text", "sentiment"]]


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Using device: {DEVICE}")

    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset missing at {DATA_PATH}")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df.columns = df.columns.astype(str).str.strip().str.lower()

    text_col  = "text"      if "text"      in df.columns else df.columns[-1]
    label_col = "sentiment" if "sentiment" in df.columns else df.columns[0]

    if df[label_col].astype(str).str.len().mean() > df[text_col].astype(str).str.len().mean():
        text_col, label_col = label_col, text_col

    df = df.dropna(subset=[text_col, label_col])

    label_map = {
        0: "negative", 2: "neutral", 4: "positive",
        "0": "negative", "2": "neutral", "4": "positive",
        "negative": "negative", "neutral": "neutral", "positive": "positive",
        "neg": "negative", "neu": "neutral", "pos": "positive",
    }
    
    df = pd.DataFrame({
        "text":      df[text_col].values,
        "sentiment": df[label_col].map(lambda x: label_map.get(x, str(x).strip().lower())).values,
    })
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)

    # 2. Pull external neutral datasets if class distributions drop
    neutral_count = (df["sentiment"] == "neutral").sum()
    if neutral_count < SAMPLE_PER_CLASS:
        needed = SAMPLE_PER_CLASS - neutral_count
        logger.info(f"Only {neutral_count} neutral entries detected. Pulling balance...")
        ext = load_external_neutral(n=needed)
        df  = pd.concat([df, ext], ignore_index=True)

    # 3. Balance classes safely
    sample_n = min(df["sentiment"].value_counts().min(), SAMPLE_PER_CLASS)
    df = balance_classes(df, sample_n)

    # 4. Clean text
    logger.info("Cleaning text...")
    df["cleaned_text"] = df["text"].apply(lambda x: clean_text(str(x)))
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)

    # 5. Encode labels (CRITICAL FIX: Explicit indexing alignment)
    le = LabelEncoder()
    le.fit(["negative", "neutral", "positive"])  
    df["label_id"] = le.transform(df["sentiment"])
    num_classes = len(le.classes_)
    logger.info(f"Synchronized class alignment locked: {le.classes_}")

    # 6. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"].tolist(),
        df["label_id"].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df["label_id"],
    )

    # 7. Tokenize Configuration
    tokenizer     = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset  = SentimentDataset(X_test,  y_test,  tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 8. Model Allocation
    logger.info(f"Initializing Sequence Classifier Architecture: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    )
    model.to(DEVICE)

    # 9. Optimizer Optimization Matrix
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # 10. Core Loop Fine-Tuning Execution
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(1, EPOCHS + 1):
        if global_step >= MAX_STEPS:
            logger.info("Maximum optimization step ceiling reached. Terminating fine-tuning.")
            break
            
        logger.info(f"── Epoch {epoch}/{EPOCHS} ──────────────────────────")
        train_loss, global_step = train_epoch(model, train_loader, optimizer, scheduler, global_step, MAX_STEPS)
        preds, labels = eval_epoch(model, test_loader)
        acc = (preds == labels).mean()
        logger.info(f"Epoch {epoch} done | avg_batch_loss={train_loss:.4f} | validation_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(os.path.join(MODELS_DIR, "distilbert_sentiment"))
            tokenizer.save_pretrained(os.path.join(MODELS_DIR, "distilbert_sentiment"))
            logger.info(f"  ✓ Preserved top performing checkpoint weights (val_acc={best_acc:.4f})")

    # 11. Print Final Classification Breakdown Matrix
    preds, labels = eval_epoch(model, test_loader)
    print("\n" + "=" * 60)
    print("              3-WAY CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=le.classes_, digits=4))
    print("=" * 60 + "\n")

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    logger.info(f"All full-stack training artifacts saved. System Ready! Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()