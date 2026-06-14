# SentiCraft-Sentiment-analysis-
3-Way Deep Learning Engine: Classifies text data seamlessly into positive, negative, or neutral categories using custom-weighted transformer pipelines.
# 🚀 SentiCraft: Transformer-Powered Sentiment Engine

SentiCraft is a full-stack, production-grade Natural Language Processing (NLP) application designed for granular sentiment analysis. Powered by a fine-tuned **DistilBERT** transformer model, the system goes beyond simple document-level classification by breaking down text into clause-level sub-sentiments. It features an elegant, custom-styled dark-themed **Streamlit** dashboard that supports both real-time text evaluation and bulk CSV data processing with automated **Executive PDF Report** generation.

---

## 📂 Project Structure

The project follows a clean, modular architecture separating the core machine learning logic from the presentation layer:

```text
SentiCraft/
├── .gitignore               # Tells Git to ignore machine-specific artifacts & large weights
├── README.md                # Project documentation
├── requirements.txt         # Package dependencies
├── reviews.csv              # Local testing dataset for bulk processing
│
├── .vscode/                 # Editor configurations
│   └── settings.json        # Global environment paths
│
└── src/                     # Core execution scripts (Source package)
    ├── __init__.py          # Package initialization marker
    ├── preprocess.py        # Text cleaning and Unicode normalization
    ├── train.py             # DistilBERT training, balancing, and fine-tuning loop
    ├── predict.py           # Lazy-loader inference engine & sub-sentiment parser
    └── app.py               # Streamlit application UI & PDF compilation report
```

---

## ✨ Key Features

* **3-Way Classification Engine:** Classifies text data safely into `positive`, `negative`, or `neutral` sentiments via custom-weighted transformer pipelines.

* **Clause-Level Sub-Sentiment Tracking:** Implements logical regex partitioning to isolate and highlight contrasting emotional tones, capturing positive highlights alongside negative caveats within complex sentence structures.

* **Bulk CSV Processing Pipeline:** Dynamically auto-detects text target columns from uploaded files, processes rows sequentially with an active progress bar, and visualizes instant distribution metrics.

* **Executive PDF Summaries:** Generates polished, corporate-ready KPI report documents containing distribution statistics and structured preview tables ready for immediate download.

* **Tailored Text Normalization:** Cleans string inputs using NFKD Unicode normalization and URL/HTML stripping while intentionally keeping natural sentence patterns intact for BERT's subword tokenization matrix.

* **Apple Silicon Optimization:** The underlying training pipeline leverages Apple Silicon GPU acceleration (`torch.device("mps")`) for hyper-fast execution matrices on Mac systems.

---

## 🛠️ Tech Stack & Architecture

| Component | Technology / Library | Role |
| --- | --- | --- |
| **Model Base** | `DistilBertForSequenceClassification` | Lean transformer variant ensuring rapid execution and minimal latency. |
| **Framework** | `PyTorch` | Model allocation, training optimization, linear learning rate scheduling, and weight backpropagation. |
| **Interface** | `Streamlit` | Production-ready web UI complete with fully responsive custom CSS styling. |
| **Reporting** | `FPDF2` | Direct runtime generation of byte-streamed PDF executive documents. |
| **Data Engine** | `pandas`, `scikit-learn`, `joblib` | Label encoding alignment, stratified dataset splitting, and baseline class balancing. |

---

## 🚀 Getting Started

### 1. Clone the Repository & Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/SentiCraft.git
cd SentiCraft
pip install -r requirements.txt
```

### 2. Run the Training Pipeline

To train the model on your local data, balance the classes with external structural variants, and save your fine-tuned weights locally, run:

```bash
python src/train.py
```

### 3. Launch the Web Application

Once the training loop completes and generates your model artifacts under `models/distilbert_sentiment/`, boot up your Streamlit dashboard using:

```bash
streamlit run src/app.py
```

---

## 🔒 Git Configurations & Model Weights

To keep the codebase lean and clean, compiled local machine artifacts and deep learning checkpoints are deliberately excluded from version control via `.gitignore`:

* **`__pycache__/`**: Omitted to prevent machine-specific bytecode overhead and history clutter.
* **`models/`**: Excluded because heavy deep learning transformer weights exceed GitHub's file size thresholds. Files remain strictly local.


