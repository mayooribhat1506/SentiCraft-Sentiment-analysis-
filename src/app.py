"""
app.py
------
Streamlit web application for the sentiment analysis tool.
Supports single text analysis and bulk CSV file uploads with robust PDF report generation.
"""

import os
import sys
import io
import pandas as pd
from fpdf import FPDF
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Page configuration – must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght=300;400;500;600&family=JetBrains+Mono:wght=400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0f1117; color: #e8e8e8; }

    .hero { text-align: center; padding: 2.5rem 0 1.5rem; }
    .hero h1 { font-family: 'DM Serif Display', serif; font-size: 3rem; letter-spacing: -0.02em; color: #f0f0f0; margin: 0 0 0.35rem; line-height: 1.1; }
    .hero .tagline { font-size: 1rem; color: #6b7280; font-weight: 300; letter-spacing: 0.04em; text-transform: uppercase; }
    .rule { border: none; border-top: 1px solid #1f2937; margin: 1.5rem 0 2rem; }

    textarea { background: #161b27 !important; border: 1.5px solid #2d3748 !important; border-radius: 10px !important; color: #e2e8f0 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.97rem !important; line-height: 1.65 !important; padding: 0.9rem 1rem !important; transition: border-color 0.2s; }
    textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; }

    .stButton > button { background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%); color: #ffffff; font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.03em; border: none; border-radius: 8px; padding: 0.65rem 2.2rem; width: 100%; cursor: pointer; transition: opacity 0.2s, transform 0.1s; }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

    .result-card { border-radius: 14px; padding: 1.8rem 2rem; margin-top: 1.8rem; animation: fadeUp 0.4s ease; }
    .result-card.positive { background: linear-gradient(135deg, #052e16 0%, #14532d 100%); border: 1px solid #166534; }
    .result-card.negative { background: linear-gradient(135deg, #2d0a0a 0%, #450a0a 100%); border: 1px solid #7f1d1d; }
    .result-card.neutral  { background: linear-gradient(135deg, #111827 0%, #1f2937 100%); border: 1px solid #374151; }

    .result-emoji { font-size: 2.8rem; line-height: 1; margin-bottom: 0.4rem; }
    .result-label { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #f9fafb; margin: 0.2rem 0; text-transform: capitalize; }
    .result-confidence { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #9ca3af; margin-top: 0.15rem; }

    .prob-section { margin-top: 1.4rem; border-top: 1px solid rgba(255,255,255,0.07); padding-top: 1.2rem; }
    .prob-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.7rem; font-weight: 500; }
    .bar-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.55rem; }
    .bar-name { font-size: 0.82rem; color: #d1d5db; width: 6.5rem; text-transform: capitalize; flex-shrink: 0; }
    .bar-track { flex: 1; background: rgba(255,255,255,0.07); border-radius: 999px; height: 7px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; }
    .bar-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #9ca3af; width: 3.5rem; text-align: right; flex-shrink: 0; }

    .error-box { background: #2d0a0a; border: 1px solid #7f1d1d; border-radius: 10px; padding: 1rem 1.2rem; color: #fca5a5; font-size: 0.9rem; margin-top: 1.2rem; }
    .footer { text-align: center; color: #374151; font-size: 0.78rem; padding: 2.5rem 0 1rem; letter-spacing: 0.03em; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LABEL_META: dict = {
    "positive": {"emoji": "😄", "css": "positive", "bar": "#22c55e"},
    "negative": {"emoji": "😞", "css": "negative", "bar": "#ef4444"},
    "neutral":  {"emoji": "😐", "css": "neutral",  "bar": "#6b7280"},
}

def _meta(label: str) -> dict:
    return LABEL_META.get(label.lower(), {"emoji": "🔍", "css": "neutral", "bar": "#6366f1"})


# ---------------------------------------------------------------------------
# PDF Generation — FIXED
# Bug 1: BytesIO cursor was never reset to 0 before download → empty file
# Bug 2: pdf.output() sometimes returns bytearray not bytes → coerce safely
# ---------------------------------------------------------------------------
class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 110, 120)
        self.cell(0, 10, "SENTIMENT ANALYSER EXECUTIVE REPORT", border=0, ln=1, align="R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def generate_pdf_report(df, text_col, label_col, summary_metrics) -> bytes:
    pdf = PDFReport()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 15, "Executive Document Sentiment Summary", ln=1)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "Generated via Fine-Tuned DistilBERT Transformer Pipeline", ln=1)
    pdf.ln(6)

    # KPI Box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(10, pdf.get_y(), 190, 28, style="FD")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "DOCUMENT DISTRIBUTION METRICS:", ln=1)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(15, 23, 42)
    metrics_str = (
        f"Total Evaluated Rows: {summary_metrics['total']}  |  "
        f"Positive: {summary_metrics['pos_pct']}%  |  "
        f"Negative: {summary_metrics['neg_pct']}%  |  "
        f"Neutral: {summary_metrics['neutral_pct']}%"
    )
    pdf.cell(0, 8, metrics_str, ln=1)
    pdf.ln(14)

    # Table header
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(145, 10, " Text Segment", border=1, fill=True)
    pdf.cell(45, 10, " Sentiment", border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for _, row in df.iterrows():
        snippet = str(row[text_col]).strip().encode("ascii", "ignore").decode("ascii")
        if len(snippet) > 85:
            snippet = snippet[:82] + "..."

        sentiment = str(row[label_col]).upper()

        # Auto page break
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(241, 245, 249)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(145, 10, " Text Segment", border=1, fill=True)
            pdf.cell(45, 10, " Sentiment", border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 9)

        pdf.set_text_color(30, 41, 59)
        pdf.cell(145, 8, f" {snippet}", border=1)

        if "POSITIVE" in sentiment:
            pdf.set_text_color(22, 163, 74)
        elif "NEGATIVE" in sentiment:
            pdf.set_text_color(220, 38, 38)
        else:
            pdf.set_text_color(100, 116, 139)

        pdf.cell(45, 8, f" {sentiment}", border=1, align="C")
        pdf.set_text_color(30, 41, 59)
        pdf.ln()

    # Use a temp file — most compatible across all fpdf2 versions
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp_path = tmp.name
    pdf.output(tmp_path)
    with open(tmp_path, 'rb') as f:
        pdf_bytes = f.read()
    os.remove(tmp_path)
    return pdf_bytes


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Sentiment Analyser</h1>
            <p class="tagline">Production Transformer · 3-Way Deep Learning Engine</p>
        </div>
        <hr class="rule">
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["✨ Single Evaluation", "📂 Bulk CSV Processing"])

    # ── TAB 1: SINGLE ──────────────────────────────────────────────────────
    with tab1:
        user_text = st.text_area(
            label="Text to analyse",
            placeholder="Paste a review, customer statement, or complex text snippet here...",
            height=140,
            key="single_text_input",
            label_visibility="collapsed",
        )

        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            analyse_clicked = st.button("Analyse Sentence Context", use_container_width=True)

        if analyse_clicked:
            if not user_text.strip():
                st.markdown('<div class="error-box">Please enter some text before analysing.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Running transformer inference..."):
                    try:
                        from predict import get_sentiment_prediction
                        res = get_sentiment_prediction(user_text)
                        meta = _meta(res["label"])

                        prob_rows = ""
                        if isinstance(res["probabilities"], dict):
                            for cls, prob in sorted(res["probabilities"].items(), key=lambda x: -x[1]):
                                pct = prob * 100
                                bar_color = _meta(cls)["bar"]
                                prob_rows += f"""
                                <div class="bar-row">
                                    <span class="bar-name">{cls}</span>
                                    <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div></div>
                                    <span class="bar-pct">{pct:.1f}%</span>
                                </div>"""
                            prob_bars_html = f'<div class="prob-section"><div class="prob-label">Class probabilities</div>{prob_rows}</div>'
                        else:
                            prob_bars_html = ""

                        st.markdown(
                            f"""
                            <div class="result-card {meta['css']}">
                                <div class="result-emoji">{meta['emoji']}</div>
                                <div class="result-label">{res['label']}</div>
                                <div class="result-confidence">confidence &nbsp;·&nbsp; {res['confidence']}</div>
                                {prob_bars_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.markdown(f'<div class="error-box">Inference error: {e}</div>', unsafe_allow_html=True)

    # ── TAB 2: BULK CSV ────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "<p style='color:#9ca3af;font-size:0.92rem;margin-bottom:1rem;'>"
            "Upload a CSV file. The engine will classify every row and export a PDF report.</p>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                bulk_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

                possible_text_cols = [c for c in bulk_df.columns if "text" in str(c).lower() or "review" in str(c).lower()]
                text_column = possible_text_cols[0] if possible_text_cols else bulk_df.columns[0]

                st.success(f"File loaded. Processing column: '{text_column}'")

                if st.button("Generate Analysis & PDF Report", use_container_width=True):
                    with st.spinner("Classifying rows..."):
                        from predict import get_sentiment_prediction

                        predictions = []
                        progress = st.progress(0)
                        total = len(bulk_df)

                        for i, (_, row) in enumerate(bulk_df.iterrows()):
                            res = get_sentiment_prediction(str(row[text_column]))
                            predictions.append(res["label"])
                            progress.progress((i + 1) / total)

                        bulk_df["Predicted_Sentiment"] = predictions

                        total_rows = len(bulk_df)
                        counts = bulk_df["Predicted_Sentiment"].value_counts()
                        summary_metrics = {
                            "total":       total_rows,
                            "pos_pct":     round((counts.get("positive", 0) / total_rows) * 100, 1),
                            "neg_pct":     round((counts.get("negative", 0) / total_rows) * 100, 1),
                            "neutral_pct": round((counts.get("neutral",  0) / total_rows) * 100, 1),
                        }

                        # Summary metrics
                        st.markdown("### Summary")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("😊 Positive", f"{summary_metrics['pos_pct']}%")
                        c2.metric("😞 Negative", f"{summary_metrics['neg_pct']}%")
                        c3.metric("😐 Neutral",  f"{summary_metrics['neutral_pct']}%")

                        st.markdown("### Preview (first 10 rows)")
                        st.dataframe(bulk_df[[text_column, "Predicted_Sentiment"]].head(10), use_container_width=True)

                        # ── Generate PDF and hand raw bytes to Streamlit ──
                        pdf_bytes = generate_pdf_report(
                            bulk_df, text_column, "Predicted_Sentiment", summary_metrics
                        )

                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,           # pure bytes — no BytesIO wrapper needed
                            file_name="Sentiment_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

            except Exception as e:
                st.markdown(f'<div class="error-box">Error: {e}</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built with Fine-Tuned DistilBERT · Streamlit · FPDF2</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()