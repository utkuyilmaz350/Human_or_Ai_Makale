from __future__ import annotations

from pathlib import Path

import streamlit as st

from human_or_ai.predict.predict import load_model, predict_one


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

MODEL_FILES = {
    "Logistic Regression": ARTIFACTS / "logreg.joblib",
    "Multinomial Naive Bayes": ARTIFACTS / "nb.joblib",
    "Linear SVC (Calibrated)": ARTIFACTS / "linear_svc_cal.joblib",
}


st.set_page_config(page_title="Human or AI", layout="wide")

st.title("Human or AI")
st.caption("Makale özetleri üzerinden metin tespiti")

text = st.text_area("Metni girin", height=220)

col1, col2 = st.columns([1, 2])

with col1:
    run = st.button("Tahmin Et", type="primary")

with col2:
    st.info("3 farklı ML modelinin AI/Human olasılıklarını (%) gösterir.")

if run:
    missing = [name for name, p in MODEL_FILES.items() if not p.exists()]
    if missing:
        st.error("Eğitilmiş modeller bulunamadı. Önce eğitim çalıştırın: python -m human_or_ai.train.train_models ...")
        st.stop()

    results = []
    for model_name, path in MODEL_FILES.items():
        pipe = load_model(str(path))
        pred = predict_one(pipe, text)
        results.append((model_name, pred))

    st.subheader("Sonuçlar")
    for model_name, pred in results:
        st.markdown(f"### {model_name}")
        st.write(f"Tahmin: **{pred.label.upper()}**")
        st.progress(int(round(pred.proba_ai * 100)), text=f"AI: {pred.proba_ai * 100:.2f}%")
        st.progress(int(round(pred.proba_human * 100)), text=f"Human: {pred.proba_human * 100:.2f}%")
