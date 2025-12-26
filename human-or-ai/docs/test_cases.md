# STD Test Case Dokümanı (En az 5)

## TC-01: Veri Toplama (Human) - CSV Oluşturma
- **Önkoşul**: İnternet erişimi, `requirements.txt` kurulu
- **Adımlar**:
  1. `python -m human_or_ai.data_collect.arxiv_collect --query "cat:cs.CL" --max-results 10 --out data/raw/arxiv_human.csv`
- **Beklenen**:
  - Çıktı CSV dosyası oluşur
  - `summary` kolonunda boş olmayan metinler vardır

## TC-02: Veri Seti Birleştirme ve Temizleme
- **Önkoşul**: `data/raw/arxiv_human.csv` ve `data/raw/arxiv_ai.csv` mevcut
- **Adımlar**:
  1. `python -m human_or_ai.data_clean.build_dataset --human data/raw/arxiv_human.csv --ai data/raw/arxiv_ai.csv --out data/processed/dataset.csv`
- **Beklenen**:
  - `dataset.csv` kolonları: `text,label`
  - `label` sadece `human` ve `ai`

## TC-03: Model Eğitimi - Artifact Üretimi
- **Önkoşul**: `data/processed/dataset.csv` mevcut
- **Adımlar**:
  1. `python -m human_or_ai.train.train_models --data data/processed/dataset.csv`
- **Beklenen**:
  - `artifacts/` içinde `logreg.joblib`, `nb.joblib`, `linear_svc_cal.joblib`
  - `artifacts/metrics.json` oluşur

## TC-04: Uygulama - Tahmin Al
- **Önkoşul**: Eğitim tamamlanmış
- **Adımlar**:
  1. `streamlit run app/streamlit_app.py`
  2. Metin gir ve `Tahmin Et` tıkla
- **Beklenen**:
  - 3 model için AI/Human % değerleri görünür

## TC-05: Hatalı Girdi - Çok Kısa Metin
- **Önkoşul**: Eğitim tamamlanmış
- **Adımlar**:
  1. Uygulamada 5-10 karakterlik metin gir
- **Beklenen**:
  - Kullanıcıya hata gösterilir (metin çok kısa)
