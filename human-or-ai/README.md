# Human or AI (Makale Özetleri Üzerinden Metin Tespiti)

Bu proje; arXiv makale özetlerini toplayıp, "Human" (orijinal özet) ve "AI" (LLM ile yeniden yazılmış özet) sınıflarıyla veri seti oluşturur; 3 farklı ML modeli ile eğitim yapar ve Streamlit arayüzünde tek bir metin için her modelin **AI/Human olasılıklarını (%)** gösterir.

## Klasör Yapısı

- `data/raw/` Arxiv’den indirilen ham özetler
- `data/processed/` Temizlenmiş ve etiketlenmiş veri seti
- `artifacts/` Eğitilmiş modeller ve metrikler
- `src/human_or_ai/` Python paket kodu
- `app/` Streamlit uygulaması
- `tests/` White-box testler
- `docs/` Sözleşme, test case dokümanı

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Veri Toplama (Human)

```powershell
python -m human_or_ai.data_collect.arxiv_collect --query "cat:cs.CL" --max-results 3000 --out data/raw/arxiv_human.csv
```

## 2) AI Özet Üretimi (LLM ile, opsiyonel)

Bu adım bir LLM API anahtarı gerektirir. Anahtarınızı environment variable olarak verin.

```powershell
$env:GOOGLE_API_KEY="..."  # veya OPENAI_API_KEY
python -m human_or_ai.data_collect.generate_ai_summaries --in data/raw/arxiv_human.csv --out data/raw/arxiv_ai.csv --provider gemini
```

## 3) Birleştirme + Temizleme

```powershell
python -m human_or_ai.data_clean.build_dataset --human data/raw/arxiv_human.csv --ai data/raw/arxiv_ai.csv --out data/processed/dataset.csv
```

## 4) Eğitim (3 Model)

```powershell
python -m human_or_ai.train.train_models --data data/processed/dataset.csv
```

## 5) Uygulama

```powershell
streamlit run app/streamlit_app.py
```

## Notlar

- Lisans konusu veri kaynağına bağlıdır. Arxiv özetlerini kullanmadan önce ders isterlerine uygun lisans/terms kontrolü yapmanız gerekir.
