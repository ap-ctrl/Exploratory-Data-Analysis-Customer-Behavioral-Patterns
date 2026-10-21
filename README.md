# 🛍️ RetailMind: E-Commerce Customer Behaviour Mining & Recommendation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end data mining and personalized recommendation system built on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online-retail?utm_source=chatgpt.com) (541k+ transactions).

---

## 🚀 Features

- **RFM Customer Segmentation:** K-Means clustering ($k=4$) backed by stability testing (ARI: 0.9984), Hierarchical (Ward), and DBSCAN.
- **Market Basket Analysis:** Frequent itemset and cross-sell mining using **FP-Growth** (2,340+ rules mined).
- **Hybrid Recommender Engine:** Blends segment-specific association rules, global rules, and popularity metrics for personalized, cold-start resistant recommendations.
- **Strict Offline Benchmark:** Protocol B (unseen new items only) evaluated on future holdout transactions ($Hit\text{-}Rate@10 = 40.9\%$, $+48.7\%$ over popularity).
- **Interactive Web App:** Streamlit dashboard for real-time customer profiling and recommendations.

---

## 📊 Key Results

| Model | Hit-Rate@10 | Precision@10 | NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: |
| Global Popularity | 0.275 | 0.039 | 0.042 | 1.97% |
| Segment Popularity | 0.302 | 0.043 | 0.047 | 2.34% |
| Global Rules Only | 0.348 | 0.058 | 0.069 | 6.48% |
| **Proposed Hybrid Model** | **0.409** | **0.068** | **0.078** | **10.99%** |

*Delivers **5.5x broader catalog exploration** while outperforming baselines on unseen transactions.*

---

## ⚡ Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/tinyelfie/E-Commerce-Customer-Behaviour-Mining-and-Personalized-Product-Recommendation.git
cd E-Commerce-Customer-Behaviour-Mining-and-Personalized-Product-Recommendation

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Web App
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore customer segments and test recommendations interactively.

### 3. Re-run Pipeline or Inspect Notebook (Optional)
```bash
# Re-run full pipeline end-to-end
python run_pipeline.py

# Or explore step-by-step
jupyter notebook notebooks/ecommerce_customer_mining_and_recommendation.ipynb
```

---

## 📁 Project Structure

```
├── app/                  # Streamlit web application (streamlit_app.py)
├── data/                 # Raw parquet cache and processed splits
├── dataset/              # Source dataset (Online Retail.xlsx)
├── models/               # Saved models (KMeans, Scaler, Association Rules)
├── notebooks/            # End-to-end research notebook
├── reports/              # Data quality, behaviour & evaluation reports + figures
├── src/                  # Modular pipeline (loader, clean, segment, rules, recsys, eval)
├── run_pipeline.py       # Single-command pipeline runner
└── requirements.txt      # Python dependencies
```

---

## 📚 Dataset & Citation

- **Dataset:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online-retail?utm_source=chatgpt.com)
- **Reference:** Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining.* Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208.

---

## 📄 License
Distributed under the [MIT License](LICENSE).
