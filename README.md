# Credit Risk Assessment System

An AI-powered credit risk analysis tool that predicts repayment behavior using machine learning and explains decisions through a RAG-powered analyst chatbot.

> **Live Demo:** [creditriskrag-fiqcxtgxjnsmweeg65i7oh.streamlit.app](https://creditriskrag-fiqcxtgxjnsmweeg65i7oh.streamlit.app/)

---

## What It Does

Input an applicant's financial profile and get:
- A **default probability score** based on real historical data
- **Explainable AI** exactly which factors drove the decision and by how much
- A **conversational analyst** you can ask *"Why was this person flagged?"* and get a data-driven answer


## Architecture

```
Streamlit Frontend (Streamlit Cloud)
        |
XGBoost Model + SHAP Explainability
        +
RAG Pipeline (LangChain + ChromaDB + Groq LLaMA 3.3)
```

---

## How It Works

### 1. Risk Prediction
- Trained on **150,000 real applicants** from the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset
- **XGBoost classifier** with class imbalance handling (`scale_pos_weight`)
- Achieves **~0.86 ROC-AUC** on held-out test set
- Model tracked and versioned with **MLflow**

### 2. Explainability (SHAP)
- Every prediction comes with **SHAP values** showing the impact of each feature
- Answers *why* not just *what*  making decisions transparent and auditable
- Critical for real-world financial applications where regulators require explainability

### 3. RAG Analyst Chatbot
- **LangChain** retrieval pipeline over a credit risk knowledge base
- **ChromaDB** vector store with `sentence-transformers/paraphrase-MiniLM-L3-v2` embeddings
- **Groq LLaMA 3.3 70B** for fast, high-quality responses
- Chatbot receives actual assessment data (probability + SHAP values) as context, answers are specific to the applicant, not generic

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| RAG Pipeline | LangChain, ChromaDB |
| Embeddings | HuggingFace sentence-transformers |
| LLM | Groq LLaMA 3.3 70B |
| Backend API | FastAPI |
| Deployment | Streamlit Cloud |

---

## Run Locally

### Prerequisites
- Python 3.10+
- Groq API key from [console.groq.com](https://console.groq.com)

### Setup
```bash
git clone https://github.com/MbarkyLyna/Credit_Risk_RAG.git
cd Credit_Risk_RAG

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

echo GROQ_API_KEY=your_key_here > .env

python setup.py

streamlit run app/streamlit_app.py
```

### Optional: FastAPI backend
```bash
uvicorn src.api.main:app --reload
```
API docs at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/assess` | Run risk assessment on applicant profile |
| POST | `/chat` | Ask the RAG analyst a question |
| GET | `/health` | Health check |

---

## Author

**Lyna Mbarky** : AI Engineering Student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lyna_Mbarky-blue)](https://linkedin.com/in/lyna-m-barky-4899b51a1)
[![GitHub](https://img.shields.io/badge/GitHub-MbarkyLyna-black)](https://github.com/MbarkyLyna)