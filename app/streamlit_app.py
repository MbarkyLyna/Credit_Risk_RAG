import streamlit as st
import pandas as pd
import joblib
import shap
import sys
import os

os.environ.get("GROQ_API_KEY")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")

from src.rag.embeddings import build_vectorstore, load_vectorstore
from src.rag.chain import ask_rag

from dotenv import load_dotenv
load_dotenv()

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]

@st.cache_resource
def load_model():
    if not os.path.exists("src/model/credit_model.pkl"):
        import subprocess
        subprocess.run(["python", "setup.py"], check=True)
    model = joblib.load("src/model/credit_model.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_resource
def load_rag():
    if not os.path.exists("chroma_db"):
        build_vectorstore()
    return load_vectorstore()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Analyzer", page_icon="", layout="wide")
st.title("Credit Risk Assessment System")
st.caption("Predicts repayment behavior using XGBoost + SHAP explainability + RAG-powered Q&A")

with st.spinner("Loading model and knowledge base..."):
    model, explainer = load_model()
    load_rag()

tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Ask the Analyst", "Document Upload"])

with tab1:
    st.subheader("Applicant Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Monthly Income ($)", 0, 50000, 4000)
        debt_ratio = st.slider("Debt Ratio", 0.0, 1.5, 0.3)
        utilization = st.slider("Revolving Credit Utilization", 0.0, 1.0, 0.3)
        dependents = st.slider("Number of Dependents", 0, 10, 1)

    with col2:
        late_30 = st.slider("Times 30-59 Days Late", 0, 10, 0)
        late_60 = st.slider("Times 60-89 Days Late", 0, 10, 0)
        late_90 = st.slider("Times 90+ Days Late", 0, 10, 0)
        open_lines = st.slider("Open Credit Lines", 0, 30, 5)
        real_estate = st.slider("Real Estate Loans", 0, 10, 1)

    if st.button("Assess Risk", type="primary"):
        applicant = {
            'RevolvingUtilizationOfUnsecuredLines': utilization,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': late_30,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': income,
            'NumberOfOpenCreditLinesAndLoans': open_lines,
            'NumberOfTimes90DaysLate': late_90,
            'NumberRealEstateLoansOrLines': real_estate,
            'NumberOfTime60-89DaysPastDueNotWorse': late_60,
            'NumberOfDependents': dependents
        }

        input_df = pd.DataFrame([applicant])
        proba = model.predict_proba(input_df)[0][1]
        sv = explainer.shap_values(input_df)[0]

        factors = sorted(
            zip(FEATURES, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        st.session_state['last_result'] = {
            "default_probability": round(float(proba), 4),
            "risk_label": "HIGH RISK" if proba > 0.5 else "LOW RISK",
            "top_factors": [{"feature": f, "impact": round(float(v), 4)} for f, v in factors]
        }

        result = st.session_state['last_result']

        col_a, col_b = st.columns(2)
        with col_a:
            color = "🔴" if result['risk_label'] == "HIGH RISK" else "🟢"
            st.metric("Risk Assessment", f"{color} {result['risk_label']}")
            st.metric("Default Probability", f"{result['default_probability']*100:.1f}%")

        with col_b:
            st.subheader("Top Risk Factors")
            for f in result['top_factors']:
                direction = "increases" if f['impact'] > 0 else "decreases"
                st.write(f"**{f['feature']}**  {direction} risk (impact: {f['impact']:.4f})")

with tab2:
    st.subheader("Ask the Credit Analyst")
    st.caption("Ask anything about the assessment, risk factors, or credit behavior.")

    if not st.session_state.get('last_result'):
        st.info("Run a risk assessment in the first tab to get context-aware answers.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        context = st.session_state.get('last_result', None)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = ask_rag(question, applicant_context=context)
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
with tab3:
    st.subheader("Loan Document Analysis")
    st.caption("Upload a PDF loan application, the agent will extract applicant data and run automatic risk assessment.")

    uploaded_file = st.file_uploader("Upload loan application PDF", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting applicant data from document..."):
            import tempfile
            import shutil

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(uploaded_file, tmp)
                tmp_path = tmp.name

            try:
                from src.agents.document_extractor import extract_applicant_profile
                profile = extract_applicant_profile(tmp_path)
                os.remove(tmp_path)

                confidence = profile.pop("confidence_score")
                missing = profile.pop("missing_fields")

                st.success(f"Extraction complete. Confidence: {confidence*100:.0f}%")

                if missing:
                    st.warning(f"Could not extract: {', '.join(missing)} , default values used.")

                st.subheader("Extracted Applicant Profile")
                st.json(profile)

                # Auto-run risk assessment
                st.subheader("Automatic Risk Assessment")
                with st.spinner("Running risk model..."):
                    # Map field names back to model format
                    applicant = {
                        'RevolvingUtilizationOfUnsecuredLines': profile['RevolvingUtilizationOfUnsecuredLines'],
                        'age': profile['age'],
                        'NumberOfTime30-59DaysPastDueNotWorse': profile['NumberOfTime30_59DaysPastDueNotWorse'],
                        'DebtRatio': profile['DebtRatio'],
                        'MonthlyIncome': profile['MonthlyIncome'],
                        'NumberOfOpenCreditLinesAndLoans': profile['NumberOfOpenCreditLinesAndLoans'],
                        'NumberOfTimes90DaysLate': profile['NumberOfTimes90DaysLate'],
                        'NumberRealEstateLoansOrLines': profile['NumberRealEstateLoansOrLines'],
                        'NumberOfTime60-89DaysPastDueNotWorse': profile['NumberOfTime60_89DaysPastDueNotWorse'],
                        'NumberOfDependents': profile['NumberOfDependents']
                    }

                    input_df = pd.DataFrame([applicant])
                    proba = model.predict_proba(input_df)[0][1]
                    sv = explainer.shap_values(input_df)[0]

                    factors = sorted(
                        zip(FEATURES, sv),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]

                    result = {
                        "default_probability": round(float(proba), 4),
                        "risk_label": "HIGH RISK" if proba > 0.5 else "LOW RISK",
                        "top_factors": [{"feature": f, "impact": round(float(v), 4)} for f, v in factors]
                    }

                    st.session_state['last_result'] = result

                    color = "🔴" if result['risk_label'] == "HIGH RISK" else "🟢"
                    st.metric("Risk Assessment", f"{color} {result['risk_label']}")
                    st.metric("Default Probability", f"{result['default_probability']*100:.1f}%")

                    st.subheader("Top Risk Factors")
                    for f in result['top_factors']:
                        direction = "increases" if f['impact'] > 0 else "decreases"
                        st.write(f"**{f['feature']}** — {direction} risk (impact: {f['impact']:.4f})")

                    st.info("Switch to the 'Ask the Analyst' tab to ask questions about this assessment.")

            except Exception as e:
                st.error(f"Extraction failed: {str(e)}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)