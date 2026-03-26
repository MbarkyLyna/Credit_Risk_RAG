from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os
from dotenv import load_dotenv
from src.rag.chain import ask_rag
from src.agents.document_extractor import extract_applicant_profile
import shutil
from fastapi import UploadFile, File
import tempfile
load_dotenv()

app = FastAPI(title="Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]

model = joblib.load("src/model/credit_model.pkl")
explainer = shap.TreeExplainer(model)


class ApplicantInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int


class ChatInput(BaseModel):
    question: str
    applicant_context: dict = None


@app.post("/assess")
def assess(data: ApplicantInput):
    applicant = {
        'RevolvingUtilizationOfUnsecuredLines': data.RevolvingUtilizationOfUnsecuredLines,
        'age': data.age,
        'NumberOfTime30-59DaysPastDueNotWorse': data.NumberOfTime30_59DaysPastDueNotWorse,
        'DebtRatio': data.DebtRatio,
        'MonthlyIncome': data.MonthlyIncome,
        'NumberOfOpenCreditLinesAndLoans': data.NumberOfOpenCreditLinesAndLoans,
        'NumberOfTimes90DaysLate': data.NumberOfTimes90DaysLate,
        'NumberRealEstateLoansOrLines': data.NumberRealEstateLoansOrLines,
        'NumberOfTime60-89DaysPastDueNotWorse': data.NumberOfTime60_89DaysPastDueNotWorse,
        'NumberOfDependents': data.NumberOfDependents
    }

    input_df = pd.DataFrame([applicant])
    proba = model.predict_proba(input_df)[0][1]
    sv = explainer.shap_values(input_df)[0]

    factors = sorted(
        zip(FEATURES, sv),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return {
        "default_probability": round(float(proba), 4),
        "risk_label": "HIGH RISK" if proba > 0.5 else "LOW RISK",
        "top_factors": [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in factors
        ]
    }
    
@app.post("/extract")
async def extract_from_document(file: UploadFile = File(...)):
    """
    Upload a PDF loan application and extract structured applicant profile.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        profile = extract_applicant_profile(tmp_path)
        return profile
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)


@app.post("/chat")
def chat(data: ChatInput):
    answer = ask_rag(data.question, applicant_context=data.applicant_context)
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}