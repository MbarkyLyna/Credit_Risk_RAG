from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
import numpy as np
import os

def compute_dataset_stats():
    try:
        df = pd.read_csv("data/cs-training.csv", index_col=0)
        df = df.dropna(subset=['SeriousDlqin2yrs'])
        df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
        df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
        df = df[df['age'] > 18]
        df = df[df['age'] < 100]
        df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1]

        defaulters = df[df['SeriousDlqin2yrs'] == 1]
        non_defaulters = df[df['SeriousDlqin2yrs'] == 0]

        stats = f"""
DATASET STATISTICS (150,000 real applicants):

Overall default rate: {df['SeriousDlqin2yrs'].mean()*100:.1f}%
Total applicants analyzed: {len(df):,}

Age:
- Average age of defaulters: {defaulters['age'].mean():.1f} years
- Average age of non-defaulters: {non_defaulters['age'].mean():.1f} years
- Applicants under 30 default rate: {df[df['age'] < 30]['SeriousDlqin2yrs'].mean()*100:.1f}%
- Applicants over 50 default rate: {df[df['age'] > 50]['SeriousDlqin2yrs'].mean()*100:.1f}%

Credit Utilization (RevolvingUtilizationOfUnsecuredLines):
- Average utilization of defaulters: {defaulters['RevolvingUtilizationOfUnsecuredLines'].mean():.2f}
- Average utilization of non-defaulters: {non_defaulters['RevolvingUtilizationOfUnsecuredLines'].mean():.2f}
- Applicants with utilization > 0.7 default rate: {df[df['RevolvingUtilizationOfUnsecuredLines'] > 0.7]['SeriousDlqin2yrs'].mean()*100:.1f}%
- Applicants with utilization < 0.3 default rate: {df[df['RevolvingUtilizationOfUnsecuredLines'] < 0.3]['SeriousDlqin2yrs'].mean()*100:.1f}%

90+ Day Late Payments:
- Applicants with 0 occurrences default rate: {df[df['NumberOfTimes90DaysLate'] == 0]['SeriousDlqin2yrs'].mean()*100:.1f}%
- Applicants with 1+ occurrences default rate: {df[df['NumberOfTimes90DaysLate'] >= 1]['SeriousDlqin2yrs'].mean()*100:.1f}%
- Applicants with 3+ occurrences default rate: {df[df['NumberOfTimes90DaysLate'] >= 3]['SeriousDlqin2yrs'].mean()*100:.1f}%
- Average 90-day late count for defaulters: {defaulters['NumberOfTimes90DaysLate'].mean():.2f}

Monthly Income:
- Median income of defaulters: ${defaulters['MonthlyIncome'].median():,.0f}
- Median income of non-defaulters: ${non_defaulters['MonthlyIncome'].median():,.0f}
- Applicants with income below $3000 default rate: {df[df['MonthlyIncome'] < 3000]['SeriousDlqin2yrs'].mean()*100:.1f}%
- Applicants with income above $7000 default rate: {df[df['MonthlyIncome'] > 7000]['SeriousDlqin2yrs'].mean()*100:.1f}%

Debt Ratio:
- Average debt ratio of defaulters: {defaulters['DebtRatio'].mean():.2f}
- Average debt ratio of non-defaulters: {non_defaulters['DebtRatio'].mean():.2f}
- Applicants with debt ratio > 0.4 default rate: {df[df['DebtRatio'] > 0.4]['SeriousDlqin2yrs'].mean()*100:.1f}%

Number of Dependents:
- Average dependents of defaulters: {defaulters['NumberOfDependents'].mean():.1f}
- Average dependents of non-defaulters: {non_defaulters['NumberOfDependents'].mean():.1f}

Model Performance:
- ROC-AUC: 0.86
- Training set size: 20,000 applicants (stratified sample)
- Most predictive feature: NumberOfTimes90DaysLate (highest SHAP magnitude)
- Second most predictive: RevolvingUtilizationOfUnsecuredLines
"""
        return stats
    except Exception:
        return ""

CREDIT_KNOWLEDGE = """
CREDIT RISK FUNDAMENTALS

RevolvingUtilizationOfUnsecuredLines:
Measures how much of available revolving credit a person is using.
Values above 0.7 indicate financial stress and strongly predict default.
Values below 0.3 indicate healthy credit management.

Age:
Younger applicants (under 30) statistically show higher default rates due to
limited credit history. Applicants over 50 tend to have more stable repayment behavior.

Past Due Payments (30-59 days, 60-89 days, 90+ days):
Any history of late payments is a strong negative signal.
90+ day delinquencies are the strongest predictor of future default in this dataset.
Multiple occurrences compound the risk significantly.

DebtRatio:
Monthly debt payments divided by monthly income.
Ratios above 0.4 indicate the applicant is heavily leveraged.
Extremely high ratios (above 1.0) may indicate data errors or extreme financial distress.

MonthlyIncome:
Higher income provides a buffer against default.
Income below 3000/month combined with high debt ratio is a critical risk flag.

NumberOfOpenCreditLinesAndLoans:
Too few lines may indicate thin credit history.
Too many (above 15) may indicate over-reliance on credit.

NumberOfDependents:
More dependents increase financial obligations.
Combined with low income, this amplifies default risk.

RISK CATEGORIES:
- LOW RISK: Default probability < 20%. Applicant shows stable financial behavior.
- MEDIUM RISK: Default probability 20-50%. Requires further review.
- HIGH RISK: Default probability > 50%. Strong indicators of financial distress present.

FAIRNESS NOTE:
This model predicts repayment behavior based on financial patterns only.
It does not consider race, gender, religion, or nationality.
All decisions should be reviewed by a human officer before final action.
"""

def build_vectorstore():
    dataset_stats = compute_dataset_stats()
    full_knowledge = CREDIT_KNOWLEDGE + dataset_stats

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([full_knowledge])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )