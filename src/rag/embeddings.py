from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Knowledge base — domain knowledge the RAG will reason over
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
- MEDIUM RISK: Default probability 20–50%. Requires further review.
- HIGH RISK: Default probability > 50%. Strong indicators of financial distress present.

FAIRNESS NOTE:
This model predicts repayment behavior based on financial patterns only.
It does not consider race, gender, religion, or nationality.
All decisions should be reviewed by a human officer before final action.
"""

def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([CREDIT_KNOWLEDGE])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )