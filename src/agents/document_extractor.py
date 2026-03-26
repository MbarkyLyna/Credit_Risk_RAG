import os
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import pdfplumber

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Pydantic schema — matches your 10 XGBoost features exactly ────────────────
class ApplicantProfile(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        description="Ratio of revolving credit used to total available. Between 0 and 1."
    )
    age: int = Field(
        description="Age of the applicant in years."
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        description="Number of times the applicant was 30-59 days past due."
    )
    DebtRatio: float = Field(
        description="Monthly debt payments divided by monthly income."
    )
    MonthlyIncome: float = Field(
        description="Monthly income of the applicant in dollars."
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        description="Number of open credit lines and loans."
    )
    NumberOfTimes90DaysLate: int = Field(
        description="Number of times applicant was 90+ days past due."
    )
    NumberRealEstateLoansOrLines: int = Field(
        description="Number of real estate loans or lines of credit."
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        description="Number of times applicant was 60-89 days past due."
    )
    NumberOfDependents: int = Field(
        description="Number of dependents in the applicant's household."
    )
    confidence_score: float = Field(
        description="Your confidence in the extraction from 0 to 1."
    )
    missing_fields: list = Field(
        description="List of fields that could not be found in the document."
    )

# ── PDF text extraction ────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ── Agent ─────────────────────────────────────────────────────────────────────
def extract_applicant_profile(pdf_path: str) -> dict:
    """
    Takes a PDF path, extracts text, sends to LLM agent,
    returns structured ApplicantProfile as dict.
    """
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text:
        raise ValueError("Could not extract text from PDF. File may be scanned or corrupted.")

    prompt = f"""You are a financial document analyst. Extract the following credit risk assessment fields from the loan application document below.

Return ONLY a valid JSON object with these exact fields:
- RevolvingUtilizationOfUnsecuredLines (float 0-1): ratio of revolving credit used
- age (int): applicant age
- NumberOfTime30_59DaysPastDueNotWorse (int): times 30-59 days late
- DebtRatio (float): monthly debt / monthly income
- MonthlyIncome (float): monthly income in dollars
- NumberOfOpenCreditLinesAndLoans (int): open credit lines and loans
- NumberOfTimes90DaysLate (int): times 90+ days late
- NumberRealEstateLoansOrLines (int): real estate loans
- NumberOfTime60_89DaysPastDueNotWorse (int): times 60-89 days late
- NumberOfDependents (int): number of dependents
- confidence_score (float 0-1): your confidence in the extraction
- missing_fields (list): fields you could not find, empty list if all found

If a field is not found, use these defaults:
- floats: 0.0
- ints: 0
- Add the field name to missing_fields

Document:
{raw_text}

Return ONLY the JSON object, no explanation, no markdown, no backticks."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    raw_output = response.choices[0].message.content.strip()

    # Clean any accidental markdown
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    extracted = json.loads(raw_output)

    # Validate with Pydantic
    profile = ApplicantProfile(**extracted)
    return profile.model_dump()