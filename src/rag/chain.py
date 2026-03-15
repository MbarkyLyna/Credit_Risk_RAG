from groq import Groq
from dotenv import load_dotenv
import os
from src.rag.embeddings import load_vectorstore

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_rag(question: str, applicant_context: dict = None) -> str:
    vectorstore = load_vectorstore()
    
    # Direct similarity search — bypasses broken retriever
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    applicant_str = ""
    if applicant_context:
        factors_formatted = "\n".join([
            f"  - {f['feature']}: impact {f['impact']} ({'increases' if f['impact'] > 0 else 'decreases'} risk)"
            for f in applicant_context.get('top_factors', [])
        ])
        applicant_str = f"""
APPLICANT ASSESSMENT DATA (you MUST reference these exact numbers in your answer):
- Default Probability: {applicant_context.get('default_probability') * 100:.1f}%
- Risk Label: {applicant_context.get('risk_label')}
- Top SHAP Factor Impacts:
{factors_formatted}
"""

    prompt = f"""You are a credit risk analyst assistant.
You MUST reference the specific numbers from the applicant profile in your answer (probability, SHAP impact values).
Never give generic answers — always tie your response to the actual data provided.
Never mention interest rates. Focus only on repayment behavior and financial patterns.

Context:
{context}

{applicant_str}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content