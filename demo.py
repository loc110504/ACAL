from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from env_config import GEMINI_API_KEY

class QAResponse(BaseModel):
    answer: str = Field(description="The final concise answer to the question.")
    explanations: str = Field(
        description="Give simple explanation or reasoning steps supporting the answer."
    )

client = client = genai.Client(api_key=GEMINI_API_KEY)

prompt = """
Question:
Why is it important to preheat the oven before baking cookies?

Answer the question clearly and concisely.
Also provide supporting explanations.
"""

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": QAResponse.model_json_schema(),
    },
)

qa_result = QAResponse.model_validate_json(response.text)
answer = qa_result.answer
explanations = qa_result.explanations

print("ANSWER:")
print(answer)

print("\nEXPLANATIONS:")
print(explanations)