import os
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI, RateLimitError
import time
import random
import ast
import re
from pydantic import BaseModel, Field, ValidationError

# ==== Pydantic model định nghĩa cấu trúc output ====
class LearnedHandsCourtsAnswer(BaseModel):
    answer: str = Field(..., description="Answer text (Yes or No)")
    explanation: str = Field(..., description="A 2-3 sentence explanation for the answer")

load_dotenv()

Message = Dict[str, str]

def gpt_generate(
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 5
) -> Optional[LearnedHandsCourtsAnswer]:
    endpoint = os.getenv("AZURE_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    if not endpoint or not api_key:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_API_KEY")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},  # yêu cầu JSON output
            )

            raw_output = completion.choices[0].message.content.strip()

            try:
                parsed_json = json.loads(raw_output)
                # Parse sang Pydantic model
                result = LearnedHandsCourtsAnswer(**parsed_json)
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                print("⚠️ Output không đúng định dạng:", e)
                print("Raw output:", raw_output)
                return None

        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"⏳ Rate limited (429). Retrying in {wait:.2f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"❌ Error on attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


def llama_generate(
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 5
) -> Optional[LearnedHandsCourtsAnswer]:
    endpoint = "https://22127-mbrabz8d-swedencentral.services.ai.azure.com/openai/v1/"
    deployment_name = "Llama-3.3-70B-Instruct"

    api_key = os.getenv("AZURE_API_KEY")

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},  # yêu cầu JSON output
            )

            raw_output = completion.choices[0].message.content.strip()

            try:
                parsed_json = json.loads(raw_output)
                # Parse sang Pydantic model
                result = LearnedHandsCourtsAnswer(**parsed_json)
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                print("⚠️ Output không đúng định dạng:", e)
                print("Raw output:", raw_output)
                return None

        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"⏳ Rate limited (429). Retrying in {wait:.2f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"❌ Error on attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


def phi4_generate(
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 5
) -> Optional[LearnedHandsCourtsAnswer]:
    endpoint = "https://22127-mbrabz8d-swedencentral.services.ai.azure.com/openai/v1/"
    model_name = "Phi-4"
    deployment_name = "Phi-4"

    api_key = os.getenv("AZURE_API_KEY")

    client = OpenAI(
        base_url=f"{endpoint}",
        api_key=api_key
    )
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},  # yêu cầu JSON output
            )

            raw_output = completion.choices[0].message.content.strip()

            try:
                parsed_json = json.loads(raw_output)
                # Parse sang Pydantic model
                result = LearnedHandsCourtsAnswer(**parsed_json)
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                print("⚠️ Output không đúng định dạng:", e)
                print("Raw output:", raw_output)
                return None

        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"⏳ Rate limited (429). Retrying in {wait:.2f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"❌ Error on attempt {attempt+1}: {e}")
            time.sleep(2)

    return None

if __name__ == "__main__":
        # Ví dụ dùng trong main.py
    test_argument = """

    """
    result = llm_generate(
        messages=[{"role": "user", "content": test_argument}],
        model="gpt-4o-mini",   # hoặc để None dùng model default từ env
        temperature=0,
        max_tokens=512,
        json_mode=True,       # True nếu muốn JSON mode
    )

    print(result)