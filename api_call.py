import os
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI, RateLimitError
import time
import random
import ast
import re
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
# from env_config import (
#     GEMINI_API_KEY,
# )
from google import genai
from google.genai import types


# ==== Pydantic model định nghĩa cấu trúc output ====
class JsonAnswer(BaseModel):
    answer: str = Field(..., description="Answer text (Yes or No)")
    explanation: str = Field(..., description="A 2-3 sentence explanation for the answer")

load_dotenv()

Message = Dict[str, str]

# def gpt_generate(
#     system_prompt: str,
#     temperature: float = 0.0,
#     max_tokens: int = 512,
#     retries: int = 5
# ) -> Optional[JsonAnswer]:
#     endpoint = "https://22127-mbrabz8d-swedencentral.openai.azure.com/"
#     api_key = os.getenv("OPENAI_API")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
#     deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

#     if not endpoint or not api_key:
#         raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_API_KEY")

#     client = AzureOpenAI(
#         azure_endpoint=endpoint,
#         api_key=api_key,
#         api_version=api_version,
#     )

#     for attempt in range(retries):
#         try:
#             completion = client.chat.completions.create(
#                 model=deployment_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                 ],
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 response_format={"type": "json_object"},  # yêu cầu JSON output
#             )

#             raw_output = completion.choices[0].message.content.strip()

#             try:
#                 parsed_json = json.loads(raw_output)
#                 # Parse sang Pydantic model
#                 result = JsonAnswer(**parsed_json)
#                 return result

#             except (json.JSONDecodeError, ValidationError) as e:
#                 print("⚠️ Output không đúng định dạng:", e)
#                 print("Raw output:", raw_output)
#                 return None

#         except RateLimitError:
#             wait = 2 ** attempt + random.uniform(0, 1)
#             print(f"⏳ Rate limited (429). Retrying in {wait:.2f}s...")
#             time.sleep(wait)
#         except Exception as e:
#             print(f"❌ Error on attempt {attempt+1}: {e}")
#             time.sleep(2)

#     return None

def gpt_generate(
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 5
) -> Optional[JsonAnswer]:
    api_key = os.getenv("OPENAI_API")

    client = OpenAI(
        api_key=api_key,
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
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
                result = JsonAnswer(**parsed_json)
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
) -> Optional[JsonAnswer]:
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
                result = JsonAnswer(**parsed_json)
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
) -> Optional[JsonAnswer]:
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
                result = JsonAnswer(**parsed_json)
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


class GeminiAnswer(BaseModel):
    answer: Literal["Yes", "No"] = Field(description="Answer text (Yes or No)")
    explanation: str = Field(description="a short explanation for the answer or reasoning steps supporting the answer.")

def clean_and_fix_json(raw_text: str) -> str:
    """
    Làm sạch và sửa JSON bị lỗi từ Gemini
    """
    # Loại bỏ markdown code blocks nếu có
    raw_text = re.sub(r'^```json\s*', '', raw_text.strip())
    raw_text = re.sub(r'\s*```$', '', raw_text.strip())
    
    # Loại bỏ whitespace thừa
    raw_text = raw_text.strip()
    
    # Nếu JSON bị cắt ngang (không có dấu } kết thúc)
    if raw_text.count('{') > raw_text.count('}'):
        # Thêm " và } để đóng JSON
        if not raw_text.rstrip().endswith('"'):
            raw_text = raw_text.rstrip() + '"'
        raw_text = raw_text + '\n}'
    
    # Xử lý trường hợp thiếu dấu ngoặc kép cho value
    # Ví dụ: "answer": No -> "answer": "No"
    raw_text = re.sub(r'"answer":\s*([A-Za-z]+)(?!\w)', r'"answer": "\1"', raw_text)
    
    # Xử lý explanation bị cắt ngang
    # Nếu có "explanation": " nhưng không đóng
    if '"explanation"' in raw_text:
        # Đếm số dấu ngoặc kép sau "explanation":
        expl_match = re.search(r'"explanation":\s*"', raw_text)
        if expl_match:
            remaining = raw_text[expl_match.end():]
            quote_count = remaining.count('"')
            # Nếu số dấu ngoặc kép lẻ -> thiếu dấu đóng
            if quote_count % 2 == 1:
                pass  # Đã xử lý ở trên với việc thêm "
    
    return raw_text

def gemini_generate(
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,  # Tăng lên để tránh bị cắt
    retries: int = 3,
) -> Optional[GeminiAnswer]:

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=system_prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                    "response_json_schema": GeminiAnswer.model_json_schema(),
                },
            )

            raw_output = response.text.strip()
            
            # Thử parse trực tiếp trước
            try:
                return GeminiAnswer.model_validate_json(raw_output)
            except (ValidationError, json.JSONDecodeError):
                # Nếu fail -> làm sạch và thử lại
                cleaned = clean_and_fix_json(raw_output)
                print(f"🔧 Cleaned JSON:\n{cleaned}")
                return GeminiAnswer.model_validate_json(cleaned)

        except ValidationError as e:
            print(f"⚠️ Gemini output sai schema (attempt {attempt+1}/{retries})")
            print(f"Raw output:\n{response.text}")
            print(f"Validation error: {e}")
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error (attempt {attempt+1}/{retries})")
            print(f"Raw output:\n{response.text}")
            print(f"JSON error: {e}")
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))

        except Exception as e:
            print(f"❌ Gemini error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    print("❌ All retries failed")
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