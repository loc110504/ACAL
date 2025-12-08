import os
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI, RateLimitError
import time
import random
import ast
import re
load_dotenv()

Message = Dict[str, str]


class AzureLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ):
        endpoint = os.getenv("LLAMA_ENDPOINT")
        api_key = os.getenv("AZURE_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        default_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

        if not endpoint or not api_key:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY")

        self.model = model or default_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def generate(
        self,
        system: str,
        messages: List[Message],
        *,
        json_mode: bool = False,
        override_model: Optional[str] = None,
        **extra_params,
    ) -> str:
        """Gọi Azure OpenAI với system + messages, có thể bật JSON mode, đổi model runtime."""
        formatted_messages: List[Message] = [
            {"role": "system", "content": system},
            *messages,
        ]

        params = dict(
            model=override_model or self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=formatted_messages,
        )

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        params.update(extra_params)

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


# Hàm tiện dụng dùng trong main.py
def gpt_generate(
    system: str,
    messages: List[Message],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    json_mode: bool = False,
    **extra_params,
) -> str:
    """
    Hàm wrapper đơn giản:
    - Chọn model & config cơ bản
    - Gọi AzureLLM.generate
    """
    llm = AzureLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    return llm.generate(
        system=system,
        messages=messages,
        json_mode=json_mode,
        **extra_params,
    )


def llama_generate(role_prompt: str, user_prompt: str, retries: int = 5):
    endpoint = "https://22127-mbrabz8d-swedencentral.services.ai.azure.com/openai/v1/"
    model_name = "Llama-3.3-70B-Instruct"
    deployment_name = "Llama-3.3-70B-Instruct"

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
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": user_prompt}

                ],
            )
            content = completion.choices[0].message.content
            result = json.loads(content)
            return result
        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"⏳ Rate limited (429). Retrying in {wait:.2f}s...")
            time.sleep(wait)

def safe_json_parse(content: str):
    """
    Parse JSON from LLM output.
    Handles:
    - ```json ... ```
    - ``` ... ```
    - Single quotes
    - Mixed formatting
    - Extra whitespace
    """

    if not isinstance(content, str) or content.strip() == "":
        return {"error": "Empty or invalid content", "raw_output": content}

    cleaned = content.strip()

    # Remove code fences like ```json or ```
    cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try json.loads first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try ast.literal_eval (handles single quotes)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        pass

    # Final fallback → return raw text
    return {
        "error": "Could not parse JSON",
        "raw_output": content
    }
def phi4_generate(role_prompt: str, user_prompt: str, retries: int = 5):
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
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": user_prompt}

                ],
            )
            content = completion.choices[0].message.content
            result = safe_json_parse(content)
            return result
        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"⏳ Rate limited (429). Retrying in {wait:.2f}s...")
            time.sleep(wait)

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