import os
import json
import requests

from dotenv import load_dotenv
load_dotenv(override=True)

from google import genai
from google.genai import types

    
def call_gemini_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    import re
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=min(max_tokens, 2048)),
        contents=prompt
    )
    text = response.text
    # Remove code block markers if present (handles lines like ```json or ```)
    cleaned = re.sub(r"^```(?:json)?\s*$", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def call_gemini_llm_stream(prompt: str, temperature: float = 0.3, max_tokens: int = 1024):
    """Stream responses from Gemini model"""
    import re
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=min(max_tokens, 1024)
            ),
            contents=prompt
        )
        buffer = ""
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                buffer += chunk.text
                # Only yield after cleaning if code block markers are detected or at the end
                if buffer.startswith("```json") or buffer.startswith("```"):
                    # Wait until the end to yield cleaned output
                    continue
                yield chunk.text
        # After streaming, clean and yield if code block markers were present
        if buffer.startswith("```json") or buffer.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*$", "", buffer, flags=re.MULTILINE)
            cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE)
            yield cleaned.strip()
    except Exception as e:
        print(f"Error in Gemini streaming: {e}")
        yield ""


call_llm = call_gemini_llm
call_llm_stream = call_gemini_llm_stream
