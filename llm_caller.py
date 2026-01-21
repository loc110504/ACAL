import os
import json
import re
from typing import Optional, Literal, Generator
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, RateLimitError
import time
import random

load_dotenv(override=True)

from google import genai
from google.genai import types


# ============ Pydantic Models ============
class LLMResponse(BaseModel):
    """Standard response model for all LLM providers"""
    content: str = Field(..., description="The generated text content")
    
    def __str__(self):
        return self.content


class JsonAnswer(BaseModel):
    """Structured JSON response for legal reasoning"""
    answer: str = Field(..., description="Answer text (Yes or No)")
    explanation: str = Field(..., description="A 2-3 sentence explanation for the answer")


# ============ Provider Implementations ============

def _call_gpt(prompt: str, temperature: float = 0.3, max_tokens: int = 1024, retries: int = 3) -> str:
    """Call Azure OpenAI GPT-4o-mini"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    if not endpoint or not api_key:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment")
    
    client = OpenAI(
        base_url=f"{endpoint}/openai/deployments/{deployment}",
        api_key=api_key,
        default_query={"api-version": "2024-08-01-preview"},
        default_headers={"api-key": api_key}
    )
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
            
        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"GPT Rate limited. Retrying in {wait:.2f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"GPT Error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    raise Exception("GPT: All retries failed")


def _call_llama(prompt: str, temperature: float = 0.3, max_tokens: int = 1024, retries: int = 3) -> str:
    """Call Azure Llama 3.3 70B"""
    endpoint = ""
    api_key = os.getenv("AZURE_API_KEY")
    
    if not api_key:
        raise ValueError("Missing AZURE_API_KEY in environment")
    
    client = OpenAI(base_url=endpoint, api_key=api_key)
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Llama-3.3-70B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
            
        except RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"Llama Rate limited. Retrying in {wait:.2f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"Llama Error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    raise Exception("Llama: All retries failed")



def _call_gemini(prompt: str, temperature: float = 0.3, max_tokens: int = 1024, retries: int = 3) -> str:
    """Call Google Gemini"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")
    
    client = genai.Client(api_key=api_key)
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens= max(max_tokens, 4096),
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0  # Disabled thinking
                    )
                ),
                contents=prompt
            )
            
            text = response.text
            # Remove code block markers
            cleaned = re.sub(r"^```(?:json)?\s*$", "", text, flags=re.MULTILINE)
            cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE)
            return cleaned.strip()
            
        except Exception as e:
            error_msg = str(e)
            # Check for quota/rate limit errors
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ Gemini quota exceeded. Switching to GPT fallback...")
                return _call_gpt(prompt, temperature, max_tokens, retries)
            
            print(f"Gemini Error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    
    raise Exception("Gemini: All retries failed")


def _call_gemini_stream(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> Generator[str, None, None]:
    """Stream responses from Gemini model"""
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
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
                if buffer.startswith("```json") or buffer.startswith("```"):
                    continue
                yield chunk.text
        
        # Clean final output if needed
        if buffer.startswith("```json") or buffer.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*$", "", buffer, flags=re.MULTILINE)
            cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE)
            yield cleaned.strip()
            
    except Exception as e:
        print(f"Error in Gemini streaming: {e}")
        yield ""


# ============ Unified Interface ============

def call_llm(
    prompt: str,
    provider: Literal["gemini", "gpt", "llama"] = "gemini",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    retries: int = 3
) -> str:
    """
    Unified LLM caller supporting multiple providers.
    
    Args:
        prompt: The input prompt
        provider: LLM provider to use ("gemini", "gpt", "llama")
        temperature: Sampling temperature (0.0 - 1.0)
        max_tokens: Maximum tokens to generate
        retries: Number of retry attempts on failure
    
    Returns:
        Generated text as string
    """
    provider_map = {
        "gemini": _call_gemini,
        "gpt": _call_gpt,
        "llama": _call_llama,
    }
    
    if provider not in provider_map:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(provider_map.keys())}")
    
    return provider_map[provider](prompt, temperature, max_tokens, retries)


def call_llm_stream(
    prompt: str,
    provider: Literal["gemini", "gpt", "llama"] = "gemini",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> Generator[str, None, None]:
    """
    Stream LLM responses (currently only supports Gemini).
    
    Args:
        prompt: The input prompt
        provider: LLM provider to use (currently only "gemini")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Yields:
        Text chunks as they are generated
    """
    if provider != "gemini":
        raise NotImplementedError(f"Streaming not yet implemented for {provider}")
    
    return _call_gemini_stream(prompt, temperature, max_tokens)
