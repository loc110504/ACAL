import os
import json
import requests
from env_config import (
    GEMINI_API_KEY,
)

from google import genai
from google.genai import types

    
def call_gemini_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Calling Gemini LLM with prompt: {prompt}")
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=min(max_tokens, 2048)),
        contents=prompt
    )
    print(f"Gemini response: {response.text}")
    return response.text


def call_gemini_llm_stream(prompt: str, temperature: float = 0.3, max_tokens: int = 1024):
    """Stream responses from Gemini model"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Use generate_content_stream for streaming responses
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=min(max_tokens, 1024)
            ),
            contents=prompt
        )
        
        # Yield each chunk of text as it arrives
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
                
    except Exception as e:
        print(f"Error in Gemini streaming: {e}")
        yield ""


call_llm = call_gemini_llm
call_llm_stream = call_gemini_llm_stream
