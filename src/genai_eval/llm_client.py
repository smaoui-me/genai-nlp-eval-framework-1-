"""
llm_client.py

Shared LLM client and call_llm function used by all extraction methods.
Reads connection settings from .env and accepts LLM parameters at call time
so configs can control temperature, max_tokens, etc.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

endpoint = os.getenv("LLM_ENDPOINT")
api_key = os.getenv("LLM_API_KEY")
deployment_name = os.getenv("LLM_DEPLOYMENT_NAME")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
    default_headers={
        "Ocp-Apim-Subscription-Key": api_key,
        "api-key": api_key,
    },
)


def call_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    top_p: float = 1.0,
) -> str:
    """Send a prompt to the configured LLM and return the response text.

    Args:
        prompt: Fully formatted prompt string.
        temperature: Sampling temperature. Use 0.0 for deterministic output.
        max_tokens: Maximum tokens in the response.
        top_p: Nucleus sampling parameter.

    Returns:
        Raw response string from the LLM.
    """
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return completion.choices[0].message.content