"""
llm_client.py

Shared LLM client and call_llm function used by all extraction methods.
Reads connection settings from .env and accepts LLM parameters at call time
so configs can control temperature, max_tokens, etc.
"""

from dotenv import load_dotenv
import httpx
import time
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from openai import OpenAI
import os

load_dotenv()

endpoint = os.getenv("LLM_ENDPOINT")
api_key = os.getenv("LLM_API_KEY")
deployment_name = os.getenv("LLM_DEPLOYMENT_NAME")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
    max_retries=0,
    default_headers={
        "Ocp-Apim-Subscription-Key": api_key,
        "api-key": api_key,
    },
)


def get_model_name() -> str:
    """Return the configured deployment/model name."""
    return deployment_name or ""


def call_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    timeout: float | None = None,
    max_retries: int = 2,
    retry_sleep_seconds: float = 2.0,
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
    effective_timeout = timeout if timeout is not None else 60.0
    request_kwargs = {
        "model": deployment_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "timeout": httpx.Timeout(
            timeout=effective_timeout,
            connect=min(10.0, effective_timeout),
            read=effective_timeout,
            write=min(30.0, effective_timeout),
            pool=min(10.0, effective_timeout),
        ),
    }

    if max_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_tokens

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(**request_kwargs)
            return completion.choices[0].message.content
        except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_seconds * (attempt + 1))

    raise last_error
