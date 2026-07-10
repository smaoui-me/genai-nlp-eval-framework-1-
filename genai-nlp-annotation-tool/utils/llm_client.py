"""
This file is the only place in the app that actually talks to the LLM.
Every other file imports `call_llm` from here instead of writing its own
network-request code.

Credential lookup order (endpoint URL, API key, model name):
1. `st.secrets` — Streamlit's built-in way to store secrets. Locally it
   reads `.streamlit/secrets.toml`; when deployed on Streamlit Community
   Cloud it reads that app's Secrets dashboard instead — same code either way.
2. A local `.env` file — a fallback for standalone scripts run with plain
   `python`, outside `streamlit run`, where `st.secrets` isn't available.
3. The eval-framework project's `.env`, if it's checked out next to this
   project — lets the same lab credentials be reused without copying them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

_APP_DIR = Path(__file__).resolve().parent.parent
_FALLBACK_ENV = _APP_DIR.parent / "genai-nlp-eval-framework-1-" / ".env"


def _load_credentials() -> tuple[str | None, str | None, str | None]:
    """Work out (endpoint, api_key, deployment_name). Any can be None if nothing was found."""
    try:
        import streamlit as st

        secrets = st.secrets
        if "LLM_API_KEY" in secrets:
            return secrets.get("LLM_ENDPOINT"), secrets.get("LLM_API_KEY"), secrets.get("LLM_DEPLOYMENT_NAME")
    except Exception:
        pass  # no secrets.toml / not running inside Streamlit — fall through to .env

    load_dotenv(_APP_DIR / ".env")  # reads KEY=value lines from .env into os.environ
    if not os.getenv("LLM_API_KEY") and _FALLBACK_ENV.exists():
        load_dotenv(_FALLBACK_ENV)
    return os.getenv("LLM_ENDPOINT"), os.getenv("LLM_API_KEY"), os.getenv("LLM_DEPLOYMENT_NAME")


# This runs once, the first time this file is imported — not every time
# call_llm() is called. `OpenAI | None` means "an OpenAI client, or None".
_endpoint, _api_key, _deployment_name = _load_credentials()

_client: OpenAI | None = None
if _api_key:
    _client = OpenAI(
        base_url=_endpoint,
        api_key=_api_key,
        max_retries=0,
        default_headers={
            "Ocp-Apim-Subscription-Key": _api_key,
            "api-key": _api_key,
        },
    )


def is_configured() -> bool:
    """Whether LLM credentials were found."""
    return _client is not None


def missing_credentials_message() -> str:
    """User-facing fix instructions, shared by every page that shows this warning."""
    return (
        "**Locally:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
        "and fill in the three keys.  \n"
        "**On Streamlit Community Cloud:** open this app → ⋮ menu → Settings → Secrets, "
        "paste the same three keys, save, then reboot the app."
    )


def get_model_name() -> str:
    """The configured model/deployment name, or "" if none was found."""
    return _deployment_name or ""


def call_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
    top_p: float = 1.0,
    timeout: float | None = None,
    max_retries: int = 2,
    retry_sleep_seconds: float = 2.0,
    model: str | None = None,
) -> str:
    """Send a prompt to the configured LLM and return the response text.

    `model` is optional: pass a different model/deployment name to use it
    just for this one call (e.g. to try a smaller/cheaper model), or leave
    it as None to use the default model from the configured credentials.
    """
    if _client is None:
        raise RuntimeError("No LLM credentials found. " + missing_credentials_message())

    effective_timeout = timeout if timeout is not None else 60.0
    request_kwargs = {
        "model": model or _deployment_name,  # use the override if one was passed in, else the configured default
        "messages": [{"role": "user", "content": prompt}],  # one user message — no multi-turn chat here
        "temperature": temperature,
        "top_p": top_p,
        # httpx.Timeout sets separate limits per phase of the request
        # (connect, read, ...) instead of one single timeout for everything.
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

    last_error: Exception | None = None
    # Try up to (max_retries + 1) times total: 1 first attempt + up to
    # max_retries retries, waiting a bit longer before each retry.
    for attempt in range(max_retries + 1):
        try:
            completion = _client.chat.completions.create(**request_kwargs)
            return completion.choices[0].message.content
        except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
            # These errors are usually temporary — worth retrying. Any other
            # error (e.g. a bad API key) is not caught, so it fails right away.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_seconds * (attempt + 1))

    raise last_error
