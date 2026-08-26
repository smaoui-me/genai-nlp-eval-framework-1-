"""
This file is the only place in the app that actually talks to an LLM.
Every other file imports `call_llm` from here instead of writing its own
network-request code.

Two things changed from the first version:

1. **More than one endpoint.** Which endpoints exist is described in
   utils/model_registry.py; this file just knows how to call one.
2. **Token log-probabilities.** When the endpoint supports it we ask for
   `logprobs`, which says how sure the model was about each token it
   produced. That is the cheapest confidence signal available, since it costs
   one normal call, and utils/uncertainty.py turns it into a per-entity score.
   Not every gateway returns them, so the code always copes with them missing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from utils.model_registry import ModelChoice, Provider, default_choice, is_any_configured


@dataclass
class TokenLogprob:
    """One generated token and how confident the model was about it."""

    token: str
    logprob: float

    @property
    def prob(self) -> float:
        """Convert a log-probability back to a plain 0..1 probability."""
        return math.exp(self.logprob)


@dataclass
class LLMResponse:
    """What one call to the model gave us back."""

    text: str
    model_id: str                                   # e.g. "project:gpt-5.4"
    token_logprobs: list[TokenLogprob] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_reported: bool = False

    @property
    def has_logprobs(self) -> bool:
        return bool(self.token_logprobs)


# ---------------------------------------------------------------------------
# Clients — one per provider, built on first use and then reused
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _client_for(provider_key: str, endpoint: str, api_key: str | None) -> OpenAI:
    """Build (once) an OpenAI-compatible client for one provider.

    The two extra headers are there because the project gateway is an Azure
    API Management front end, which expects the key under its own header names
    rather than the standard Authorization one. Sending all three is harmless
    for providers that only read one of them.
    """
    return OpenAI(
        base_url=endpoint,
        api_key=api_key or "not-needed",
        max_retries=0,  # we do our own retries below, so we control the wait
        default_headers={
            "Ocp-Apim-Subscription-Key": api_key or "",
            "api-key": api_key or "",
        },
    )


def is_configured() -> bool:
    """Whether at least one provider has credentials."""
    return is_any_configured()


def missing_credentials_message() -> str:
    """User-facing fix instructions, shared by every page that shows this warning."""
    return (
        "**Locally:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
        "and fill in the three `LLM_*` keys.  \n"
        "**On Streamlit Community Cloud:** open this app → ⋮ menu → Settings → Secrets, "
        "paste the same keys, save, then reboot the app.  \n"
        "**No key at all?** Install [Ollama](https://ollama.com), run "
        "`ollama pull qwen3:4b`, and pick the *Local (Ollama)* model in the dropdown."
    )


def get_model_name() -> str:
    """The default model name, or "" if nothing is configured."""
    choice = default_choice()
    return choice.model if choice else ""


# ---------------------------------------------------------------------------
# The actual call
# ---------------------------------------------------------------------------


def call_llm_full(
    prompt: str,
    choice: ModelChoice | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    top_p: float = 1.0,
    timeout: float | None = None,
    max_retries: int = 2,
    retry_sleep_seconds: float = 2.0,
    want_logprobs: bool = False,
    seed: int | None = None,
) -> LLMResponse:
    """Send one prompt and return the full response object.

    Args:
        choice: which (provider, model) to use. Defaults to the project endpoint.
        want_logprobs: ask the endpoint for per-token confidence. Dropped
            silently if the endpoint rejects it, and simply absent from the
            result if the endpoint ignores it.
        seed: passed through where supported, so repeated runs are reproducible.
    """
    choice = choice or default_choice()
    if choice is None:
        raise RuntimeError("No LLM credentials found. " + missing_credentials_message())

    provider: Provider = choice.provider
    if not provider.configured:
        raise RuntimeError(
            f"Provider '{provider.label}' is not configured. " + missing_credentials_message()
        )

    client = _client_for(provider.key, provider.endpoint, provider.api_key)
    effective_timeout = timeout if timeout is not None else 60.0

    request_kwargs: dict = {
        "model": choice.model,
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
    if seed is not None:
        request_kwargs["seed"] = seed
    if want_logprobs:
        request_kwargs["logprobs"] = True

    last_error: Exception | None = None
    # Try up to (max_retries + 1) times: one first attempt plus retries,
    # waiting a bit longer before each one.
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(**request_kwargs)
            return _to_response(completion, choice)
        except APIStatusError as exc:
            # Some gateways reject unknown parameters outright with a 400. If
            # that is because we asked for logprobs or a seed, drop them and
            # retry instead of failing the whole annotation run.
            dropped = False
            if exc.status_code == 400 and "logprobs" in request_kwargs:
                request_kwargs.pop("logprobs", None)
                dropped = True
            if exc.status_code == 400 and "seed" in request_kwargs:
                request_kwargs.pop("seed", None)
                dropped = True
            if not dropped:
                raise
            last_error = exc
        except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
            # Usually temporary, so worth retrying. Anything else (a bad key,
            # for instance) is not caught and fails immediately.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_seconds * (attempt + 1))

    raise last_error if last_error else RuntimeError("LLM call failed for an unknown reason")


def _to_response(completion, choice: ModelChoice) -> LLMResponse:
    """Pull the text and (if present) the token logprobs out of the API object."""
    message = completion.choices[0].message
    text = message.content or ""

    token_logprobs: list[TokenLogprob] = []
    # The logprobs block is optional and its exact shape differs a little
    # between providers, so read it defensively rather than trusting it.
    logprobs_obj = getattr(completion.choices[0], "logprobs", None)
    content = getattr(logprobs_obj, "content", None) if logprobs_obj else None
    if content:
        for item in content:
            tok = getattr(item, "token", None)
            lp = getattr(item, "logprob", None)
            if tok is not None and lp is not None:
                token_logprobs.append(TokenLogprob(token=tok, logprob=float(lp)))

    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    usage_reported = prompt_tokens is not None and completion_tokens is not None
    input_tokens = int(prompt_tokens or 0)
    output_tokens = int(completion_tokens or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0) if usage else 0
    return LLMResponse(
        text=text, model_id=choice.id, token_logprobs=token_logprobs,
        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
        usage_reported=usage_reported,
    )


def call_llm(prompt: str, **kwargs) -> str:
    """Text-only version, kept because most call sites only want the string.

    Accepts `model` as a plain string for backwards compatibility with the
    older code, in which case it is looked up in the registry.
    """
    model = kwargs.pop("model", None)
    if model is not None and "choice" not in kwargs:
        kwargs["choice"] = resolve_model_string(model)
    return call_llm_full(prompt, **kwargs).text


def resolve_model_string(model: str) -> ModelChoice | None:
    """Turn a bare model name (or a "provider:model" id) into a ModelChoice."""
    from utils.model_registry import available_choices, choice_by_id

    if ":" in model:
        return choice_by_id(model)
    for choice in available_choices(include_unconfigured=True):
        if choice.model == model:
            return choice
    # An unknown name is treated as a deployment on the project endpoint,
    # which is what the old free-text model box allowed. That still works.
    base = default_choice()
    return ModelChoice(provider=base.provider, model=model) if base else None
