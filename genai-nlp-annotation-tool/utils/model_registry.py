"""
The list of LLM endpoints the app can talk to.

Until now the app could only use one model: the endpoint from the project
credentials. That was fine for a demo, but two things pushed us to make it a
list instead:

1. Someone reviewing annotations may want a cheaper/faster model for a first
   pass and a stronger one for a second opinion.
2. Comparing two models against each other is itself useful — where they
   disagree is exactly where a human should look (see utils/uncertainty.py).

## How to add another endpoint

Anything that speaks the OpenAI chat-completions API works, which is most
things nowadays. Add a block to `.streamlit/secrets.toml` (locally) or to the
Secrets box in the Streamlit Cloud dashboard:

    [providers.my_provider]
    label    = "My provider"          # shown in the dropdown
    endpoint = "https://..../v1"
    api_key  = "sk-..."
    models   = ["model-a", "model-b"]
    supports_logprobs = true          # optional, default false

That is the only change needed — no Python edits.

## Free options, if you don't have another key

- **Ollama** (https://ollama.com) runs models on the machine that runs this
  app, needs no key and no account. Install it, run `ollama pull qwen3:4b`, and
  the preset below finds it.

  One thing to be clear about: "local" means local *to the Streamlit server*,
  not to the browser. If you run `streamlit run app.py` on your laptop, Ollama
  on that same laptop works and no text leaves the machine. On the deployed
  Community Cloud app, `localhost` is the cloud container, which has no Ollama
  installed, so the option is offered but will fail with a connection error.
  That is why `ollama_available()` below checks before we show it as usable.
- Google AI Studio, Groq and OpenRouter all have free tiers with rate limits
  and OpenAI-compatible endpoints, so they fit the block above as-is. They are
  free as in "no invoice", not free as in "no data leaves the building".

OpenAI's own small models (gpt-4.1-nano, gpt-5-nano) are cheap but not free;
they still need a paid API key.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent.parent
_FALLBACK_ENV = _APP_DIR.parent / ".env"


@dataclass(frozen=True)
class Provider:
    """One place we can send prompts to, plus which models it offers."""

    key: str                       # short internal id, e.g. "project"
    label: str                     # what the dropdown shows
    endpoint: str | None
    api_key: str | None
    models: tuple[str, ...]
    supports_logprobs: bool = False
    needs_key: bool = True         # Ollama and friends run locally with no key

    @property
    def configured(self) -> bool:
        """Can we actually call this provider right now?"""
        if not self.endpoint:
            return False
        return bool(self.api_key) or not self.needs_key


@dataclass(frozen=True)
class ModelChoice:
    """One (provider, model) pair — this is what the UI dropdown holds."""

    provider: Provider
    model: str

    @property
    def id(self) -> str:
        """Stable id used in session state and written into gold exports."""
        return f"{self.provider.key}:{self.model}"

    @property
    def label(self) -> str:
        """What the dropdown shows, e.g. "gpt-5.4 (hosted)"."""
        return f"{self.model} ({self.provider.label})"

    @property
    def short(self) -> str:
        """Just the model name — used in the review table's Source column.

        The provider prefix matters when picking a model, but in a table of
        forty rows "gpt-5.4" is what the reviewer needs, not
        "hosted:gpt-5.4".
        """
        return self.model


def _read_secrets() -> dict:
    """Read st.secrets if we're inside Streamlit, otherwise return {}.

    Wrapped in try/except because this module is also imported by plain
    scripts run with `python`, where Streamlit's secrets machinery either
    isn't available or raises when no secrets file exists.
    """
    try:
        import streamlit as st

        secrets = dict(st.secrets)
        if secrets:
            return secrets
    except Exception:
        pass
    # Batch scripts are normally launched from the repository root, where
    # Streamlit does not discover the app-local secrets file automatically.
    local_file = _APP_DIR / ".streamlit" / "secrets.toml"
    if local_file.exists():
        with local_file.open("rb") as handle:
            return tomllib.load(handle)
    return {}


def _load_env() -> None:
    """Load .env files so standalone scripts get the same credentials."""
    load_dotenv(_APP_DIR / ".env")
    if not os.getenv("LLM_API_KEY") and _FALLBACK_ENV.exists():
        load_dotenv(_FALLBACK_ENV)


def _project_provider(secrets: dict) -> Provider:
    """The endpoint from the project credentials — the one that always exists."""
    endpoint = secrets.get("LLM_ENDPOINT") or os.getenv("LLM_ENDPOINT")
    api_key = secrets.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    deployment = secrets.get("LLM_DEPLOYMENT_NAME") or os.getenv("LLM_DEPLOYMENT_NAME") or ""

    # Extra deployment names can be listed in secrets as
    #   LLM_EXTRA_DEPLOYMENTS = ["gpt-4.1-mini", "gpt-4.1-nano"]
    # for gateways that expose more than one. They are offered in the
    # dropdown but may 404 if the gateway doesn't actually have them.
    extra = secrets.get("LLM_EXTRA_DEPLOYMENTS") or []
    if isinstance(extra, str):
        extra = [m.strip() for m in extra.split(",") if m.strip()]

    models = tuple(dict.fromkeys([m for m in [deployment, *extra] if m]))
    return Provider(
        key="hosted",
        label="hosted",
        endpoint=endpoint,
        api_key=api_key,
        models=models or ("gpt-5.4",),
        # Whether the gateway returns token logprobs is not knowable up front,
        # so we ask for them and cope if they don't come back — see
        # utils/llm_client.py.
        supports_logprobs=bool(secrets.get("LLM_SUPPORTS_LOGPROBS", False)),
    )


def _ollama_provider(secrets: dict) -> Provider:
    """A local Ollama server, if one is running. Free, offline, no key."""
    endpoint = (
        secrets.get("OLLAMA_ENDPOINT")
        or os.getenv("OLLAMA_ENDPOINT")
        or "http://localhost:11434/v1"
    )
    models = secrets.get("OLLAMA_MODELS") or os.getenv("OLLAMA_MODELS") or "qwen3:4b,llama3.2:3b"
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    return Provider(
        key="ollama",
        label="on your machine",
        endpoint=endpoint,
        api_key="ollama",  # the OpenAI client wants *some* string; Ollama ignores it
        models=tuple(models),
        supports_logprobs=False,
        needs_key=False,
    )


def _custom_providers(secrets: dict) -> list[Provider]:
    """Any extra providers declared under [providers.*] in secrets.toml."""
    providers = []
    for key, cfg in (secrets.get("providers") or {}).items():
        if not isinstance(cfg, dict):
            continue
        models = cfg.get("models") or []
        if isinstance(models, str):
            models = [m.strip() for m in models.split(",") if m.strip()]
        providers.append(
            Provider(
                key=str(key),
                label=str(cfg.get("label", key)),
                endpoint=cfg.get("endpoint"),
                api_key=cfg.get("api_key"),
                models=tuple(models),
                supports_logprobs=bool(cfg.get("supports_logprobs", False)),
                needs_key=bool(cfg.get("needs_key", True)),
            )
        )
    return providers


# Built once on import. Streamlit re-imports modules rarely enough that this
# is fine, and it keeps credential reading out of the render loop.
_load_env()
_SECRETS = _read_secrets()

PROVIDERS: dict[str, Provider] = {}
for _p in [_project_provider(_SECRETS), *_custom_providers(_SECRETS), _ollama_provider(_SECRETS)]:
    PROVIDERS[_p.key] = _p


def ollama_available(timeout_seconds: float = 0.4) -> bool:
    """Is an Ollama server actually reachable from wherever this app runs?

    We ask instead of assuming, because the answer differs by deployment. Run
    locally, Ollama on the same laptop answers instantly. On Streamlit
    Community Cloud, `localhost` is the cloud container and nothing answers, so
    offering the option there just produces a confusing connection error later.

    The timeout is deliberately short: this runs while the page is drawing, and
    a slow check would be worse than a missing dropdown entry.
    """
    provider = PROVIDERS.get("ollama")
    if provider is None or not provider.endpoint:
        return False
    try:
        import httpx

        # /api/tags is Ollama's "what models do I have" endpoint. The OpenAI
        # compatibility layer lives under /v1, so strip that off first.
        base = provider.endpoint.rsplit("/v1", 1)[0]
        return httpx.get(f"{base}/api/tags", timeout=timeout_seconds).status_code == 200
    except Exception:
        return False


def available_choices(include_unconfigured: bool = False) -> list[ModelChoice]:
    """Every (provider, model) pair we could send a prompt to.

    Ollama is only listed when a server actually answers, so the dropdown never
    offers a model that is guaranteed to fail.
    """
    choices: list[ModelChoice] = []
    ollama_up = ollama_available()
    for provider in PROVIDERS.values():
        if not provider.configured and not include_unconfigured and provider.needs_key:
            continue
        if provider.key == "ollama" and not ollama_up and not include_unconfigured:
            continue
        for model in provider.models:
            choices.append(ModelChoice(provider=provider, model=model))
    return choices


def default_choice() -> ModelChoice | None:
    """The project endpoint's first model — what everything defaults to."""
    choices = available_choices()
    for choice in choices:
        if choice.provider.key == "hosted":
            return choice
    return choices[0] if choices else None


def choice_by_id(choice_id: str | None) -> ModelChoice | None:
    """Look a choice back up from the id stored in session state."""
    if not choice_id:
        return default_choice()
    for choice in available_choices(include_unconfigured=True):
        if choice.id == choice_id:
            return choice
    return default_choice()


def is_any_configured() -> bool:
    return any(p.configured for p in PROVIDERS.values() if p.needs_key)
