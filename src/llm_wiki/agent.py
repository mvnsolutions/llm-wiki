"""Deep Agent factory — wires the model, filesystem backend, and checkpointer."""

from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

# LangGraph graph-iteration cap. Not Python stack depth — this bounds the
# tool-call → model → tool-call loop count before LangGraph raises
# GraphRecursionError.
AGENT_RECURSION_LIMIT = 250

# Canonical (init_chat_model) provider id → ordered list of supported models.
# First entry is the default. Single source of truth for the dispatcher
# (default lookup) and the suggestion bar (tab-complete options).
MODELS: dict[str, list[str]] = {
    "anthropic":    ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
    "google_genai": ["gemini-3.1-flash-lite-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
    "openai":       ["gpt-5", "gpt-5-mini", "gpt-4.1"],
}

# Canonical → user-facing label. The label is what the suggestion bar
# advertises; aliases below resolve back to the canonical id.
PROVIDER_LABELS: dict[str, str] = {
    "anthropic":    "anthropic",
    "google_genai": "gemini",
    "openai":       "openai",
}

PROVIDER_ALIASES: dict[str, str] = {
    "anthropic":    "anthropic",
    "claude":       "anthropic",
    "gemini":       "google_genai",
    "google":       "google_genai",
    "google_genai": "google_genai",
    "openai":       "openai",
    "gpt":          "openai",
}

DEFAULT_MODELS: dict[str, str] = {p: ms[0] for p, ms in MODELS.items()}

# Reasoning effort — canonical levels and aliases the user can type. Maps
# to per-provider knobs (Anthropic extended thinking, OpenAI reasoning_effort,
# Gemini thinking_budget). "off" disables the reasoning surcharge.
EFFORT_LEVELS = ("off", "low", "medium", "high")
EFFORT_ALIASES: dict[str, str] = {
    "off":    "off",
    "low":    "low",
    "medium": "medium",
    "middle": "medium",
    "mid":    "medium",
    "high":   "high",
}

_EFFORT_BUDGETS = {
    "anthropic":    {"low": 2_000,  "medium": 8_000,  "high": 16_000},
    "google_genai": {"low": 1_024,  "medium": 8_192,  "high": 24_576},
}


def _effort_kwargs(provider: str, level: str) -> dict:
    if level == "off":
        return {}
    if provider == "anthropic":
        budget = _EFFORT_BUDGETS["anthropic"][level]
        # Anthropic requires max_tokens > budget_tokens. Give the response a
        # fair budget on top of the thinking budget.
        return {
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "max_tokens": budget * 2 + 4_000,
        }
    if provider == "openai":
        return {"reasoning_effort": level}
    if provider == "google_genai":
        return {"thinking_budget": _EFFORT_BUDGETS["google_genai"][level]}
    return {}


def normalize_effort(level: str) -> str:
    key = level.strip().lower()
    if key not in EFFORT_ALIASES:
        raise ValueError(
            f"unknown effort level {level!r}; use one of: "
            f"{' | '.join(EFFORT_LEVELS)}"
        )
    return EFFORT_ALIASES[key]

SYSTEM_PROMPT_TEMPLATE = """You are llm-wiki, a terminal coding assistant.

Working directory: {cwd}

The filesystem tools (ls, read_file, write_file, edit_file, glob, grep) operate on the
real filesystem. Default to paths relative to the working directory above; only use an
absolute path when the user explicitly refers to one. Calling `ls` with no path or with
"." lists the working directory.

You also have a planning tool (write_todos) and can delegate to subagents (task).

Be concise. Prefer reading files over guessing. When making code changes,
use edit_file rather than rewriting entire files.
"""


def normalize_provider(name: str) -> str:
    """Resolve a user-typed provider alias to its canonical id."""
    key = name.strip().lower()
    if key not in PROVIDER_ALIASES:
        raise ValueError(
            f"unknown provider {name!r}; expected one of: "
            f"{', '.join(sorted(set(PROVIDER_ALIASES)))}"
        )
    return PROVIDER_ALIASES[key]


def default_model_for(provider: str) -> str:
    return DEFAULT_MODELS[normalize_provider(provider)]


def build_agent(
    provider: str = "anthropic",
    model: str | None = None,
    working_dir: str | Path | None = None,
    effort: str = "off",
):
    """Construct the deep agent. Returns a compiled LangGraph runnable."""
    canonical = normalize_provider(provider)
    chosen = model or DEFAULT_MODELS[canonical]
    level = normalize_effort(effort)
    root = str(Path(working_dir or os.getcwd()).resolve())

    chat_model = init_chat_model(
        f"{canonical}:{chosen}", **_effort_kwargs(canonical, level)
    )

    return create_deep_agent(
        model=chat_model,
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(cwd=root),
        backend=FilesystemBackend(root_dir=root, virtual_mode=False),
        checkpointer=MemorySaver(),
        memory=["./AGENTS.md"],
    )


def startup_defaults() -> tuple[str, str]:
    """Read provider/model from env, returning (provider, model)."""
    provider = normalize_provider(os.environ.get("LLM_WIKI_PROVIDER", "anthropic"))
    model = os.environ.get("LLM_WIKI_MODEL") or DEFAULT_MODELS[provider]
    return provider, model
