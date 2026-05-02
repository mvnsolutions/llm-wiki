"""Slash-command registry shared by the dispatcher and the suggestion bar.

Each command has a short help string and an optional list of *positional* value
specs. A value spec is either:

- ``("choice", [...])`` — one of a fixed list (rendered as colored options)
- ``("free", "<placeholder>")`` — free-form text (rendered as a dim placeholder)

Subsequent positional values can depend on the first one (e.g. for ``/model``
the model-name choices depend on the chosen provider). Nested branches live in
a ``branches`` dict keyed by the previous arg's resolved value.
"""

from __future__ import annotations

import os.path
import shlex
from dataclasses import dataclass, field
from typing import Literal

from llm_wiki.agent import EFFORT_LEVELS, MODELS, PROVIDER_ALIASES, PROVIDER_LABELS

ChoiceSpec = tuple[Literal["choice"], list[str]]
FreeSpec = tuple[Literal["free"], str]
ValueSpec = ChoiceSpec | FreeSpec


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    args: list[ValueSpec] = field(default_factory=list)
    branches: dict[str, list[ValueSpec]] = field(default_factory=dict)


def _model_branches() -> dict[str, list[ValueSpec]]:
    # Every alias points to the same canonical model list, so users can type
    # `/model gemini` or `/model google` and get the same completions.
    return {
        alias: [("choice", MODELS[canonical])]
        for alias, canonical in PROVIDER_ALIASES.items()
    }


COMMANDS: dict[str, CommandSpec] = {
    "model": CommandSpec(
        name="model",
        help="switch the active model",
        args=[("choice", list(PROVIDER_LABELS.values()))],
        branches=_model_branches(),
    ),
    "effort": CommandSpec(
        name="effort",
        help="set reasoning effort (off | low | medium | high)",
        args=[("choice", list(EFFORT_LEVELS))],
    ),
    "debug": CommandSpec(
        name="debug",
        help="toggle debug view (tool calls). No arg toggles.",
        args=[("choice", ["on", "off"])],
    ),
    "clear": CommandSpec(name="clear", help="clear the conversation"),
    "help":  CommandSpec(name="help",  help="show available commands"),
    "quit":  CommandSpec(name="quit",  help="exit"),
    "exit":  CommandSpec(name="exit",  help="exit"),
}

# Sorted once at import time; dispatch and suggestion bar both hit this on
# every keystroke.
_COMMAND_NAMES: list[str] = sorted(COMMANDS)


def command_names() -> list[str]:
    return _COMMAND_NAMES


# ---------------------------------------------------------------------------- #
# Tab-completion                                                                #
# ---------------------------------------------------------------------------- #


def parse(text: str) -> tuple[list[str], str, int]:
    """Split a slash-input into (committed_tokens, prefix_being_typed, position).

    `position` is 0 when the command name itself is being typed, 1 for the
    first value, 2 for the second, and so on. A trailing space means the
    user has finished typing the previous token, so prefix is empty and
    position points to the *next* slot.
    """
    body = text[1:]  # caller ensures text starts with "/"
    trailing = body.endswith(" ")
    try:
        tokens = shlex.split(body) if body.strip() else []
    except ValueError:
        tokens = body.split()
    if not tokens:
        return [], "", 0
    if trailing:
        return tokens, "", len(tokens)
    return tokens[:-1], tokens[-1], len(tokens) - 1


def value_spec_at(
    spec: CommandSpec, args_after_cmd: list[str], pos: int
) -> ValueSpec | None:
    """Resolve which value spec applies at value-position `pos` (0-indexed)."""
    if pos == 0:
        return spec.args[0] if spec.args else None
    if args_after_cmd:
        first = args_after_cmd[0]
        if first in spec.branches:
            branch = spec.branches[first]
            idx = pos - 1
            return branch[idx] if idx < len(branch) else None
    if pos < len(spec.args):
        return spec.args[pos]
    return None


def candidates(text: str) -> tuple[list[str], str]:
    """Return (matching_candidates, current_prefix) for a slash input."""
    if not text.startswith("/"):
        return [], ""
    committed, prefix, pos = parse(text)
    if pos == 0:
        return [c for c in _COMMAND_NAMES if c.startswith(prefix)], prefix
    cmd = committed[0]
    if cmd not in COMMANDS:
        return [], prefix
    vs = value_spec_at(COMMANDS[cmd], committed[1:], pos - 1)
    if vs is None or vs[0] != "choice":
        return [], prefix
    return [c for c in vs[1] if c.startswith(prefix)], prefix


def complete(text: str) -> str | None:
    """Return the next-step completion for `text`, or None if nothing applies.

    - Single matching candidate → completes to the full token plus a space.
    - Multiple candidates with a longer common prefix → completes to that LCP.
    - Otherwise → returns None (no progress to make).
    """
    if not text.startswith("/"):
        return None
    cands, prefix = candidates(text)
    if not cands:
        return None
    if len(cands) == 1:
        suffix = cands[0] + " "
    else:
        lcp = os.path.commonprefix(cands)
        if len(lcp) <= len(prefix):
            return None
        suffix = lcp
    if prefix:
        return text[: -len(prefix)] + suffix
    return text + suffix
