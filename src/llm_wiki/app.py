"""Textual TUI for the llm-wiki deep agent."""

from __future__ import annotations

import getpass
import json
import shlex
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage
from rich.markup import escape
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

from llm_wiki.agent import (
    AGENT_RECURSION_LIMIT,
    EFFORT_LEVELS,
    build_agent,
    default_model_for,
    normalize_effort,
    normalize_provider,
    startup_defaults,
)
from llm_wiki.commands import (
    COMMANDS,
    CommandSpec,
    command_names,
    complete as complete_command,
    parse as parse_command,
    value_spec_at,
)


# ---------------------------------------------------------------------------- #
# Conversation widgets                                                         #
# ---------------------------------------------------------------------------- #


@dataclass
class _Turn:
    role: str
    widget: Markdown
    text: str = ""


class ChatLog(VerticalScroll):
    """Non-focusable so Tab can't steal focus from the prompt input."""

    can_focus = False

    DEFAULT_CSS = """
    ChatLog { padding: 1 2; background: $surface; }
    ChatLog > .turn-user        { color: $text 70%; margin-top: 1; }
    ChatLog > .turn-assistant   { margin-bottom: 1; }
    ChatLog > .turn-system      { color: $text-muted; text-style: italic; margin: 0 0 1 0; }
    ChatLog > .turn-thinking    { color: $text-muted; text-style: italic; margin: 0 0 1 0; }
    ChatLog > .turn-tool-call   { color: $accent; margin-top: 1; }
    ChatLog > .turn-tool-result { color: $text-muted 80%; margin: 0 0 1 0; }
    """

    def add_turn(self, role: str, initial_text: str = "") -> _Turn:
        """Mount a non-streaming turn whose content is final at creation."""
        widget = Markdown(initial_text)
        widget.add_class(f"turn-{role}")
        self.mount(widget)
        self.scroll_end(animate=False)
        return _Turn(role=role, widget=widget, text=initial_text)

    async def add_streaming_turn(self, role: str) -> _Turn:
        """Mount a turn that will be progressively updated via append_to.

        Subsequent .update() calls are silently dropped against an unmounted
        widget, so streaming turns must wait for mount to complete.
        """
        widget = Markdown("")
        widget.add_class(f"turn-{role}")
        await self.mount(widget)
        self.scroll_end(animate=False)
        return _Turn(role=role, widget=widget, text="")

    async def append_to(self, turn: _Turn, delta: str) -> None:
        if not delta:
            return
        turn.text += delta
        await turn.widget.update(turn.text)
        self.scroll_end(animate=False)

    def add_tool_call(self, name: str, tool_input: object) -> None:
        body = _format_payload(tool_input)
        self.add_turn("tool-call", f"**▸ {name}**\n\n```json\n{body}\n```")

    def add_tool_result(self, name: str, output: object) -> None:
        body = _format_payload(output, max_chars=600)
        self.add_turn("tool-result", f"**◂ {name}**\n\n```\n{body}\n```")


# ---------------------------------------------------------------------------- #
# Prompt input                                                                  #
# ---------------------------------------------------------------------------- #


class PromptInput(TextArea):
    """Multi-line input. Enter submits; Shift+Enter inserts a newline."""

    DEFAULT_CSS = """
    PromptInput {
        border: none;
        background: transparent;
        padding: 0;
        height: auto;
        min-height: 1;
        max-height: 10;
        scrollbar-size: 0 0;
    }
    PromptInput:focus {
        border: none;
    }
    PromptInput > .text-area--cursor-line {
        background: transparent;
    }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def _on_key(self, event: events.Key) -> None:
        # Shift+Enter requires a terminal that emits it as a distinct key
        # (Kitty keyboard protocol). Ctrl+J is the universal fallback.
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self._submit()
            return
        if event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            self._try_complete()
            return

    def _try_complete(self) -> None:
        completed = complete_command(self.text)
        if completed is None or completed == self.text:
            return
        self.text = completed
        self.move_cursor(self.document.end)

    def _submit(self) -> None:
        value = self.text.strip()
        if not value:
            return
        self.post_message(self.Submitted(value))
        self.clear()


# ---------------------------------------------------------------------------- #
# Suggestion bar — colors slash commands and shows value options                #
# ---------------------------------------------------------------------------- #


class SuggestionBar(Static):
    DEFAULT_CSS = """
    SuggestionBar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $surface;
    }
    """

    IDLE_HINT = (
        "[dim]enter[/] send · [dim]shift+enter[/] newline · "
        "type [bold cyan]/[/] for commands"
    )

    def show_idle(self) -> None:
        self.update(self.IDLE_HINT)

    def show_for(self, text: str) -> None:
        if not text.startswith("/"):
            self.show_idle()
            return

        committed, prefix, pos = parse_command(text)

        if pos == 0:
            self._render_command_name(prefix, raw_after_slash=text[1:])
            return

        cmd = committed[0]
        if cmd not in COMMANDS:
            self.update(
                f"[red]unknown[/] [bold]/{escape(cmd)}[/]  ·  type [bold cyan]/help[/]"
            )
            return

        self._render_value(COMMANDS[cmd], committed[1:], prefix, pos=pos)

    def _render_command_name(self, prefix: str, *, raw_after_slash: str) -> None:
        if not raw_after_slash:
            opts = " ".join(f"[bold cyan]/{c}[/]" for c in command_names())
            self.update(f"[dim]commands:[/] {opts}")
            return
        if prefix in COMMANDS:
            # Exact match — preview the command's first value, as if the user
            # had already typed a trailing space.
            self._render_value(COMMANDS[prefix], [], "", pos=1)
            return
        matches = [c for c in command_names() if c.startswith(prefix)]
        if matches:
            opts = "  ".join(f"[bold cyan]/{m}[/]" for m in matches)
            self.update(f"[dim]matches:[/] {opts}")
        else:
            self.update(
                f"[red]unknown[/] [bold]/{escape(prefix)}[/]  ·  type [bold cyan]/help[/]"
            )

    def _render_value(
        self,
        spec: CommandSpec,
        committed_values: list[str],
        prefix: str,
        *,
        pos: int,
    ) -> None:
        head = f"[bold cyan]/{spec.name}[/]"
        vs = value_spec_at(spec, committed_values, pos - 1)
        if vs is None:
            self.update(f"{head}  [dim]·[/]  {escape(spec.help)}")
            return
        kind, payload = vs
        if kind == "choice":
            visible = [c for c in payload if c.startswith(prefix)] or payload
            opts = "  ".join(self._color_choice(c, prefix) for c in visible)
            label = "values" if pos == 1 else "options"
            self.update(f"{head}  [dim]·[/]  [dim]{label}:[/] {opts}")
        else:
            self.update(f"{head}  [dim]·[/]  [dim]{escape(payload)}[/]")

    @staticmethod
    def _color_choice(choice: str, prefix: str) -> str:
        if prefix and choice.startswith(prefix):
            return f"[bold magenta]{escape(choice)}[/]"
        return f"[magenta]{escape(choice)}[/]"


# ---------------------------------------------------------------------------- #
# App                                                                           #
# ---------------------------------------------------------------------------- #


class LLMWikiApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #chat { height: 1fr; }

    #input-row {
        height: auto;
        layout: horizontal;
        padding: 0 2;
        border-top: solid $primary 30%;
        border-bottom: solid $primary 30%;
        background: $surface;
    }
    #prompt-mark {
        width: 2;
        color: $accent;
        content-align: left middle;
        padding: 0;
    }
    /* Multi-line input — anchor the arrow to the top so it sits next to
       the first line, not in the visual middle of a tall box. */
    #input-row.multiline #prompt-mark {
        content-align: left top;
    }

    #status {
        dock: bottom;
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Pin the agent's filesystem root to where the TUI was launched, so
        # /model switches (which rebuild the agent) keep the same root even
        # if the process cwd has changed in the meantime.
        self.launch_dir = Path.cwd().resolve()
        self.provider, self.model = startup_defaults()
        self.effort = "off"
        self.agent = self._build_agent()
        self.thread_id = self._new_thread_id()
        self._busy = False
        self._debug = False
        self._status_prefix = self._build_status_prefix()

    # ----------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat")
        yield SuggestionBar(id="suggestions")
        with Horizontal(id="input-row"):
            yield Static("❯", id="prompt-mark")
            yield PromptInput(id="prompt", language=None, soft_wrap=True)
        yield Static("", id="status")

    def on_mount(self) -> None:
        self.title = "llm-wiki"
        self._refresh_status("ready")
        self.query_one(SuggestionBar).show_idle()
        self._system_message(
            "**llm-wiki** — type a message, or `/help` to list commands. "
            "Press **Enter** to send, **Shift+Enter** for a newline."
        )
        self.query_one(PromptInput).focus()

    # ------------------------------------------------------------------ events

    @on(PromptInput.Submitted)
    def _on_submit(self, event: PromptInput.Submitted) -> None:
        if self._busy:
            return
        text = event.value
        self.query_one(SuggestionBar).show_idle()
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._run_turn(text)

    @on(TextArea.Changed, "#prompt")
    def _on_text_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        self.query_one(SuggestionBar).show_for(text)
        self.query_one("#input-row").set_class("\n" in text, "multiline")

    def action_clear(self) -> None:
        self._do_clear()

    # ---------------------------------------------------------------- commands

    def _handle_command(self, raw: str) -> None:
        try:
            parts = shlex.split(raw[1:])
        except ValueError as exc:
            self._system_message(f"bad command: {exc}")
            return
        if not parts:
            self._system_message("empty command — try `/help`")
            return
        name, *args = parts
        if name not in COMMANDS:
            self._system_message(f"unknown command `/{name}` — try `/help`")
            return
        handler = {
            "model":  self._cmd_model,
            "effort": self._cmd_effort,
            "debug":  self._cmd_debug,
            "clear":  lambda _a: self._do_clear(),
            "help":   lambda _a: self._cmd_help(),
            "quit":   lambda _a: self.exit(),
            "exit":   lambda _a: self.exit(),
        }[name]
        handler(args)

    def _cmd_effort(self, args: list[str]) -> None:
        if not args:
            self._system_message(
                f"reasoning effort: **`{self.effort}`**  ·  "
                f"levels: {' | '.join(EFFORT_LEVELS)}"
            )
            return
        try:
            level = normalize_effort(args[0])
        except ValueError as exc:
            self._system_message(str(exc))
            return
        self._switch_effort(level)

    def _switch_effort(self, level: str) -> None:
        prev = self.effort
        self.effort = level
        try:
            self.agent = self._build_agent()
        except Exception as exc:
            self.effort = prev
            self._system_message(f"failed to apply effort={level}: `{exc}`")
            return
        self.thread_id = self._new_thread_id()
        self._system_message(f"effort **`{level}`** (conversation reset)")
        self._refresh_status("ready")

    def _cmd_debug(self, args: list[str]) -> None:
        if not args:
            new = not self._debug
        elif args[0].lower() in ("on", "true", "1"):
            new = True
        elif args[0].lower() in ("off", "false", "0"):
            new = False
        else:
            self._system_message("usage: `/debug [on|off]` (no arg toggles)")
            return
        self._debug = new
        self._system_message(f"debug **{'on' if new else 'off'}**")
        self._refresh_status("ready")

    def _cmd_help(self) -> None:
        lines = ["**commands**", ""]
        for name in command_names():
            lines.append(f"- `/{name}` — {COMMANDS[name].help}")
        self._system_message("\n".join(lines))

    def _cmd_model(self, args: list[str]) -> None:
        if not args:
            self._system_message(
                f"current model: **`{self.provider}:{self.model}`**\n\n"
                "Usage: `/model <anthropic|gemini|openai> [name]`."
            )
            return
        try:
            provider = normalize_provider(args[0])
        except ValueError as exc:
            self._system_message(str(exc))
            return
        model = args[1] if len(args) >= 2 else default_model_for(provider)
        self._switch_model(provider, model)

    def _switch_model(self, provider: str, model: str) -> None:
        prev = (self.provider, self.model)
        self.provider, self.model = provider, model
        try:
            self.agent = self._build_agent()
        except Exception as exc:
            self.provider, self.model = prev
            self._system_message(f"failed to build agent: `{exc}`")
            return
        self.thread_id = self._new_thread_id()
        self._system_message(f"switched to **`{provider}:{model}`** (conversation reset)")
        self._refresh_status("ready")

    def _build_agent(self):
        return build_agent(
            self.provider,
            self.model,
            working_dir=self.launch_dir,
            effort=self.effort,
        )

    def _do_clear(self) -> None:
        if self._busy:
            return
        self.thread_id = self._new_thread_id()
        self.query_one("#chat", ChatLog).remove_children()
        self._refresh_status("cleared")

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _new_thread_id() -> str:
        return str(uuid.uuid4())

    def _system_message(self, markdown: str) -> None:
        self.query_one("#chat", ChatLog).add_turn("system", markdown)

    def _build_status_prefix(self) -> str:
        host = socket.gethostname().split(".")[0]
        try:
            cwd = "~/" + self.launch_dir.relative_to(Path.home()).as_posix()
        except ValueError:
            cwd = str(self.launch_dir)
        return f"{getpass.getuser()}@{host}  |  {cwd}"

    def _refresh_status(self, state: str) -> None:
        parts = [
            self._status_prefix,
            f"{self.provider}:{self.model}",
            state,
        ]
        if self._debug:
            parts.append("debug")
        parts.append(f"effort:{self.effort}")
        self.query_one("#status", Static).update("  |  ".join(parts))

    # ------------------------------------------------------------------ worker

    @work(exclusive=True)
    async def _run_turn(self, user_text: str) -> None:
        self._busy = True
        self._refresh_status("thinking…")
        chat = self.query_one("#chat", ChatLog)
        chat.add_turn("user", f"> {user_text}")

        # text and thinking each get their own widget that we keep appending
        # to, until a tool call (in debug mode) or the other content kind
        # interrupts and forces a fresh widget.
        text_turn: _Turn | None = None
        think_turn: _Turn | None = None

        config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }
        inputs = {"messages": [{"role": "user", "content": user_text}]}

        try:
            async for event in self.agent.astream_events(
                inputs, config=config, version="v2"
            ):
                kind = event.get("event", "")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    text, thinking = _split_chunk(chunk)
                    if thinking:
                        text_turn = None
                        if think_turn is None:
                            think_turn = await chat.add_streaming_turn("thinking")
                            await chat.append_to(think_turn, "_thinking_\n\n")
                        await chat.append_to(think_turn, thinking)
                    if text:
                        think_turn = None
                        if text_turn is None:
                            text_turn = await chat.add_streaming_turn("assistant")
                        await chat.append_to(text_turn, text)
                    continue

                # Tool calls only render in debug mode. When hidden, we don't
                # reset text_turn, so the assistant's reply before and after a
                # tool call flows as a single message.
                if kind == "on_tool_start":
                    if self._debug:
                        text_turn = think_turn = None
                        chat.add_tool_call(
                            event.get("name", "tool"),
                            event.get("data", {}).get("input"),
                        )
                    continue

                if kind == "on_tool_end":
                    if self._debug:
                        text_turn = think_turn = None
                        chat.add_tool_result(
                            event.get("name", "tool"),
                            event.get("data", {}).get("output"),
                        )
                    continue
        except Exception as exc:
            chat.add_turn("system", f"**error:** `{exc!r}`")
        finally:
            self._busy = False
            self._refresh_status("ready")


# ---------------------------------------------------------------------------- #
# Helpers                                                                       #
# ---------------------------------------------------------------------------- #


def _split_chunk(chunk) -> tuple[str, str]:
    """Return (visible_text, thinking_text) extracted from a streaming chunk.

    Anthropic emits content as a list of typed blocks (`text`, `thinking`,
    `tool_use`, …); OpenAI o-series uses `reasoning` blocks; Gemini sends
    plain strings.
    """
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content, ""
    text_parts: list[str] = []
    think_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                text_parts.append(block.get("text", ""))
            elif t == "thinking":
                think_parts.append(block.get("thinking", ""))
            elif t == "reasoning":
                # OpenAI o-series: payload is in "text" or "summary".
                think_parts.append(block.get("text") or block.get("summary") or "")
    return "".join(text_parts), "".join(think_parts)


def _format_payload(value: object, max_chars: int = 400) -> str:
    """Compact, code-block-friendly preview of a tool input/output."""
    if value is None:
        return ""
    if isinstance(value, BaseMessage):
        value = value.content
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, ensure_ascii=False, default=repr)
        except Exception:
            text = repr(value)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…"
    return text
