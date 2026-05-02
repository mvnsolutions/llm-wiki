# llm-wiki

A terminal AI assistant in the spirit of Claude Code, built on:

- **[Textual](https://textual.textualize.io/)** for the TUI
- **[Deep Agents](https://github.com/langchain-ai/deepagents)** on top of LangChain 1.0 for the agent harness (planning, filesystem tools, subagents, skills)
- **Google Gemini**, **Anthropic Claude** or **OpenAI GPT** as the model

## Setup

```sh
uv venv
uv sync
cp .env.example .env   # then add providers API keys
```

## Run

```sh
uv run llm-wiki
```

The agent's filesystem root is pinned to the directory where you launched the TUI.

## Keys

- `Enter` — send message
- `Shift+Enter` — insert a newline (Kitty keyboard protocol terminals; on legacy terminals, `Ctrl+J` is the universal fallback)
- `Tab` — autocomplete slash command / value
- `Ctrl+L` — clear conversation
- `Ctrl+C` — quit

## Slash commands

Type these in the prompt; tab-complete is wired for every command and its value choices.

| Command | Purpose |
|---|---|
| `/model <anthropic\|gemini\|openai> [name]` | Switch the active model. With no name, picks the provider's default. |
| `/effort <off\|low\|medium\|high>` | Reasoning effort. Maps to Anthropic extended thinking, OpenAI `reasoning_effort`, or Gemini `thinking_budget` per provider. |
| `/debug [on\|off]` | Show tool calls + results in the chat (off by default). Reasoning blocks are always shown when the model emits them. |
| `/clear` | Clear the conversation. |
| `/help` | List commands. |
| `/quit` / `/exit` | Exit. |

`/model`, `/effort`, and `/clear` reset the conversation thread.

## Status bar

Bottom line, left to right:

```
user@host  |  cwd  |  provider:model  |  state  [|  debug]  |  effort:<level>
```

## Configuration

Environment is read from `.env`:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required to use the Anthropic provider |
| `GOOGLE_API_KEY` | Required to use the Gemini provider |
| `OPENAI_API_KEY` | Required to use the OpenAI provider |
| `LLM_WIKI_PROVIDER` | Startup default — `anthropic`, `google_genai`, or `openai` |
| `LLM_WIKI_MODEL` | Startup default model id; falls back to the provider's first entry |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_TRACING` | Optional LangSmith tracing |
