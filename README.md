# tokenmeter

A small middleman that counts the tokens any AI model call uses.

It is one file, [tokenmeter.py](tokenmeter.py), with no dependencies beyond
Python 3.9+. Copy that file into any project, on any computer, and send your
model calls through it. Each call becomes one row in a ledger file, and the
ledger gives you totals per model, per round, or however you label your calls.

## Try it

No internet, API key or install needed:

```bash
python3 examples/quickstart.py
```

```bash
python3 -m unittest
```

## Use it with your own model

```python
from tokenmeter import TokenMeter

meter = TokenMeter("token_ledger.db")          # the ledger file, created if missing

reply = meter.measure(
    "my-model-name",                           # the name to count this under
    lambda: call_my_model(prompt),             # your normal model call, unchanged
    prompt=prompt,                             # what you sent
    task="SQLI-01", round=1, step="patch",     # labels, all optional
)

meter.print_report(by=["model", "round"])
```

`measure` runs your call, returns exactly what your call returned, and records
the tokens. If the call raises an error, the failure is recorded and the error
still reaches your code.

### Labels

Attach any of these to a call and group reports by them:

| Label | Meant for |
| --- | --- |
| `experiment` | which experiment or arm (`"single-pass"`, `"loop"`) |
| `run` | which repetition (`"rep-1"`) |
| `task` | what was being worked on (a vulnerability id) |
| `round` | attempt number (a whole number) |
| `step` | what the call was for (`"analyse"`, `"patch"`) |

Any other name is kept as a tag on the row. To label a whole block at once:

```python
with meter.context(run="rep-2", task="AUTH-01", round=3):
    meter.measure(...)
    meter.measure(...)
```

### Other ways to record

```python
ask = meter.wrap(call_my_model, "my-model-name")   # a metered copy of your function
ask("hello")

from tokenmeter import Usage                       # you already have the numbers
meter.record("my-model-name", Usage(input_tokens=900, output_tokens=150), round=1)
```

Use `record` for streamed replies: let the stream finish, then record the
final usage it reported. (OpenAI-style streams only include usage when you ask
for it with `stream_options={"include_usage": True}`.)

## Which models work

The meter reads the token counts out of the reply. These formats are
recognized automatically:

| Provider | Extractor | Example |
| --- | --- | --- |
| OpenAI and OpenAI-compatible servers | `from_openai` | [openai_compatible.py](examples/openai_compatible.py) |
| Claude, Anthropic API | `from_anthropic` | [anthropic_claude.py](examples/anthropic_claude.py) |
| Ollama (Qwen and other local models) | `from_ollama` | [ollama_qwen.py](examples/ollama_qwen.py) |
| Google Gemini | `from_gemini` | |
| Claude Code CLI (`--output-format json`) | `from_claude_code` | [claude_code_cli.py](examples/claude_code_cli.py) |
| Codex CLI (`exec --json`) | `from_codex_jsonl` | |

Replies can be plain dicts or SDK response objects.

**Any other model:** write one small function that turns its reply into a
`Usage`, and pass it as `extract=`. [quickstart.py](examples/quickstart.py)
shows this.

```python
def read_usage(reply):
    return Usage(input_tokens=reply["stats"]["in"], output_tokens=reply["stats"]["out"])

meter.measure("new-model", lambda: call_it(prompt), extract=read_usage, prompt=prompt)
```

**A model that reports no token counts at all:** the meter estimates from the
text length (characters ÷ 4) and marks the row `estimated`, so an estimate can
never pass as a real measurement.

## What the numbers mean

Every provider is converted to the same meaning:

| Column | Meaning |
| --- | --- |
| `input_tokens` | everything sent to the model, cached tokens included |
| `cache_read_tokens` | the part of the input served from the provider's cache |
| `cache_write_tokens` | the part of the input written to the cache |
| `output_tokens` | everything generated, reasoning included |
| `reasoning_tokens` | the part of the output that was reasoning, when reported |
| `total_tokens` | input + output |
| `usage_source` | `reported`, `estimated`, or `none` (the call failed) |
| `prompt_chars`, `response_chars` | sizes in characters, independent of any tokenizer |
| `cost_usd` | the provider's cost if it gives one, else from your price table, else empty |

The provider's original numbers are stored with each row (`raw_usage`), so a
count can always be checked later.

For cost, give the meter your prices in dollars per million tokens:

```python
meter = TokenMeter("token_ledger.db", prices={"my-model-name": (2.00, 10.00)})   # (input, output)
```

This is a simple estimate: it prices cached input the same as normal input.

## Reports and export

In code:

```python
meter.print_report(by=["model", "task", "round"])
meter.print_report(by=["model"], experiment="loop")        # filter
rows = meter.summary(by=["model", "round"])                # the same data as a list of dicts
meter.export_csv("token_calls.csv")                        # one row per call
```

From the terminal, on any ledger file:

```bash
python3 tokenmeter.py report token_ledger.db --by model round
```

```bash
python3 tokenmeter.py export token_ledger.db token_calls.csv
```

The ledger is a normal SQLite file with one table, `calls`, so it also opens
in any SQLite viewer, pandas or R.

## Before you trust the numbers

- **Check each connection once.** The extractors are tested against sample
  replies in [test_tokenmeter.py](test_tokenmeter.py), not against live
  services. Make one real call per model, then compare the row with what the
  provider's own dashboard or raw reply says. The Codex one in particular has
  never been run against the real tool.
- **Tokens differ between model families.** GPT, Claude and Qwen each split
  text into tokens their own way, so the same text is a different number of
  tokens for each. Compare models on tokens with that stated, and use the
  character columns when you need a tokenizer-independent size.
- **Agent tools carry overhead.** Claude Code and Codex run several model
  turns per call and send their own built-in instructions, all of which count.
  A plain API or Ollama call is a single turn.
- **Keep API keys out of files.** The examples read them from environment
  variables.
