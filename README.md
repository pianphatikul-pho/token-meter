# tokenmeter

A small middleman that counts the tokens any AI model call uses.

Every model call goes through the meter. It makes the call, reads the token
counts out of the reply, and saves one row per call in a ledger file. From the
ledger you get totals per model, per round, or however you label your calls,
and a CSV for your own analysis. It can also hold a token budget and refuse
calls once the budget is spent.

Everything is in one file, [tokenmeter.py](tokenmeter.py), with nothing to
install beyond Python 3.9 or newer. It runs on Windows, macOS and Linux. The
file holds the meter, the eight model-and-agent configurations from Chapter 3
of the project, and the terminal commands, so it is the only file you need to
copy or run.

## What is in this repository

| File | What it is |
| --- | --- |
| [tokenmeter.py](tokenmeter.py) | **the whole program**: the meter, the planned GPT, Claude and Qwen models with their settings, and the terminal commands. Copy this one file into any project. |
| [test_tokenmeter.py](test_tokenmeter.py) | the automated tests |
| [examples/quickstart.py](examples/quickstart.py) | a tour of every feature, using a pretend model |
| [examples/local_test_server.py](examples/local_test_server.py) | a stand-in model server, for testing without a GPU or API key |
| [examples/ollama_qwen.py](examples/ollama_qwen.py) | a local Qwen model through Ollama |
| [examples/openai_compatible.py](examples/openai_compatible.py) | GPT, or any server that speaks the OpenAI chat format |
| [examples/anthropic_claude.py](examples/anthropic_claude.py) | Claude through the Anthropic Python SDK |
| [examples/claude_code_cli.py](examples/claude_code_cli.py) | Claude Code (and, the same way, Codex) |

## Set up, step by step

The commands below use `python`, as on Windows. **On macOS and Linux, type
`python3` wherever this guide says `python`.**

**1. Install Python 3.9 or newer.** On Windows, use the installer from
python.org and tick "Add python.exe to PATH" on its first screen. Then open a
new terminal and check it:

```bash
python --version
```

It should print `Python 3.9` or higher. If Windows opens the Microsoft Store
instead, Python is not on the PATH yet: re-run the installer with that box
ticked, or type `py` instead of `python`.

**2. Install Git**, from git-scm.com on Windows. macOS offers to install it the
first time you type `git`.

**3. Download the code.** If the repository is private, sign in to GitHub
when asked:

```bash
git clone https://github.com/pianphatikul-pho/token-meter.git
```

**4. Go into the folder:**

```bash
cd token-meter
```

**5. Check that everything works.** There is nothing to install: this runs the
automated tests, which need no internet, model or API key.

```bash
python -m unittest
```

The last line should be `OK`. If anything says `FAILED`, stop and fix that
first.

## Run it, step by step

### Step 1. See what it does, with no model at all

```bash
python examples/quickstart.py
```

A pretend model answers a few requests, and the meter prints two tables of
token totals: per model, and per task and round.

### Step 2. Test the whole setup with the stand-in server

This tries every part of the real workflow (the planned Qwen settings, the
token counts, the configuration records and the reports) against a stand-in
server that answers like Ollama. Its token counts are made up, so this checks
that everything is wired correctly, not how good a model is.

You need two terminals, both in the `token-meter` folder.

**In the first terminal**, start the stand-in server and leave it running:

```bash
python examples/local_test_server.py
```

**In the second terminal**, check each planned Qwen configuration. The
`--ledger test_ledger.db` keeps these made-up numbers away from your real
ledger.

```bash
python tokenmeter.py check qwen2.5-14b-instruct --host http://127.0.0.1:11435 --ledger test_ledger.db
```

```bash
python tokenmeter.py check qwen2.5-coder-14b-instruct --host http://127.0.0.1:11435 --ledger test_ledger.db
```

```bash
python tokenmeter.py check qwen3-30b-a3b --host http://127.0.0.1:11435 --ledger test_ledger.db
```

Each one should print `reply: ready`, `usage_source reported` and empty
`flags`. Then read the totals back from the ledger:

```bash
python tokenmeter.py report test_ledger.db --by model
```

When you are done, stop the server with Ctrl+C in the first terminal, and
delete `test_ledger.db`.

### Step 3. Connect the real models (on the experiment computer)

Do this once for each configuration before collecting any data. Each check
makes one small real call and prints what the meter recorded next to the
provider's own numbers. Without `--ledger`, it writes to `token_ledger.db`.

**Qwen, through Ollama.** Install Ollama from ollama.com, then download the
three models (about 9 GB, 9 GB and 19 GB):

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

```bash
ollama pull qwen2.5-coder:14b-instruct-q4_K_M
```

```bash
ollama pull qwen3:30b-a3b-q4_K_M
```

Then check each one, for example:

```bash
python tokenmeter.py check qwen3-30b-a3b
```

**Claude, through Claude Code.** Install Claude Code, sign in, and update it
to the newest version (older versions do not accept the Extra High effort
setting). Run the check from a normal terminal, not from inside a Claude Code
session: Claude Code will not start inside another Claude Code session.

```bash
python tokenmeter.py check claude-haiku-5-5
```

**GPT, through Codex.** Install the Codex command-line tool and sign in, then:

```bash
python tokenmeter.py check gpt-6-astra
```

The Codex path has never been run against the real tool. Read its check
output carefully, and confirm the two GPT model names.

Each check should end with `usage_source reported` and no flags. If it says
`estimated`, that connection is not returning token counts and its numbers
must not be used. To see every planned configuration:

```bash
python tokenmeter.py list
```

### Step 4. Use it in your experiment

**From the terminal.** `ask` sends a prompt to a planned model, prints the
reply, and records the tokens. The labels say which run, vulnerability, round
and step the call belongs to.

```bash
python tokenmeter.py ask qwen3-30b-a3b "Explain this vulnerability." --run sqli-01/loop/seed-42 --task SQLI-01 --round 1 --step patch --seed 42
```

For a long prompt, put it in a text file. `--out` saves the reply to a file:

```bash
python tokenmeter.py ask claude-opus-5-5 --prompt-file prompt.txt --out reply.txt --run sqli-01/loop/rep-1 --task SQLI-01 --round 1
```

After the reply, one line shows the tokens used, the budget left and any
flags. With `--run`, a local model is held to the run's 120,000-token budget:
once it is spent, `ask` refuses to send and exits with code 2. Run
`python tokenmeter.py ask --help` for every option, including `--system-file`,
`--workdir` and `--allow-edits` for the coding agents.

**From your own Python code:**

```python
from tokenmeter import TokenMeter, ask, start_run

meter = TokenMeter("token_ledger.db")
run = "sqli-01/loop/seed-42"
budget = start_run(meter, "qwen3-30b-a3b", run=run)        # saves the configuration record
reply = ask(meter, "qwen3-30b-a3b", prompt, seed=42, budgets=[budget] if budget else [],
            run=run, task="SQLI-01", round=1, step="patch")
print(reply.text)
```

Call `start_run` once at the start of each run. For a Qwen configuration it
returns the run's 120,000-token budget; for GPT and Claude it returns `None`,
because their usage is recorded but cannot be limited inside a single call.

### Step 5. Read the results

All of these work on any ledger file.

```bash
python tokenmeter.py report token_ledger.db --by model round
```

```bash
python tokenmeter.py export token_ledger.db token_calls.csv
```

```bash
python tokenmeter.py budgets token_ledger.db
```

```bash
python tokenmeter.py configs token_ledger.db
```

`report` prints token totals, grouped by any of `model`, `provider`,
`reported_model`, `experiment`, `run`, `task`, `round`, `step`, `usage_source`
and `stop_reason`. Add `--where model=qwen3:30b-a3b-q4_K_M round=2` to filter.
`export` writes one CSV row per call, `budgets` shows what each budget has used,
and `configs` prints the saved configuration records.

## Changing things

`tokenmeter.py` starts with a guide that says where each kind of change goes,
and the file is split into numbered sections, each with a note on what it does.

| To change | Go to |
| --- | --- |
| a planned model: add, remove, rename, or change its settings | section 1, `MODELS` |
| the limits, the token budget or the seeds | section 1, `LOCAL_LIMITS`, `LOCAL_RUN_TOKEN_BUDGET`, `LOCAL_SEEDS` |
| the default Ollama address | section 1, `OLLAMA_HOST` |
| support a provider whose replies look different | section 4: write one extractor function |
| support a new kind of client (not Ollama, Claude Code or Codex) | section 8, `ask()` and `describe()` |
| store an extra column for every call | section 2 and `TokenMeter.record()` in section 6 |
| add a terminal command | section 9 |

To add a model, copy an entry of the same client in `MODELS` and change its
short name and `"model"`. It then works in every command (`list`, `check`,
`ask`, `describe`). After any change, run `python -m unittest` again.

## The planned models

| Configuration | Reached through | Exact model | Settings applied |
| --- | --- | --- | --- |
| `gpt-6-astra`, `gpt-6-luna` | Codex CLI | as named in the plan (confirm the ids) | Extra High effort |
| `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` | Claude Code CLI | same ids | Extra High effort (`xhigh`) |
| `qwen2.5-14b-instruct` | Ollama | `qwen2.5:14b-instruct-q4_K_M` | temperature 0.2, top-p 0.9, top-k 40 |
| `qwen2.5-coder-14b-instruct` | Ollama | `qwen2.5-coder:14b-instruct-q4_K_M` | temperature 0.2, top-p 0.9, top-k 40 |
| `qwen3-30b-a3b` | Ollama | `qwen3:30b-a3b-q4_K_M` | thinking on, temperature 0.6, top-p 0.95, top-k 20 |

For the Qwen configurations it also applies the 16,384-token context window,
the 8,192-token output limit per request, the seed you pass (42, 43, 44) and
the 120,000-token budget per run. Each request records how much of the model
is on the GPU.

Their settings are in section 1 of [tokenmeter.py](tokenmeter.py).
`start_run` saves a configuration record: exact model id, model-file hash,
quantization, Ollama or client version, and every setting. For Codex and
Claude Code, the controls they do not expose (temperature, top-p, top-k, seed)
are written as "not configurable".

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

### Which reply formats it reads

| Provider | Extractor | Example |
| --- | --- | --- |
| OpenAI and OpenAI-compatible servers | `from_openai` | [openai_compatible.py](examples/openai_compatible.py) |
| Claude, Anthropic API | `from_anthropic` | [anthropic_claude.py](examples/anthropic_claude.py) |
| Ollama (Qwen and other local models) | `from_ollama` | [ollama_qwen.py](examples/ollama_qwen.py) |
| Google Gemini | `from_gemini` | |
| Claude Code CLI (`--output-format json`) | `from_claude_code` | [claude_code_cli.py](examples/claude_code_cli.py) |
| Codex CLI (`exec --json`) | `from_codex_jsonl` | |

These are recognized automatically. Replies can be plain dicts or SDK response
objects.

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

## Token budgets

A budget is a token allowance kept in the ledger. The meter checks it before
each call and refuses the call when nothing is left; the refusal is recorded
and `BudgetExceeded` is raised, so your loop can stop and save its state.

```python
from tokenmeter import output_allowance

total = meter.budget("run-7/total", 120_000)
blue = meter.budget("run-7/blue", 80_000)                  # a per-agent budget, if you want one

limit = output_allowance([blue, total], cap=8192)          # output tokens the next request may use
reply = meter.measure(model, lambda: call(prompt, max_tokens=limit), budgets=[blue, total], prompt=prompt)
```

- A call is charged to every budget you pass it. Calls made without a budget,
  such as final evaluation, are not charged, which gives them a separate
  allowance.
- A budget's limit cannot be changed after it is created.
- The check happens between calls. One call can still overshoot, by at most
  its own size, and is then flagged `over_budget`. For Qwen the output is
  capped per request, so the overshoot is bounded. For Codex and Claude Code
  one call is many model turns, so a budget there is a stop rule between
  calls, not an exact limit.

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
| `usage_source` | `reported`, `estimated`, or `none` (the call failed or was refused) |
| `reported_model` | the model the provider says actually answered |
| `stop_reason` | why generation ended, in the provider's words |
| `flags` | anything that needs a second look (below) |
| `prompt_chars`, `response_chars` | sizes in characters, independent of any tokenizer |
| `reasoning_chars` | size of the visible thinking text, when the model returns it |
| `cost_usd` | the provider's cost if it gives one, else from your price table, else empty |

A breakdown cell is **empty, not 0,** when the provider does not report it
(shown as `-` in reports). Claude does not report thinking tokens separately.
Ollama reports neither cache nor thinking tokens: for Qwen3 the thinking is
inside `output_tokens`, and `reasoning_chars` shows how much of the reply was
thinking.

| Flag | Meaning |
| --- | --- |
| `model_mismatch` | a different model answered than the one requested, for example after a downgrade |
| `multiple_models` | an agent tool used more than one model in the call |
| `output_truncated` | the reply was cut off at the output limit |
| `input_over_limit`, `output_at_limit`, `context_full` | the call hit a per-request limit you passed as `limits=` |
| `budget_refused` | the call was not made because a budget was spent |
| `over_budget` | this call pushed a budget past its limit |

The provider's original numbers are stored with each row (`raw_usage`), so a
count can always be checked later. The ledger is a normal SQLite file with one
table, `calls`, so it also opens in any SQLite viewer, pandas or R.

For cost, give the meter your prices in dollars per million tokens. This is a
simple estimate: it prices cached input the same as normal input.

```python
meter = TokenMeter("token_ledger.db", prices={"my-model-name": (2.00, 10.00)})   # (input, output)
```

## Before you trust the numbers

- **Check each connection once** (Step 3 above). The automated tests use
  sample replies and the stand-in server, not live services. The Codex path in
  particular has never been run against the real tool.
- **The 8,192-token input limit is checked after the call, not before.** The
  exact token count of a prompt is only known once the model has read it, so
  an over-long prompt is flagged `input_over_limit` rather than blocked.
- **When a model is downgraded after hitting a usage limit,** send the
  request with the downgraded configuration and add the label
  `downgraded_from="..."`. If the client downgrades by itself, the row is
  flagged `model_mismatch` where the client reports which model answered.
- **Tokens differ between model families.** GPT, Claude and Qwen each split
  text into tokens their own way, so the same text is a different number of
  tokens for each. Compare models on tokens with that stated, and use the
  character columns when you need a tokenizer-independent size.
- **Agent tools carry overhead.** Claude Code and Codex run several model
  turns per call and send their own built-in instructions, all of which count.
  A plain API or Ollama call is a single turn.
- **Keep API keys out of files.** The examples read them from environment
  variables.

## Troubleshooting

| You see | What to do |
| --- | --- |
| `python` is not recognized, or the Microsoft Store opens | Python is not on the PATH. Re-run the installer with "Add python.exe to PATH" ticked, or use `py` instead of `python`. |
| `cannot reach http://... Is the model server running` | Start Ollama (or the stand-in server in Step 2), and check the `--host` address. |
| `model '...' not found` | Download it first with `ollama pull` (Step 3). |
| `` `claude` is not installed or not on PATH `` (or `codex`) | Install that tool, open a new terminal, and check that typing its name works. |
| `Claude Code cannot be launched inside another Claude Code session` | Run the check from a normal terminal instead. |
| `usage_source estimated` | That connection returned no token counts. Do not use its numbers. |
| `budget exhausted` | The run used its whole token budget. That is the intended stop; start the next run under a new run name. |
