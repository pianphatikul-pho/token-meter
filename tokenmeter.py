"""tokenmeter: count, limit and report the tokens that AI model calls use.

Everything is in this one file. It needs only Python 3.9 or newer (nothing to
install) and runs on Windows, macOS and Linux.

WHAT IT DOES
    Every model call goes through the meter. The meter makes the call, reads
    the token counts out of the model's reply, and saves one row per call in a
    small database file (the "ledger", token_ledger.db by default). From the
    ledger you get totals per model, per round or per step, and a CSV for
    your own analysis. For local models it also enforces a token budget.

RUN IT FROM THE TERMINAL  (on macOS / Linux type python3 instead of python)
    python tokenmeter.py list                        the planned models
    python tokenmeter.py check qwen3-30b-a3b         one small real call, to test a connection
    python tokenmeter.py ask qwen3-30b-a3b "..."     send a prompt and count its tokens
    python tokenmeter.py report token_ledger.db      token totals
    python tokenmeter.py export token_ledger.db out.csv
    python tokenmeter.py --help                      every command and option

USE IT FROM YOUR OWN PYTHON CODE
    from tokenmeter import TokenMeter, ask, start_run

    meter = TokenMeter("token_ledger.db")
    budget = start_run(meter, "qwen3-30b-a3b", run="sqli-01/loop/seed-42")
    reply = ask(meter, "qwen3-30b-a3b", prompt, seed=42, budgets=[budget],
                run="sqli-01/loop/seed-42", task="SQLI-01", round=1, step="patch")
    print(reply.text)

    # any other model: wrap your own call
    reply = meter.measure("my-model", lambda: call_my_model(prompt), prompt=prompt)

WHERE TO CHANGE THINGS
    Add, remove or change a planned model ...... section 1, MODELS
    Change limits, the budget or the seeds ..... section 1, LOCAL_LIMITS,
                                                 LOCAL_RUN_TOKEN_BUDGET, LOCAL_SEEDS
    Change the default Ollama address .......... section 1, OLLAMA_HOST
    Support a provider whose replies look
      different (read its token counts) ........ section 4: write one "extractor"
                                                 function, like the ones there
    Support a new kind of client (not Ollama,
      Claude Code or Codex) .................... section 8, ask() and describe()
    Store an extra column for every call ....... section 2 (table layout) and
                                                 TokenMeter.record() in section 6
    Add a terminal command ..................... section 9, main()
    The parts you will normally never need to touch are sections 2 to 7.
    After any change, run the tests:  python -m unittest

SECTIONS
    1. Settings: planned models and limits      (edit here)
    2. Ledger layout                            (the database tables)
    3. The usage record                         (one call's token counts)
    4. Reading token counts from replies        (one "extractor" per provider)
    5. Token budgets
    6. The meter                                (records calls, makes reports)
    7. Talking to models                        (Ollama, OpenAI-style servers, command-line tools)
    8. The planned models                       (ask, describe, start_run)
    9. Command line                             (the terminal commands)

WHAT THE NUMBERS MEAN  (the same for every provider)
    input_tokens       everything sent to the model, cached tokens included
      cache_read_tokens    the part of the input served from the provider's cache
      cache_write_tokens   the part of the input written to the cache
    output_tokens      everything generated, reasoning / thinking included
      reasoning_tokens     the part of the output that was reasoning
    total_tokens       input_tokens + output_tokens
    A breakdown is EMPTY (not 0) when the provider does not report it, so
    "unavailable" is never mistaken for "zero".

    usage_source: where the numbers came from
        reported    the provider returned them
        estimated   the provider returned none, so characters / 4 was used
        none        the call failed or was refused before any usage came back

    flags: anything that needs a second look
        model_mismatch     a different model answered than the one requested
        multiple_models    an agent tool used more than one model in the call
        output_truncated   the reply was cut off at the output limit
        input_over_limit   more input tokens than the per-request limit
        output_at_limit    output reached the per-request limit
        context_full       input + output filled the context window
        budget_refused     the call was not made: a budget was already spent
        over_budget        this call pushed a budget past its limit
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence

__version__ = "2.0.0"


# =============================================================================
# 1. SETTINGS: PLANNED MODELS AND LIMITS                          (edit here)
#
# These are the Chapter 3 settings. Change them here and everything else
# follows: requests, budgets, configuration records and the `list` command.
# =============================================================================

# Where Ollama listens. Ollama's own default; change it if yours is elsewhere.
# Every command that talks to Ollama also accepts --host to override it.
OLLAMA_HOST = "http://localhost:11434"

# Per-request limits for local (Ollama) models: the context window, and the
# most input and output tokens one request may use. The output limit is sent
# to the model; the input limit is checked after the call and flagged.
LOCAL_LIMITS = {"context_window": 16_384, "max_input_tokens": 8_192, "max_output_tokens": 8_192}

# Token budget for one run of a local model: input + output tokens across all
# repair and prompt-refinement requests. Requests are refused once it is spent.
LOCAL_RUN_TOKEN_BUDGET = 120_000

# One seed per repetition (local models only; cloud clients have no seed).
LOCAL_SEEDS = (42, 43, 44)

# The quantization every local model file is expected to have. The
# configuration record says whether the installed file matches.
EXPECTED_QUANTIZATION = "Q4_K_M"

# Sampling settings shared by more than one model.
QWEN_14B_SAMPLING = {"temperature": 0.2, "top_p": 0.9, "top_k": 40}
QWEN_30B_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# The planned model-and-agent configurations.
#
# Each entry is   "short-name": {settings}   and the short name is what you
# type in commands (python tokenmeter.py check short-name).
#
#   client     how the model is reached:
#                "ollama"       a local model through Ollama
#                "claude-code"  Claude through the Claude Code command-line tool
#                "codex"        GPT through the Codex command-line tool
#   model      the exact model name that client expects
#   effort     (claude-code, codex) reasoning effort; "Extra High" is "xhigh"
#   think      (ollama) True for thinking models such as Qwen3; None for models
#              that cannot think (Ollama rejects the setting for them)
#   sampling   (ollama) temperature, top_p and top_k
#
# To ADD a model, copy an entry of the same client and change the short name
# and "model". For a new Ollama model, also run `ollama pull <model>` first.
MODELS: dict = {
    # GPT through Codex. TODO: confirm these two model names with Codex itself;
    # they are written as the plan names them.
    "gpt-6-astra": {"client": "codex", "model": "gpt-6-astra", "effort": "xhigh"},
    "gpt-6-luna": {"client": "codex", "model": "gpt-6-luna", "effort": "xhigh"},

    # Claude through Claude Code.
    "claude-opus-5-5": {"client": "claude-code", "model": "claude-opus-5-5", "effort": "xhigh"},
    "claude-sonnet-5-5": {"client": "claude-code", "model": "claude-sonnet-5-5", "effort": "xhigh"},
    "claude-haiku-5-5": {"client": "claude-code", "model": "claude-haiku-5-5", "effort": "xhigh"},

    # Qwen through Ollama, Q4_K_M files. Tags checked on ollama.com on 2026-11-17.
    "qwen2.5-14b-instruct": {
        "client": "ollama", "model": "qwen2.5:14b-instruct-q4_K_M", "think": None, "sampling": QWEN_14B_SAMPLING,
    },
    "qwen2.5-coder-14b-instruct": {
        "client": "ollama", "model": "qwen2.5-coder:14b-instruct-q4_K_M", "think": None,
        "sampling": QWEN_14B_SAMPLING,
    },
    # Plain `qwen3:30b` is a DIFFERENT file (the later "thinking-2507"
    # release). This tag is the original Qwen3-30B-A3B at Q4_K_M.
    "qwen3-30b-a3b": {
        "client": "ollama", "model": "qwen3:30b-a3b-q4_K_M", "think": True, "sampling": QWEN_30B_SAMPLING,
    },
}

# The ledger file used when a command is not given one.
DEFAULT_LEDGER = "token_ledger.db"

# When a provider sends no token counts at all, tokens are estimated as
# characters divided by this, and the row is marked "estimated".
CHARS_PER_TOKEN_ESTIMATE = 4


# =============================================================================
# 2. LEDGER LAYOUT
#
# The ledger is a normal SQLite file. It opens in any SQLite viewer, pandas
# or R. Tables:
#   calls           one row per model call (the main table)
#   budgets         each token budget and its limit
#   budget_charges  which call was charged to which budget
#   configs         configuration records saved at the start of each run
#
# To store an extra column for every call: add it to _CALLS_TABLE and to
# _CALL_COLUMNS, then fill it in TokenMeter.record() (section 6). Ledgers
# made by older versions are upgraded automatically when opened.
# =============================================================================

UNAVAILABLE = "unavailable"

# The labels you can attach to a call (and group reports by).
CONTEXT_COLUMNS = ("experiment", "run", "task", "round", "step")
GROUP_COLUMNS = ("model", "provider", "reported_model") + CONTEXT_COLUMNS + ("usage_source", "stop_reason")
# Totals that always exist, and breakdowns a provider may not report.
ALWAYS_KNOWN = ("input_tokens", "output_tokens", "total_tokens", "seconds", "prompt_chars", "response_chars")
MAY_BE_UNAVAILABLE = ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "reasoning_chars", "cost_usd")
SUM_COLUMNS = (
    "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens", "cost_usd", "seconds", "prompt_chars", "response_chars", "reasoning_chars",
)
# The columns of the CSV export, in order.
CSV_COLUMNS = (
    ("call_id", "time_utc", "model", "reported_model", "provider") + CONTEXT_COLUMNS + SUM_COLUMNS
    + ("usage_source", "stop_reason", "flags", "error", "tags")
)
# Stop reasons that mean "the reply was cut off" (flagged output_truncated).
TRUNCATED_STOP_REASONS = {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded"}

_CALLS_TABLE = """
CREATE TABLE calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_utc TEXT NOT NULL,
    model TEXT NOT NULL,
    reported_model TEXT,
    provider TEXT NOT NULL DEFAULT '',
    experiment TEXT, run TEXT, task TEXT, round INTEGER, step TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    seconds REAL NOT NULL DEFAULT 0,
    prompt_chars INTEGER NOT NULL DEFAULT 0,
    response_chars INTEGER NOT NULL DEFAULT 0,
    reasoning_chars INTEGER,
    usage_source TEXT NOT NULL DEFAULT 'reported',
    stop_reason TEXT,
    flags TEXT NOT NULL DEFAULT '',
    error TEXT,
    tags TEXT NOT NULL DEFAULT '{}',
    raw_usage TEXT NOT NULL DEFAULT '{}'
)
"""
_CALL_COLUMNS = (
    "call_id", "time_utc", "model", "reported_model", "provider", "experiment", "run", "task", "round", "step",
    "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens", "cost_usd", "seconds", "prompt_chars", "response_chars", "reasoning_chars",
    "usage_source", "stop_reason", "flags", "error", "tags", "raw_usage",
)
_OTHER_TABLES = """
CREATE TABLE IF NOT EXISTS budgets (
    name TEXT PRIMARY KEY, limit_tokens INTEGER NOT NULL, created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS budget_charges (
    budget TEXT NOT NULL, call_id INTEGER NOT NULL, tokens INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS configs (
    config_id INTEGER PRIMARY KEY AUTOINCREMENT, time_utc TEXT NOT NULL, name TEXT NOT NULL, config TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# 3. THE USAGE RECORD
#
# A Usage holds one call's token counts after they have been converted to the
# common meaning described at the top of this file. Extractors (section 4)
# produce it; the meter (section 6) stores it.
# =============================================================================

@dataclass
class Usage:
    """Token counts for one call. Leave a breakdown as None when the provider
    does not report it."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None  # only when the provider reports a cost
    source: str = "reported"
    model: Optional[str] = None  # the model the provider says actually answered
    stop_reason: Optional[str] = None  # why generation ended, in the provider's words
    reasoning_chars: Optional[int] = None  # size of visible thinking text, if any
    raw: dict = field(default_factory=dict)  # the provider's own numbers, kept for auditing
    tags: dict = field(default_factory=dict)  # extra facts to store with the call

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _as_dict(obj: Any) -> dict:
    """Accept a plain dict or an SDK response object."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for name in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, dict):
                return value
    return dict(getattr(obj, "__dict__", None) or {})


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _maybe(container: dict, key: str) -> Optional[int]:
    """The count if the provider sent it, otherwise None (unavailable)."""
    return _int(container[key]) if container.get(key) is not None else None


def _text(value: Any) -> Optional[str]:
    return str(value) if value not in (None, "") else None


# =============================================================================
# 4. READING TOKEN COUNTS FROM REPLIES
#
# Every provider reports tokens in its own format. An "extractor" turns one
# provider's reply into a Usage, or returns None if the reply has no counts.
#
# To support a provider not listed here, write one more function like these
# and pass it to measure() as extract=your_function. detect() picks the right
# built-in extractor automatically.
# =============================================================================

def from_openai(response: Any) -> Optional[Usage]:
    """OpenAI and OpenAI-compatible servers (Chat Completions or Responses API).

    Their input count already includes cached tokens and their output count
    already includes reasoning tokens.
    """
    data = _as_dict(response)
    usage = _as_dict(data.get("usage"))
    if not usage:
        return None
    if "prompt_tokens" in usage or "completion_tokens" in usage:  # Chat Completions
        details_in = _as_dict(usage.get("prompt_tokens_details"))
        details_out = _as_dict(usage.get("completion_tokens_details"))
        choices = data.get("choices") or [{}]
        stop = _as_dict(choices[0]).get("finish_reason")
        return Usage(
            input_tokens=_int(usage.get("prompt_tokens")),
            output_tokens=_int(usage.get("completion_tokens")),
            cache_read_tokens=_maybe(details_in, "cached_tokens"),
            reasoning_tokens=_maybe(details_out, "reasoning_tokens"),
            model=_text(data.get("model")),
            stop_reason=_text(stop),
            raw=usage,
        )
    details_in = _as_dict(usage.get("input_tokens_details"))  # Responses API
    details_out = _as_dict(usage.get("output_tokens_details"))
    incomplete = _as_dict(data.get("incomplete_details")).get("reason")
    return Usage(
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_tokens=_maybe(details_in, "cached_tokens"),
        reasoning_tokens=_maybe(details_out, "reasoning_tokens"),
        model=_text(data.get("model")),
        stop_reason=_text(incomplete or data.get("status")),
        raw=usage,
    )


def from_anthropic(response: Any) -> Optional[Usage]:
    """Claude through the Anthropic Messages API.

    Anthropic reports uncached input separately from cache reads and cache
    writes, so the three are added to get the full input. Thinking tokens are
    inside output_tokens and are not reported separately.
    """
    data = _as_dict(response)
    usage = _as_dict(data.get("usage"))
    if not usage:
        return None
    cache_read, cache_write = _maybe(usage, "cache_read_input_tokens"), _maybe(usage, "cache_creation_input_tokens")
    return Usage(
        input_tokens=_int(usage.get("input_tokens")) + (cache_read or 0) + (cache_write or 0),
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        model=_text(data.get("model")),
        stop_reason=_text(data.get("stop_reason")),
        raw=usage,
    )


def from_ollama(response: Any) -> Optional[Usage]:
    """Local models through Ollama's own API (/api/chat or /api/generate).

    For a thinking model (Qwen3), eval_count covers thinking and answer
    together. Ollama gives no separate thinking-token count, so
    reasoning_tokens stays empty and the size of the thinking text is kept in
    reasoning_chars instead.
    """
    data = _as_dict(response)
    if "prompt_eval_count" not in data and "eval_count" not in data:
        return None
    message = _as_dict(data.get("message"))
    thinking = message.get("thinking") if message else data.get("thinking")
    return Usage(
        input_tokens=_int(data.get("prompt_eval_count")),
        output_tokens=_int(data.get("eval_count")),
        cost_usd=0.0,  # local: no API bill
        model=_text(data.get("model")),
        stop_reason=_text(data.get("done_reason")),
        reasoning_chars=len(thinking) if isinstance(thinking, str) else None,
        raw={k: v for k, v in data.items() if k.endswith(("_count", "_duration")) or k == "done_reason"},
    )


def from_gemini(response: Any) -> Optional[Usage]:
    """Google Gemini (`usageMetadata`). Thinking tokens are counted as output."""
    data = _as_dict(response)
    usage = _as_dict(data.get("usageMetadata") or data.get("usage_metadata"))
    if not usage:
        return None

    def pick(camel: str, snake: str) -> Optional[int]:
        value = usage.get(camel, usage.get(snake))
        return _int(value) if value is not None else None

    prompt = pick("promptTokenCount", "prompt_token_count") or 0
    candidates = pick("candidatesTokenCount", "candidates_token_count") or 0
    thoughts = pick("thoughtsTokenCount", "thoughts_token_count")
    total = pick("totalTokenCount", "total_token_count")
    # Use the provider's own total to decide whether thoughts are already
    # inside the candidates count.
    output = candidates if total and prompt + candidates == total else candidates + (thoughts or 0)
    first = _as_dict((data.get("candidates") or [{}])[0])
    return Usage(
        input_tokens=prompt,
        output_tokens=output,
        cache_read_tokens=pick("cachedContentTokenCount", "cached_content_token_count"),
        reasoning_tokens=thoughts,
        model=_text(data.get("modelVersion") or data.get("model_version")),
        stop_reason=_text(first.get("finishReason") or first.get("finish_reason")),
        raw=usage,
    )


def from_claude_code(response: Any) -> Optional[Usage]:
    """The JSON printed by `claude -p --output-format json` (dict or text).

    One call here can be several model turns; the counts are for all of them.
    `reported_model` lists every model the tool says it used, so a downgrade
    or a helper model shows up.
    """
    data = json.loads(response) if isinstance(response, str) else _as_dict(response)
    usage = from_anthropic(data)
    if usage is None:
        return None
    if data.get("total_cost_usd") is not None:
        usage.cost_usd = float(data["total_cost_usd"])
    per_model = _as_dict(data.get("modelUsage"))
    usage.model = ", ".join(sorted(per_model)) if per_model else _text(data.get("model"))
    usage.stop_reason = _text(data.get("subtype"))
    if data.get("num_turns") is not None:
        usage.tags["turns"] = data["num_turns"]
    usage.raw = {k: v for k, v in data.items() if k != "result"}
    return usage


def from_codex_jsonl(output: Any) -> Optional[Usage]:
    """The event lines printed by `codex exec --json`.

    NOT VERIFIED against a live Codex CLI: check one real output before
    relying on it (python tokenmeter.py check gpt-6-astra).
    """
    if not isinstance(output, str):
        return None
    turns, model = [], None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        model = model or _text(event.get("model"))
        if event.get("type") == "turn.completed" and event.get("usage"):
            turns.append(event["usage"])
    if not turns:
        return None

    def total(key: str) -> Optional[int]:
        return sum(_int(t.get(key)) for t in turns) if any(t.get(key) is not None for t in turns) else None

    return Usage(
        input_tokens=total("input_tokens") or 0,
        output_tokens=total("output_tokens") or 0,
        cache_read_tokens=total("cached_input_tokens"),
        reasoning_tokens=total("reasoning_output_tokens"),
        model=model,
        raw={"turns": turns},
        tags={"turns": len(turns)},
    )


def codex_final_message(output: str) -> str:
    """The last message the Codex agent wrote, from `codex exec --json` output."""
    text = ""
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if isinstance(item, dict) and item.get("type") == "agent_message" and item.get("text"):
            text = item["text"]
    return text


def detect(response: Any) -> Optional[Usage]:
    """Work out which provider a reply came from and read its usage.

    This is the default extractor. Pass a specific one if you know which you
    need, or your own function for anything else.
    """
    if isinstance(response, Usage):
        return response
    if isinstance(response, str):
        text = response.strip()
        if text.startswith("{") and '"usage"' in text:
            try:
                return detect(json.loads(text))
            except json.JSONDecodeError:
                pass
        return from_codex_jsonl(response)
    data = _as_dict(response)
    if not data:
        return None
    if "prompt_eval_count" in data or "eval_count" in data:
        return from_ollama(data)
    if "usageMetadata" in data or "usage_metadata" in data:
        return from_gemini(data)
    usage = _as_dict(data.get("usage"))
    if not usage:
        return None
    if "total_cost_usd" in data and "result" in data:
        return from_claude_code(data)
    if data.get("type") == "message" or "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        return from_anthropic(data)
    return from_openai(data)


def guess_text(response: Any) -> str:
    """Best effort: the text the model wrote, for the response size in characters."""
    if isinstance(response, str):
        return codex_final_message(response) or response
    data = _as_dict(response)
    if not data:
        return ""
    for key in ("output_text", "result", "response"):
        if isinstance(data.get(key), str):
            return data[key]
    try:
        if data.get("choices"):  # OpenAI chat
            return _as_dict(_as_dict(data["choices"][0]).get("message")).get("content") or ""
        if isinstance(data.get("content"), list):  # Anthropic
            return "".join(_as_dict(b).get("text") or "" for b in data["content"])
        if data.get("message"):  # Ollama chat
            return _as_dict(data["message"]).get("content") or ""
        if data.get("candidates"):  # Gemini
            parts = _as_dict(_as_dict(data["candidates"][0]).get("content")).get("parts") or []
            return "".join(_as_dict(p).get("text") or "" for p in parts)
    except (IndexError, KeyError, TypeError):
        pass
    return ""


def _chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(value))


def same_model(requested: str, reported: str) -> bool:
    """True if any reported model is the requested one (allowing date suffixes)."""
    wanted = requested.strip().lower()
    for name in reported.lower().split(","):
        name = name.strip()
        if name and (wanted in name or name in wanted):
            return True
    return False


# =============================================================================
# 5. TOKEN BUDGETS
#
# A budget is a token allowance stored in the ledger. measure() checks it
# BEFORE each call and refuses the call when nothing is left. It cannot stop a
# call that is already running, so one call can overshoot by at most its own
# size (cap the output with output_allowance() to keep that small). For agent
# tools that make many model turns inside one call, a budget is a stop rule
# between calls, not an exact limit.
# =============================================================================

class BudgetExceeded(RuntimeError):
    """Raised instead of making a call when a budget has nothing left."""

    def __init__(self, budget: "Budget"):
        self.budget = budget
        super().__init__(f"budget exhausted: {budget.name} (used {budget.used():,} of {budget.limit:,} tokens)")


class Budget:
    """One token budget. Get one from meter.budget(name, limit)."""

    def __init__(self, meter: "TokenMeter", name: str, limit: int):
        self.meter, self.name, self.limit = meter, name, int(limit)

    def used(self) -> int:
        return self.meter._scalar("SELECT COALESCE(SUM(tokens), 0) FROM budget_charges WHERE budget = ?", [self.name])

    def remaining(self) -> int:
        return max(0, self.limit - self.used())

    @property
    def exhausted(self) -> bool:
        return self.remaining() <= 0

    def max_output(self, cap: Optional[int] = None) -> int:
        """How many output tokens the next request may be allowed."""
        left = self.remaining()
        return left if cap is None else min(int(cap), left)

    def __repr__(self) -> str:
        return f"Budget({self.name!r}, used {self.used():,} of {self.limit:,})"


def output_allowance(budgets: Sequence[Budget], cap: int) -> int:
    """The output-token limit to send with the next request: your per-request
    cap, or less if any budget has less than that left."""
    return min([int(cap)] + [b.remaining() for b in budgets])


# =============================================================================
# 6. THE METER
#
# TokenMeter is the middleman. measure() makes a call and records it;
# record() stores numbers you already have; the reading methods (rows,
# summary, report, export_csv) turn the ledger into results.
# =============================================================================

class TokenMeter:
    """Create one per ledger file and send every call through it.

    path     the SQLite ledger file (created if missing). ":memory:" keeps
             nothing on disk.
    prices   optional {model: (input_usd_per_million, output_usd_per_million)},
             used for cost when the provider does not report one. This prices
             cached input the same as normal input.
    context  default labels for every call: experiment, run, task, round, step.
             Any other name is stored as a tag.
    """

    def __init__(self, path: str = DEFAULT_LEDGER, prices: Optional[dict] = None, **context: Any):
        self.path = str(path)
        self.prices = dict(prices or {})
        self._context = dict(context)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._prepare()

    def _prepare(self) -> None:
        """Create the tables, upgrading a ledger made by an older version."""
        existing = {row["name"]: row for row in self._db.execute("PRAGMA table_info(calls)")}
        if not existing:
            self._db.execute(_CALLS_TABLE)
        elif any(c not in existing for c in _CALL_COLUMNS) or any(
            existing[c]["notnull"] for c in MAY_BE_UNAVAILABLE if c in existing
        ):
            shared = ", ".join(c for c in _CALL_COLUMNS if c in existing)
            self._db.execute("ALTER TABLE calls RENAME TO calls_old")
            self._db.execute(_CALLS_TABLE)
            self._db.execute(f"INSERT INTO calls ({shared}) SELECT {shared} FROM calls_old")
            self._db.execute("DROP TABLE calls_old")
        self._db.executescript(_OTHER_TABLES)
        self._db.commit()

    def _scalar(self, sql: str, values: Sequence[Any] = ()) -> Any:
        with self._lock:
            return self._db.execute(sql, list(values)).fetchone()[0]

    # ---- labelling

    @contextmanager
    def context(self, **labels: Any) -> Iterator["TokenMeter"]:
        """Label every call made inside the `with` block.

            with meter.context(task="SQLI-01", round=2):
                meter.measure(...)
        """
        previous = self._context
        self._context = {**previous, **labels}
        try:
            yield self
        finally:
            self._context = previous

    # ---- budgets

    def budget(self, name: str, limit: Optional[int] = None) -> Budget:
        """Create a token budget, or reopen one that is already in the ledger.

            total = meter.budget("run-7/total", 120_000)

        A budget's limit is fixed once created: reopening it with a different
        limit is an error, so a run cannot quietly get more tokens.
        """
        with self._lock:
            row = self._db.execute("SELECT limit_tokens FROM budgets WHERE name = ?", [name]).fetchone()
            if row is None:
                if limit is None:
                    raise ValueError(f"no budget named {name!r} in this ledger; give a limit to create it")
                self._db.execute("INSERT INTO budgets VALUES (?, ?, ?)", [name, int(limit), _now()])
                self._db.commit()
                return Budget(self, name, int(limit))
            if limit is not None and int(limit) != row["limit_tokens"]:
                raise ValueError(f"budget {name!r} already exists with limit {row['limit_tokens']:,}")
            return Budget(self, name, row["limit_tokens"])

    def budgets(self) -> list[dict]:
        """Every budget in the ledger with what it has used."""
        with self._lock:
            rows = self._db.execute(
                "SELECT b.name, b.limit_tokens, COALESCE(SUM(c.tokens), 0) AS used_tokens, COUNT(c.call_id) AS calls "
                "FROM budgets b LEFT JOIN budget_charges c ON c.budget = b.name GROUP BY b.name ORDER BY b.name"
            ).fetchall()
        return [{**dict(r), "remaining_tokens": max(0, r["limit_tokens"] - r["used_tokens"])} for r in rows]

    # ---- recording

    def record(
        self,
        model: str,
        usage: Optional[Usage] = None,
        *,
        prompt: Any = "",
        response: Any = "",
        seconds: float = 0.0,
        error: Optional[str] = None,
        provider: str = "",
        budgets: Sequence[Budget] = (),
        limits: Optional[dict] = None,
        flags: Sequence[str] = (),
        **labels: Any,
    ) -> dict:
        """Write one call to the ledger and return the stored row.

        Use this directly when you already have the numbers (for example after
        a streamed reply finishes). Otherwise use measure().

        budgets  budgets to charge this call's total tokens to
        limits   optional per-request limits to check the call against:
                 {"max_input_tokens": .., "max_output_tokens": .., "context_window": ..}
        """
        prompt_chars, response_chars = _chars(prompt), _chars(response)
        if usage is None:
            if error is not None:
                usage = Usage(source="none")
            else:  # the model answered but gave no counts: estimate, and say so
                usage = Usage(
                    input_tokens=prompt_chars // CHARS_PER_TOKEN_ESTIMATE,
                    output_tokens=response_chars // CHARS_PER_TOKEN_ESTIMATE,
                    source="estimated",
                )
        cost = usage.cost_usd
        if cost is None and model in self.prices:
            price_in, price_out = self.prices[model]
            cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000

        # Flags: anything about this call that needs a second look.
        found = list(flags)
        if usage.model:
            if not same_model(model, usage.model):
                found.append("model_mismatch")
            elif "," in usage.model:
                found.append("multiple_models")
        if (usage.stop_reason or "").lower() in TRUNCATED_STOP_REASONS:
            found.append("output_truncated")
        limits = limits or {}
        if limits.get("max_input_tokens") and usage.input_tokens > limits["max_input_tokens"]:
            found.append("input_over_limit")
        if limits.get("max_output_tokens") and usage.output_tokens >= limits["max_output_tokens"]:
            found.append("output_at_limit")
        if limits.get("context_window") and usage.total_tokens >= limits["context_window"]:
            found.append("context_full")

        merged = {**self._context, **labels, **usage.tags}
        core = {name: merged.pop(name, None) for name in CONTEXT_COLUMNS}
        row = {
            "time_utc": _now(),
            "model": model,
            "reported_model": usage.model,
            "provider": provider,
            **core,
            "input_tokens": usage.input_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": cost,
            "seconds": round(seconds, 4),
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
            "reasoning_chars": usage.reasoning_chars,
            "usage_source": usage.source,
            "stop_reason": usage.stop_reason,
            "flags": "",
            "error": error,
            "tags": json.dumps(merged, default=str, sort_keys=True),
            "raw_usage": json.dumps(usage.raw, default=str),
        }
        names = list(row)
        with self._lock:
            cursor = self._db.execute(
                f"INSERT INTO calls ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
                [row[n] for n in names],
            )
            row["call_id"] = cursor.lastrowid
            for budget in budgets:
                self._db.execute("INSERT INTO budget_charges VALUES (?, ?, ?)",
                                 [budget.name, row["call_id"], usage.total_tokens])
            if any(b.used() > b.limit for b in budgets):
                found.append("over_budget")
            row["flags"] = ",".join(dict.fromkeys(found))
            self._db.execute("UPDATE calls SET flags = ? WHERE call_id = ?", [row["flags"], row["call_id"]])
            self._db.commit()
        return row

    def measure(
        self,
        model: str,
        call: Callable[[], Any],
        *,
        extract: Callable[[Any], Optional[Usage]] = detect,
        prompt: Any = "",
        text: Optional[Callable[[Any], str]] = None,
        provider: str = "",
        budgets: Sequence[Budget] = (),
        limits: Optional[dict] = None,
        **labels: Any,
    ) -> Any:
        """Run call(), record what it used, and return its result unchanged.

        call     a function with no arguments that makes the model call,
                 usually a lambda
        extract  how to read token counts from the result (section 4). The
                 default recognizes OpenAI, Anthropic, Ollama, Gemini, Claude
                 Code and Codex output.
        prompt   what you sent (text or messages), for the size in characters
                 and for the estimate when no counts come back
        text     optional function returning the reply text from the result
        budgets  budgets to check before the call and charge after it. If any
                 is already spent, the call is NOT made: the refusal is
                 recorded and BudgetExceeded is raised.
        limits   per-request limits to flag (see record)

        If the call raises, the failure is recorded and the error is raised
        again, so your own error handling still works.
        """
        for budget in budgets:
            if budget.exhausted:
                error = BudgetExceeded(budget)
                self.record(model, None, prompt=prompt, error=str(error), provider=provider,
                            flags=["budget_refused"], **labels)
                raise error
        started = time.perf_counter()
        try:
            result = call()
        except Exception as exc:
            self.record(model, None, prompt=prompt, seconds=time.perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}"[:1000], provider=provider, **labels)
            raise
        seconds = time.perf_counter() - started
        reply = text(result) if text is not None else guess_text(result)
        self.record(model, extract(result), prompt=prompt, response=reply, seconds=seconds,
                    provider=provider, budgets=budgets, limits=limits, **labels)
        return result

    def wrap(self, function: Callable[..., Any], model: str, **options: Any) -> Callable[..., Any]:
        """Turn any function that calls a model into a metered one.

            ask_it = meter.wrap(ask_my_model, "my-model")
            ask_it("hello")          # counted

        The first positional argument is taken as the prompt.
        """
        def metered(*args: Any, **kwargs: Any) -> Any:
            prompt = options.get("prompt", args[0] if args else "")
            extras = {k: v for k, v in options.items() if k != "prompt"}
            return self.measure(model, lambda: function(*args, **kwargs), prompt=prompt, **extras)

        metered.__name__ = getattr(function, "__name__", "metered")
        metered.__doc__ = function.__doc__
        return metered

    # ---- configuration records

    def save_config(self, name: str, config: dict) -> int:
        """Store a snapshot of how a model was set up (versions, hashes,
        settings), so every run shows exactly what produced it."""
        with self._lock:
            cursor = self._db.execute("INSERT INTO configs (time_utc, name, config) VALUES (?, ?, ?)",
                                      [_now(), name, json.dumps(config, default=str, sort_keys=True)])
            self._db.commit()
        return cursor.lastrowid

    def configs(self, name: Optional[str] = None) -> list[dict]:
        where, values = (" WHERE name = ?", [name]) if name else ("", [])
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM configs{where} ORDER BY config_id", values).fetchall()
        return [{**dict(r), "config": json.loads(r["config"])} for r in rows]

    # ---- reading

    @staticmethod
    def _where(filters: dict) -> tuple[str, list]:
        clauses, values = [], []
        for name, value in filters.items():
            if name not in GROUP_COLUMNS:
                raise ValueError(f"cannot filter by {name!r}; use one of {GROUP_COLUMNS}")
            clauses.append(f"{name} = ?")
            values.append(value)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", values

    def rows(self, **filters: Any) -> list[dict]:
        """Every recorded call, oldest first. Filter with e.g. rows(model="x", round=2)."""
        where, values = self._where(filters)
        with self._lock:
            found = self._db.execute(f"SELECT * FROM calls{where} ORDER BY call_id", values).fetchall()
        return [dict(r) for r in found]

    def summary(self, by: Sequence[str] = ("model",), **filters: Any) -> list[dict]:
        """Totals grouped by any of GROUP_COLUMNS (section 2). A total is None
        when no call in the group reported that number."""
        by = [by] if isinstance(by, str) else list(by)
        for name in by:
            if name not in GROUP_COLUMNS:
                raise ValueError(f"cannot group by {name!r}; use any of {GROUP_COLUMNS}")
        where, values = self._where(filters)
        keys = ", ".join(by)
        sums = ", ".join(
            f"COALESCE(SUM({c}), 0) AS {c}" if c in ALWAYS_KNOWN else f"SUM({c}) AS {c}" for c in SUM_COLUMNS
        )
        sql = (
            f"SELECT {keys + ', ' if by else ''}COUNT(*) AS calls, "
            "SUM(error IS NOT NULL) AS failed_calls, "
            "SUM(usage_source = 'estimated') AS estimated_calls, "
            "SUM(flags != '') AS flagged_calls, "
            f"{sums} FROM calls{where}"
            + (f" GROUP BY {keys} ORDER BY {keys}" if by else "")
        )
        with self._lock:
            found = self._db.execute(sql, values).fetchall()
        return [dict(r) for r in found]

    def report(self, by: Sequence[str] = ("model",), **filters: Any) -> str:
        """The summary as a text table. "-" means the number is unavailable."""
        by = [by] if isinstance(by, str) else list(by)
        rows = self.summary(by, **filters)
        if not rows:
            return "No calls recorded."
        columns = by + ["calls", "failed_calls", "flagged_calls", "input_tokens", "cache_read_tokens",
                        "output_tokens", "reasoning_tokens", "total_tokens", "cost_usd", "seconds"]

        def cell(name: str, value: Any) -> str:
            if value is None:
                return "-"
            if name == "cost_usd":
                return f"{value:.4f}"
            if name == "seconds":
                return f"{value:.1f}"
            if isinstance(value, int) and name not in by:
                return f"{value:,}"
            return str(value)

        table = [columns] + [[cell(c, row.get(c)) for c in columns] for row in rows]
        widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
        lines = ["  ".join(text.ljust(width) for text, width in zip(line, widths)).rstrip() for line in table]
        estimated = sum(row["estimated_calls"] or 0 for row in rows)
        if estimated:
            lines.append(f"\nNote: {estimated} call(s) returned no token counts; those rows are estimates.")
        flagged = sum(row["flagged_calls"] or 0 for row in rows)
        if flagged:
            lines.append(f"Note: {flagged} call(s) are flagged; see the flags column in the CSV export.")
        return "\n".join(lines)

    def print_report(self, by: Sequence[str] = ("model",), **filters: Any) -> None:
        print(self.report(by, **filters))

    def export_csv(self, path: str, **filters: Any) -> int:
        """Write one CSV row per call. Empty cells are unavailable numbers.
        Returns how many calls were written."""
        rows = self.rows(**filters)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "TokenMeter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# =============================================================================
# 7. TALKING TO MODELS
#
# Small helpers that make the actual requests, using only Python's standard
# library. They return the provider's reply; pass them to meter.measure().
# =============================================================================

def _http(request: urllib.request.Request, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        exc.close()
        raise RuntimeError(f"HTTP {exc.code} from {request.full_url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {request.full_url} ({exc.reason}). "
                           "Is the model server running, and is the address right?") from exc


def post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout: float = 600.0) -> dict:
    return _http(urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    ), timeout)


def get_json(url: str, timeout: float = 10.0) -> dict:
    return _http(urllib.request.Request(url, method="GET"), timeout)


def ollama_chat(model: str, prompt: str, system: Optional[str] = None, host: str = OLLAMA_HOST,
                think: Any = None, timeout: float = 3600.0, **options: Any) -> dict:
    """One chat turn with a local Ollama model.

    think    True / False (or a level) for thinking models such as Qwen3.
             Leave as None for models without thinking (Qwen2.5): Ollama
             rejects the setting for them.
    options  Ollama options, e.g. temperature=0.2, top_p=0.9, top_k=40,
             seed=42, num_ctx=16384 (context window), num_predict=8192
             (maximum output tokens).
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload: dict = {"model": model, "messages": messages, "stream": False, "options": options}
    if think is not None:
        payload["think"] = think
    return post_json(f"{host.rstrip('/')}/api/chat", payload, timeout=timeout)


def ollama_gpu_percent(model: str, host: str = OLLAMA_HOST) -> Optional[float]:
    """Share of the loaded model that sits on the GPU (100 = fully offloaded,
    0 = CPU only), from Ollama's list of running models. None if unknown."""
    try:
        running = get_json(f"{host.rstrip('/')}/api/ps").get("models") or []
    except Exception:
        return None
    for entry in running:
        if model in (entry.get("name"), entry.get("model")) and entry.get("size"):
            return round(100.0 * _int(entry.get("size_vram")) / _int(entry["size"]), 1)
    return None


def ollama_model_info(model: str, host: str = OLLAMA_HOST) -> dict:
    """What exactly is being run: Ollama version, the model file's digest
    (its hash), quantization, parameter size and current GPU offload.
    Anything that cannot be read is reported as "unavailable"."""
    base = host.rstrip("/")
    info: dict = {"model": model, "ollama_version": UNAVAILABLE, "digest": UNAVAILABLE,
                  "quantization_level": UNAVAILABLE, "parameter_size": UNAVAILABLE,
                  "family": UNAVAILABLE, "format": UNAVAILABLE, "gpu_percent": UNAVAILABLE}
    try:
        info["ollama_version"] = get_json(f"{base}/api/version").get("version", UNAVAILABLE)
    except Exception:
        return info  # Ollama is not reachable
    try:
        for entry in get_json(f"{base}/api/tags").get("models") or []:
            if model in (entry.get("name"), entry.get("model")):
                info["digest"] = entry.get("digest", UNAVAILABLE)
                info["size_bytes"] = entry.get("size", UNAVAILABLE)
    except Exception:
        pass
    try:
        details = post_json(f"{base}/api/show", {"model": model}, timeout=30).get("details") or {}
        for key in ("quantization_level", "parameter_size", "family", "format"):
            info[key] = details.get(key) or UNAVAILABLE
    except Exception:
        pass
    gpu = ollama_gpu_percent(model, host)
    info["gpu_percent"] = gpu if gpu is not None else "unavailable (model not loaded)"
    return info


def openai_compatible_chat(base_url: str, model: str, messages: list, api_key: Optional[str] = None,
                           **params: Any) -> dict:
    """One chat completion from any server that speaks the OpenAI chat format
    (OpenAI, LM Studio, vLLM, llama.cpp server, Ollama's /v1, and others)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return post_json(f"{base_url.rstrip('/')}/chat/completions",
                     {"model": model, "messages": messages, **params}, headers)


def run_cli(command: Sequence[str], prompt: str, cwd: Optional[str] = None, timeout: float = 3600.0) -> str:
    """Run a command-line AI tool, sending the prompt on standard input, and
    return what it printed.

    The prompt goes through standard input, not the command line, because
    Windows cuts command lines at about 8,000 characters. The program is
    looked up on PATH first, so `claude` finds `claude.cmd` on Windows.
    """
    program = shutil.which(command[0])
    if program is None:
        raise RuntimeError(f"`{command[0]}` is not installed or not on PATH")
    try:
        finished = subprocess.run([program, *command[1:]], input=prompt, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{command[0]} timed out after {exc.timeout:.0f} seconds") from exc
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout or "").strip()[:500]
        raise RuntimeError(f"{command[0]} exited with code {finished.returncode}: {detail}")
    return finished.stdout


def cli_version(program: str) -> str:
    """What `<program> --version` prints, or "unavailable"."""
    path = shutil.which(program)
    if path is None:
        return UNAVAILABLE
    try:
        finished = subprocess.run([path, "--version"], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=30)
    except Exception:
        return UNAVAILABLE
    output = (finished.stdout or finished.stderr or "").strip()
    return output.splitlines()[0] if output else UNAVAILABLE


# =============================================================================
# 8. THE PLANNED MODELS
#
# ask() sends one request to a model from MODELS (section 1), through the
# meter, with that model's settings and limits applied. describe() builds the
# configuration record, and start_run() saves it at the start of a run.
#
# To support a new kind of client, add a branch for it in ask() and in
# describe(), and use its name as "client" in MODELS.
# =============================================================================

NOT_CONFIGURABLE = "not configurable (managed by the client)"
CLIENT_DEFAULT = "client-managed default"

# The program each command-line client runs.
_CLI_PROGRAM = {"codex": "codex", "claude-code": "claude"}


class Reply(NamedTuple):
    text: str  # what the model answered
    raw: Any  # the provider's full reply


def _plan(key: str) -> dict:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose one of: {', '.join(MODELS)}")
    return MODELS[key]


def ask(
    meter: TokenMeter,
    key: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    seed: Optional[int] = None,
    budgets: Sequence[Budget] = (),
    workdir: Optional[str] = None,
    allow_edits: bool = False,
    host: str = OLLAMA_HOST,
    timeout: float = 3600.0,
    **labels: Any,
) -> Reply:
    """Send one request to a planned model, through the meter.

    key          a short name from MODELS, e.g. "qwen3-30b-a3b"
    system       optional instructions sent before the prompt
    seed         used for local models (42, 43, 44 in the plan). Codex and
                 Claude Code have no seed control; it is stored as a label only.
    budgets      checked before the request and charged after it. For local
                 models the output limit sent to the model is the per-request
                 limit or whatever the budgets have left, whichever is smaller.
    workdir      folder the coding agent works in (Codex / Claude Code)
    allow_edits  let the coding agent change files in workdir
    labels       experiment, run, task, round, step, and anything else to store

    If the model was downgraded after hitting a usage limit, call this with
    the downgraded model's key and add downgraded_from="<key>".
    """
    plan = _plan(key)
    model, client = plan["model"], plan["client"]
    labels = {"config": key, "seed": seed, **labels}

    if client == "ollama":
        num_predict = output_allowance(budgets, LOCAL_LIMITS["max_output_tokens"])
        options = {**plan["sampling"], "num_ctx": LOCAL_LIMITS["context_window"], "num_predict": max(num_predict, 1)}
        if seed is not None:
            options["seed"] = seed

        def call_ollama() -> dict:
            reply = ollama_chat(model, prompt, system=system, host=host, think=plan["think"],
                                timeout=timeout, **options)
            gpu = ollama_gpu_percent(model, host)  # how much of the model is on the GPU right now
            reply["_gpu_percent"] = gpu if gpu is not None else UNAVAILABLE
            return reply

        def read_ollama(reply: dict) -> Optional[Usage]:
            usage = from_ollama(reply)
            if usage is not None:
                usage.tags.update(gpu_percent=reply["_gpu_percent"], num_predict=options["num_predict"],
                                  thinking=bool(plan["think"]))
            return usage

        raw = meter.measure(model, call_ollama, extract=read_ollama, prompt=[system, prompt], provider="ollama",
                            budgets=budgets, limits=LOCAL_LIMITS, **labels)
        return Reply(guess_text(raw), raw)

    # Coding-agent clients get the system text and the prompt as one input on
    # standard input, so every model receives the same words.
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    labels["effort"] = plan["effort"]

    if client == "claude-code":
        command = ["claude", "-p", "--output-format", "json", "--model", model, "--effort", plan["effort"],
                   "--permission-mode", "acceptEdits" if allow_edits else "plan", "--no-session-persistence"]
        raw = meter.measure(
            model, lambda: json.loads(run_cli(command, full_prompt, cwd=workdir, timeout=timeout)),
            extract=from_claude_code, prompt=full_prompt, provider="claude-code", budgets=budgets, **labels,
        )
        if raw.get("is_error"):  # the tokens are already recorded above
            raise RuntimeError(f"Claude Code reported an error: {str(raw.get('result'))[:500]}")
        return Reply(str(raw.get("result") or ""), raw)

    if client == "codex":
        # NOT VERIFIED against a live Codex CLI (flags and output format).
        command = ["codex", "exec", "--json", "--model", model, "-c", f"model_reasoning_effort={plan['effort']}",
                   "--sandbox", "workspace-write" if allow_edits else "read-only", "-"]
        raw = meter.measure(
            model, lambda: run_cli(command, full_prompt, cwd=workdir, timeout=timeout),
            extract=from_codex_jsonl, text=codex_final_message, prompt=full_prompt, provider="codex",
            budgets=budgets, **labels,
        )
        return Reply(codex_final_message(raw), raw)

    raise ValueError(f"no access path for client {client!r}")


def describe(key: str, host: str = OLLAMA_HOST) -> dict:
    """The configuration record for one planned model: exact model name,
    model-file hash, client and inference-service versions, and every setting,
    with controls the client does not expose marked as such."""
    plan = _plan(key)
    record: dict = {
        "config": key,
        "client": plan["client"],
        "model": plan["model"],
        "tokenmeter_version": __version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if plan["client"] == "ollama":
        info = ollama_model_info(plan["model"], host)
        quantization = info.get("quantization_level")
        record.update(info)
        record.update({
            "expected_quantization": EXPECTED_QUANTIZATION,
            "quantization_matches": (quantization == EXPECTED_QUANTIZATION) if quantization != UNAVAILABLE else UNAVAILABLE,
            "thinking": bool(plan["think"]),
            **plan["sampling"],
            "seed": f"set per run: {', '.join(map(str, LOCAL_SEEDS))}",
            "context_window": LOCAL_LIMITS["context_window"],
            "max_input_tokens_per_request": LOCAL_LIMITS["max_input_tokens"],
            "max_output_tokens_per_request": LOCAL_LIMITS["max_output_tokens"],
            "run_token_budget": LOCAL_RUN_TOKEN_BUDGET,
            "token_limit_enforcement": "enforced by the meter: checked before each request, output capped per request",
        })
    else:
        record.update({
            "client_version": cli_version(_CLI_PROGRAM[plan["client"]]),
            "reasoning_effort": plan["effort"],
            "temperature": NOT_CONFIGURABLE,
            "top_p": NOT_CONFIGURABLE,
            "top_k": NOT_CONFIGURABLE,
            "seed": NOT_CONFIGURABLE,
            "context_window": CLIENT_DEFAULT,
            "max_output_tokens_per_request": CLIENT_DEFAULT,
            "token_limit_enforcement": "recorded only: usage is counted per request, no exact limit is enforced inside one",
        })
    return record


def start_run(meter: TokenMeter, key: str, run: str, host: str = OLLAMA_HOST) -> Optional[Budget]:
    """Call once at the start of each run. Saves the configuration record and,
    for local models, returns the run's token budget (None for cloud models,
    whose usage is recorded but not limited)."""
    meter.save_config(key, {"run": run, **describe(key, host)})
    if _plan(key)["client"] == "ollama":
        return meter.budget(f"{run}/repair", LOCAL_RUN_TOKEN_BUDGET)
    return None


def _continue_run(meter: TokenMeter, key: str, run: str, host: str) -> Optional[Budget]:
    """start_run() the first time a run name is used in this ledger; after
    that, reopen the same budget without saving the record again."""
    if not any(c["config"].get("run") == run for c in meter.configs(key)):
        return start_run(meter, key, run, host)
    if _plan(key)["client"] == "ollama":
        return meter.budget(f"{run}/repair")
    return None


# =============================================================================
# 9. COMMAND LINE
#
# Each command is one small function below, and main() connects the words
# you type to them. To add a command: write a function like these, then add a
# parser for it in main().
# =============================================================================

def _cmd_list(args: argparse.Namespace) -> int:
    for key, plan in MODELS.items():
        extra = f"effort {plan['effort']}" if "effort" in plan else (
            ", ".join(f"{k} {v}" for k, v in plan["sampling"].items()) + (", thinking on" if plan["think"] else ""))
        print(f"{key:<28} {plan['client']:<12} {plan['model']:<36} {extra}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    print(json.dumps(describe(args.key, args.host), indent=2))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """One small real call, to confirm the connection reports tokens."""
    with TokenMeter(args.ledger, experiment="meter-check") as meter:
        meter.save_config(args.key, describe(args.key, args.host))
        try:
            reply = ask(meter, args.key, "Reply with the single word: ready", seed=args.seed, host=args.host,
                        step="check")
        except Exception as exc:
            print(f"The call failed: {exc}")
            return 1
        row = meter.rows(experiment="meter-check")[-1]
    print(f"reply: {reply.text.strip()[:200]}\n")
    for name in ("model", "reported_model", "usage_source", "input_tokens", "cache_read_tokens",
                 "cache_write_tokens", "output_tokens", "reasoning_tokens", "reasoning_chars", "total_tokens",
                 "cost_usd", "seconds", "stop_reason", "flags", "tags"):
        value = row[name]
        print(f"  {name:<20} {'(unavailable)' if value is None else value}")
    print("\nRaw usage from the provider (compare with the numbers above):")
    print(json.dumps(json.loads(row["raw_usage"]), indent=2)[:3000])
    if row["usage_source"] != "reported":
        print("\nWARNING: no token counts came back. Fix this connection before collecting data.")
        return 1
    if row["flags"]:
        print(f"\nCheck the flags: {row['flags']}")
    return 0


def _read_text(value: Optional[str], path: Optional[str], what: str) -> Optional[str]:
    if value is not None and path is not None:
        raise SystemExit(f"give the {what} as text or as a file, not both")
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    return value


def _cmd_ask(args: argparse.Namespace) -> int:
    """Send one prompt to a planned model and count its tokens. The reply goes
    to the screen (or --out), the token summary to the error stream."""
    prompt = _read_text(args.prompt, args.prompt_file, "prompt")
    if prompt is None:
        if sys.stdin is None or sys.stdin.isatty():
            raise SystemExit("give a prompt: as text, with --prompt-file, or piped in")
        prompt = sys.stdin.read()
    system = _read_text(None, args.system_file, "system text")
    labels = {name: getattr(args, name) for name in ("experiment", "run", "task", "round", "step")
              if getattr(args, name) is not None}
    with TokenMeter(args.ledger) as meter:
        budget = _continue_run(meter, args.key, args.run, args.host) if args.run else None
        try:
            reply = ask(meter, args.key, prompt, system=system, seed=args.seed,
                        budgets=[budget] if budget else [], workdir=args.workdir,
                        allow_edits=args.allow_edits, host=args.host, **labels)
        except BudgetExceeded as exc:
            print(f"Not sent: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"The call failed: {exc}", file=sys.stderr)
            return 1
        row = meter.rows()[-1]
        left = f" · budget left {budget.remaining():,}" if budget else ""
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(reply.text)
    else:
        print(reply.text)
    sys.stdout.flush()  # the reply first, then the token summary
    print(f"[tokenmeter] {args.key}: input {row['input_tokens']:,}, output {row['output_tokens']:,}, "
          f"total {row['total_tokens']:,} tokens ({row['usage_source']}){left} · "
          f"flags: {row['flags'] or 'none'}", file=sys.stderr)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    filters = {}
    for item in args.where:
        name, _, value = item.partition("=")
        filters[name] = int(value) if name == "round" and value.lstrip("-").isdigit() else value
    with TokenMeter(args.ledger) as meter:
        meter.print_report(by=args.by, **filters)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    with TokenMeter(args.ledger) as meter:
        print(f"wrote {meter.export_csv(args.csv)} calls to {args.csv}")
    return 0


def _cmd_budgets(args: argparse.Namespace) -> int:
    with TokenMeter(args.ledger) as meter:
        rows = meter.budgets()
    if not rows:
        print("No budgets in this ledger.")
    for row in rows:
        print(f"{row['name']}: used {row['used_tokens']:,} of {row['limit_tokens']:,} tokens "
              f"({row['remaining_tokens']:,} left, {row['calls']} calls)")
    return 0


def _cmd_configs(args: argparse.Namespace) -> int:
    with TokenMeter(args.ledger) as meter:
        print(json.dumps(meter.configs(), indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tokenmeter",
        description="Count, limit and report the tokens that AI model calls use.",
        epilog="On macOS and Linux, type python3 instead of python.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    model_keys = list(MODELS)

    # ---- the planned models
    command = commands.add_parser("list", help="show the planned models and their settings")
    command.set_defaults(run_command=_cmd_list)

    command = commands.add_parser("describe", help="print the configuration record for one model")
    command.add_argument("key", choices=model_keys, metavar="MODEL")
    command.add_argument("--host", default=OLLAMA_HOST, help="Ollama address (local models)")
    command.set_defaults(run_command=_cmd_describe)

    command = commands.add_parser("check", help="make one small real call and show what was recorded")
    command.add_argument("key", choices=model_keys, metavar="MODEL")
    command.add_argument("--ledger", default=DEFAULT_LEDGER)
    command.add_argument("--seed", type=int, default=LOCAL_SEEDS[0])
    command.add_argument("--host", default=OLLAMA_HOST, help="Ollama address (local models)")
    command.set_defaults(run_command=_cmd_check)

    command = commands.add_parser("ask", help="send a prompt to a planned model and count its tokens")
    command.add_argument("key", choices=model_keys, metavar="MODEL")
    command.add_argument("prompt", nargs="?", help="the prompt (or use --prompt-file, or pipe it in)")
    command.add_argument("--prompt-file", help="read the prompt from this text file")
    command.add_argument("--system-file", help="read instructions to send before the prompt from this file")
    command.add_argument("--out", help="write the reply to this file instead of the screen")
    command.add_argument("--ledger", default=DEFAULT_LEDGER)
    command.add_argument("--run", help="run name; for local models this also applies the run's token budget")
    command.add_argument("--experiment")
    command.add_argument("--task")
    command.add_argument("--round", type=int)
    command.add_argument("--step")
    command.add_argument("--seed", type=int, help="local models only, e.g. 42, 43 or 44")
    command.add_argument("--host", default=OLLAMA_HOST, help="Ollama address (local models)")
    command.add_argument("--workdir", help="folder a coding agent (Codex, Claude Code) works in")
    command.add_argument("--allow-edits", action="store_true", help="let a coding agent change files in --workdir")
    command.set_defaults(run_command=_cmd_ask)

    # ---- reading a ledger
    command = commands.add_parser("report", help="print token totals from a ledger")
    command.add_argument("ledger")
    command.add_argument("--by", nargs="+", default=["model"], choices=GROUP_COLUMNS, metavar="COLUMN",
                         help=f"group by any of: {', '.join(GROUP_COLUMNS)}")
    command.add_argument("--where", nargs="*", default=[], metavar="COLUMN=VALUE",
                         help="only calls matching these, e.g. --where model=qwen3:30b-a3b-q4_K_M round=2")
    command.set_defaults(run_command=_cmd_report)

    command = commands.add_parser("export", help="write every call in a ledger to a CSV file")
    command.add_argument("ledger")
    command.add_argument("csv")
    command.set_defaults(run_command=_cmd_export)

    command = commands.add_parser("budgets", help="show each token budget and what it has used")
    command.add_argument("ledger")
    command.set_defaults(run_command=_cmd_budgets)

    command = commands.add_parser("configs", help="print the saved configuration records")
    command.add_argument("ledger")
    command.set_defaults(run_command=_cmd_configs)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # a reply may hold characters a Windows console cannot show
        sys.stdout.reconfigure(errors="replace")
    return args.run_command(args)


if __name__ == "__main__":
    sys.exit(main())
