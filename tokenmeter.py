"""tokenmeter: count the tokens that any AI model call uses.

One file, standard library only (Python 3.9+, Windows / macOS / Linux). Copy
it next to your code.

    from tokenmeter import TokenMeter

    meter = TokenMeter("token_ledger.db")
    reply = meter.measure("my-model", lambda: call_my_model(prompt), prompt=prompt, round=1)
    meter.print_report(by=["model", "round"])

The meter sits between your code and the model. It runs the call, reads the
token counts out of the model's reply, and saves one row per call in a small
SQLite file (the ledger). It can also hold a token budget and refuse calls
once the budget is spent.

How counts are normalized, so every provider means the same thing:

    input_tokens       everything sent to the model, cached tokens included
      cache_read_tokens    part of the input that was served from a cache
      cache_write_tokens   part of the input that was written to a cache
    output_tokens      everything generated, reasoning / thinking included
      reasoning_tokens     part of the output that was reasoning
    total_tokens       input_tokens + output_tokens

A breakdown column is EMPTY (not 0) when the provider does not report it, so
"unavailable" is never mistaken for "zero".

`usage_source` on each row says where the numbers came from:

    reported    the provider returned them
    estimated   the provider returned none, so characters / 4 was used
    none        the call failed or was refused before any usage came back

`flags` on each row lists anything that needs a second look:

    model_mismatch     a different model answered than the one you asked for
    multiple_models    an agent tool used more than one model in this call
    output_truncated   the reply was cut off at the output limit
    input_over_limit   more input tokens than your per-request limit
    output_at_limit    output reached your per-request limit
    context_full       input + output filled the context window
    budget_refused     the call was not made: a budget was already spent
    over_budget        this call pushed a budget past its limit
"""

from __future__ import annotations

import argparse
import csv
import json
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
from typing import Any, Callable, Iterator, Optional, Sequence

__version__ = "1.1.0"

CHARS_PER_TOKEN_ESTIMATE = 4
UNAVAILABLE = "unavailable"

# Labels you can attach to a call and group reports by.
CONTEXT_COLUMNS = ("experiment", "run", "task", "round", "step")
GROUP_COLUMNS = ("model", "provider", "reported_model") + CONTEXT_COLUMNS + ("usage_source", "stop_reason")
ALWAYS_KNOWN = ("input_tokens", "output_tokens", "total_tokens", "seconds", "prompt_chars", "response_chars")
MAY_BE_UNAVAILABLE = ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "reasoning_chars", "cost_usd")
SUM_COLUMNS = (
    "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens", "cost_usd", "seconds", "prompt_chars", "response_chars", "reasoning_chars",
)
CSV_COLUMNS = (
    ("call_id", "time_utc", "model", "reported_model", "provider") + CONTEXT_COLUMNS + SUM_COLUMNS
    + ("usage_source", "stop_reason", "flags", "error", "tags")
)
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


# --------------------------------------------------------------------- usage

@dataclass
class Usage:
    """Token counts for one call, already normalized (see the module docstring).

    Leave a breakdown field as None when the provider does not report it.
    """

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


# ----------------------------------------------------- provider extractors
#
# An extractor takes whatever the model call returned and gives back a Usage,
# or None if the reply carried no token counts. To support a new provider,
# write one more function like these and pass it as `extract=`.

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
    relying on it.
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

    Good enough for the formats above. Pass a specific extractor if you know
    which one you need, or your own function for anything else.
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


# ------------------------------------------------------------------- budgets

class BudgetExceeded(RuntimeError):
    """Raised instead of making a call when a budget has nothing left."""

    def __init__(self, budget: "Budget"):
        self.budget = budget
        super().__init__(f"budget exhausted: {budget.name} (used {budget.used():,} of {budget.limit:,} tokens)")


class Budget:
    """A token allowance stored in the ledger. Get one from `meter.budget()`.

    The meter checks it BEFORE each call and refuses the call when nothing is
    left. It cannot stop a call that is already running, so one call can still
    overshoot: cap the output yourself with `max_output()`. For agent tools
    that make many model turns inside one call, treat this as a stop rule
    between calls, not an exact limit.
    """

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


# --------------------------------------------------------------------- meter

class TokenMeter:
    """The middleman. Create one per ledger file and send every call through it.

    path     the SQLite ledger file (created if missing). ":memory:" keeps
             nothing on disk.
    prices   optional {model: (input_usd_per_million, output_usd_per_million)},
             used for cost when the provider does not report one.
    context  default labels for every call: experiment, run, task, round, step.
             Any other name is stored as a tag.
    """

    def __init__(self, path: str = "token_ledger.db", prices: Optional[dict] = None, **context: Any):
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
        a streamed reply finishes). Otherwise use `measure`.

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
        """Run `call()`, record what it used, and return its result unchanged.

        call     a function with no arguments that makes the model call,
                 usually a lambda
        extract  how to read token counts from the result. The default
                 recognizes OpenAI, Anthropic, Ollama, Gemini, Claude Code
                 and Codex output.
        prompt   what you sent (text or messages), for the size in characters
                 and for the estimate when no counts come back
        text     optional function returning the reply text from the result
        budgets  budgets to check before the call and charge after it. If any
                 is already spent, the call is NOT made: the refusal is
                 recorded and BudgetExceeded is raised.
        limits   per-request limits to flag (see `record`)

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

            ask = meter.wrap(ask_my_model, "my-model")
            ask("hello")          # counted

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
        """Totals grouped by any of: model, provider, reported_model,
        experiment, run, task, round, step. A total is None when no call in
        the group reported that number."""
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


# ------------------------------------------------------ ready-made callers
#
# Small helpers so you can talk to a model with nothing installed. They return
# the provider's reply; pass them to `meter.measure`.

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


def ollama_chat(model: str, prompt: str, system: Optional[str] = None, host: str = "http://localhost:11434",
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


def ollama_gpu_percent(model: str, host: str = "http://localhost:11434") -> Optional[float]:
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


def ollama_model_info(model: str, host: str = "http://localhost:11434") -> dict:
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
    return (finished.stdout or finished.stderr).strip().splitlines()[0] if (finished.stdout or finished.stderr).strip() else UNAVAILABLE


# ----------------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenmeter", description="Read a token ledger.")
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="print token totals")
    report.add_argument("ledger")
    report.add_argument("--by", nargs="+", default=["model"], choices=GROUP_COLUMNS, metavar="COLUMN",
                        help=f"group by any of: {', '.join(GROUP_COLUMNS)}")
    report.add_argument("--where", nargs="*", default=[], metavar="COLUMN=VALUE",
                        help="only calls matching these, e.g. --where model=qwen3:30b round=2")

    export = commands.add_parser("export", help="write every call to a CSV file")
    export.add_argument("ledger")
    export.add_argument("csv")

    budgets = commands.add_parser("budgets", help="show each token budget and what it has used")
    budgets.add_argument("ledger")

    configs = commands.add_parser("configs", help="print the saved model configuration records")
    configs.add_argument("ledger")

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # never crash a report on a Windows console
        sys.stdout.reconfigure(errors="replace")
    with TokenMeter(args.ledger) as meter:
        if args.command == "report":
            filters = {}
            for item in args.where:
                name, _, value = item.partition("=")
                filters[name] = int(value) if name == "round" and value.lstrip("-").isdigit() else value
            meter.print_report(by=args.by, **filters)
        elif args.command == "export":
            print(f"wrote {meter.export_csv(args.csv)} calls to {args.csv}")
        elif args.command == "budgets":
            rows = meter.budgets()
            if not rows:
                print("No budgets in this ledger.")
            for row in rows:
                print(f"{row['name']}: used {row['used_tokens']:,} of {row['limit_tokens']:,} tokens "
                      f"({row['remaining_tokens']:,} left, {row['calls']} calls)")
        else:
            print(json.dumps(meter.configs(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
