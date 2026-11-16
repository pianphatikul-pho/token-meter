"""tokenmeter: count the tokens that any AI model call uses.

One file, standard library only (Python 3.9+). Copy it next to your code.

    from tokenmeter import TokenMeter

    meter = TokenMeter("token_ledger.db")
    reply = meter.measure("my-model", lambda: call_my_model(prompt), prompt=prompt, round=1)
    meter.print_report(by=["model", "round"])

The meter sits between your code and the model. It runs the call, reads the
token counts out of the model's reply, and saves one row per call in a small
SQLite file (the ledger). Reports and a CSV export come from that file.

How counts are normalized, so every provider means the same thing:

    input_tokens       everything sent to the model, cached tokens included
      cache_read_tokens    part of the input that was served from a cache
      cache_write_tokens   part of the input that was written to a cache
    output_tokens      everything generated, reasoning / thinking included
      reasoning_tokens     part of the output that was reasoning, if reported
    total_tokens       input_tokens + output_tokens

`usage_source` on each row says where the numbers came from:

    reported    the provider returned them
    estimated   the provider returned none, so characters / 4 was used
    none        the call failed before any usage came back
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, Sequence

__version__ = "1.0.0"

CHARS_PER_TOKEN_ESTIMATE = 4

# Labels you can attach to a call and group reports by.
CONTEXT_COLUMNS = ("experiment", "run", "task", "round", "step")
GROUP_COLUMNS = ("model", "provider") + CONTEXT_COLUMNS + ("usage_source",)
SUM_COLUMNS = (
    "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens", "cost_usd", "seconds", "prompt_chars", "response_chars",
)
CSV_COLUMNS = (
    ("call_id", "time_utc", "model", "provider") + CONTEXT_COLUMNS + SUM_COLUMNS
    + ("usage_source", "error", "tags")
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_utc TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    experiment TEXT, run TEXT, task TEXT, round INTEGER, step TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    seconds REAL NOT NULL DEFAULT 0,
    prompt_chars INTEGER NOT NULL DEFAULT 0,
    response_chars INTEGER NOT NULL DEFAULT 0,
    usage_source TEXT NOT NULL DEFAULT 'reported',
    error TEXT,
    tags TEXT NOT NULL DEFAULT '{}',
    raw_usage TEXT NOT NULL DEFAULT '{}'
)
"""


# --------------------------------------------------------------------- usage

@dataclass
class Usage:
    """Token counts for one call, already normalized (see the module docstring)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None  # only when the provider reports a cost
    source: str = "reported"
    raw: dict = field(default_factory=dict)  # the provider's own numbers, kept for auditing

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
    usage = _as_dict(_as_dict(response).get("usage"))
    if not usage:
        return None
    if "prompt_tokens" in usage or "completion_tokens" in usage:  # Chat Completions
        details_in = _as_dict(usage.get("prompt_tokens_details"))
        details_out = _as_dict(usage.get("completion_tokens_details"))
        return Usage(
            input_tokens=_int(usage.get("prompt_tokens")),
            output_tokens=_int(usage.get("completion_tokens")),
            cache_read_tokens=_int(details_in.get("cached_tokens")),
            reasoning_tokens=_int(details_out.get("reasoning_tokens")),
            raw=usage,
        )
    details_in = _as_dict(usage.get("input_tokens_details"))  # Responses API
    details_out = _as_dict(usage.get("output_tokens_details"))
    return Usage(
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_tokens=_int(details_in.get("cached_tokens")),
        reasoning_tokens=_int(details_out.get("reasoning_tokens")),
        raw=usage,
    )


def from_anthropic(response: Any) -> Optional[Usage]:
    """Claude through the Anthropic Messages API.

    Anthropic reports uncached input separately from cache reads and cache
    writes, so the three are added to get the full input.
    """
    usage = _as_dict(_as_dict(response).get("usage"))
    if not usage:
        return None
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_write = _int(usage.get("cache_creation_input_tokens"))
    return Usage(
        input_tokens=_int(usage.get("input_tokens")) + cache_read + cache_write,
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        raw=usage,
    )


def from_ollama(response: Any) -> Optional[Usage]:
    """Local models through Ollama's own API (/api/chat or /api/generate)."""
    data = _as_dict(response)
    if "prompt_eval_count" not in data and "eval_count" not in data:
        return None
    return Usage(
        input_tokens=_int(data.get("prompt_eval_count")),
        output_tokens=_int(data.get("eval_count")),
        cost_usd=0.0,  # local: no API bill
        raw={k: v for k, v in data.items() if k.endswith(("_count", "_duration"))},
    )


def from_gemini(response: Any) -> Optional[Usage]:
    """Google Gemini (`usageMetadata`). Thinking tokens are counted as output."""
    data = _as_dict(response)
    usage = _as_dict(data.get("usageMetadata") or data.get("usage_metadata"))
    if not usage:
        return None

    def pick(camel: str, snake: str) -> int:
        return _int(usage.get(camel, usage.get(snake)))

    prompt = pick("promptTokenCount", "prompt_token_count")
    candidates = pick("candidatesTokenCount", "candidates_token_count")
    thoughts = pick("thoughtsTokenCount", "thoughts_token_count")
    total = pick("totalTokenCount", "total_token_count")
    # Use the provider's own total to decide whether thoughts are already
    # inside the candidates count.
    output = candidates if total and prompt + candidates == total else candidates + thoughts
    return Usage(
        input_tokens=prompt,
        output_tokens=output,
        cache_read_tokens=pick("cachedContentTokenCount", "cached_content_token_count"),
        reasoning_tokens=thoughts,
        raw=usage,
    )


def from_claude_code(response: Any) -> Optional[Usage]:
    """The JSON printed by `claude -p ... --output-format json` (dict or text).

    One call here can be several model turns; the counts are for all of them.
    """
    data = json.loads(response) if isinstance(response, str) else _as_dict(response)
    usage = from_anthropic(data)
    if usage is None:
        return None
    if data.get("total_cost_usd") is not None:
        usage.cost_usd = float(data["total_cost_usd"])
    usage.raw = {k: v for k, v in data.items() if k != "result"}
    return usage


def from_codex_jsonl(output: Any) -> Optional[Usage]:
    """The event lines printed by `codex exec --json`.

    NOT VERIFIED against a live Codex CLI: check one real output before
    relying on it.
    """
    if not isinstance(output, str):
        return None
    turns = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed" and event.get("usage"):
            turns.append(event["usage"])
    if not turns:
        return None
    return Usage(
        input_tokens=sum(_int(t.get("input_tokens")) for t in turns),
        output_tokens=sum(_int(t.get("output_tokens")) for t in turns),
        cache_read_tokens=sum(_int(t.get("cached_input_tokens")) for t in turns),
        reasoning_tokens=sum(_int(t.get("reasoning_output_tokens")) for t in turns),
        raw={"turns": turns},
    )


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
        return response
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
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute(_SCHEMA)
            self._db.commit()

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
        **labels: Any,
    ) -> dict:
        """Write one call to the ledger and return the stored row.

        Use this directly when you already have the numbers (for example after
        a streamed reply finishes). Otherwise use `measure`.
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

        merged = {**self._context, **labels}
        core = {name: merged.pop(name, None) for name in CONTEXT_COLUMNS}
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
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
            "usage_source": usage.source,
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
            self._db.commit()
        row["call_id"] = cursor.lastrowid
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

        If the call raises, the failure is recorded and the error is raised
        again, so your own error handling still works.
        """
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
                    provider=provider, **labels)
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
        """Totals grouped by any of: model, provider, experiment, run, task, round, step."""
        by = [by] if isinstance(by, str) else list(by)
        for name in by:
            if name not in GROUP_COLUMNS:
                raise ValueError(f"cannot group by {name!r}; use any of {GROUP_COLUMNS}")
        where, values = self._where(filters)
        keys = ", ".join(by)
        # cost stays empty (not 0) when no call in the group has a known cost
        sums = ", ".join(
            f"SUM({c}) AS {c}" if c == "cost_usd" else f"COALESCE(SUM({c}), 0) AS {c}" for c in SUM_COLUMNS
        )
        sql = (
            f"SELECT {keys + ', ' if by else ''}COUNT(*) AS calls, "
            "SUM(error IS NOT NULL) AS failed_calls, "
            "SUM(usage_source = 'estimated') AS estimated_calls, "
            f"{sums} FROM calls{where}"
            + (f" GROUP BY {keys} ORDER BY {keys}" if by else "")
        )
        with self._lock:
            found = self._db.execute(sql, values).fetchall()
        return [dict(r) for r in found]

    def report(self, by: Sequence[str] = ("model",), **filters: Any) -> str:
        """The summary as a text table."""
        by = [by] if isinstance(by, str) else list(by)
        rows = self.summary(by, **filters)
        if not rows:
            return "No calls recorded."
        columns = by + ["calls", "failed_calls", "input_tokens", "cache_read_tokens", "output_tokens",
                        "reasoning_tokens", "total_tokens", "cost_usd", "seconds"]

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
        return "\n".join(lines)

    def print_report(self, by: Sequence[str] = ("model",), **filters: Any) -> None:
        print(self.report(by, **filters))

    def export_csv(self, path: str, **filters: Any) -> int:
        """Write one CSV row per call. Returns how many calls were written."""
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
# Two small helpers so you can talk to a model with nothing installed. They
# return the provider's JSON reply as a dict; pass them to `meter.measure`.

def post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout: float = 600.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        exc.close()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def ollama_chat(model: str, prompt: str, system: Optional[str] = None,
                host: str = "http://localhost:11434", **options: Any) -> dict:
    """One chat turn with a local Ollama model."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return post_json(f"{host.rstrip('/')}/api/chat",
                     {"model": model, "messages": messages, "stream": False, "options": options})


def openai_compatible_chat(base_url: str, model: str, messages: list, api_key: Optional[str] = None,
                           **params: Any) -> dict:
    """One chat completion from any server that speaks the OpenAI chat format
    (OpenAI, LM Studio, vLLM, llama.cpp server, Ollama's /v1, and others)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return post_json(f"{base_url.rstrip('/')}/chat/completions",
                     {"model": model, "messages": messages, **params}, headers)


# ----------------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenmeter", description="Read a token ledger.")
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="print token totals")
    report.add_argument("ledger")
    report.add_argument("--by", nargs="+", default=["model"], choices=GROUP_COLUMNS, metavar="COLUMN",
                        help=f"group by any of: {', '.join(GROUP_COLUMNS)}")
    report.add_argument("--where", nargs="*", default=[], metavar="COLUMN=VALUE",
                        help="only calls matching these, e.g. --where model=qwen3:14b round=2")

    export = commands.add_parser("export", help="write every call to a CSV file")
    export.add_argument("ledger")
    export.add_argument("csv")

    args = parser.parse_args(argv)
    with TokenMeter(args.ledger) as meter:
        if args.command == "report":
            filters = {}
            for item in args.where:
                name, _, value = item.partition("=")
                filters[name] = int(value) if name == "round" and value.lstrip("-").isdigit() else value
            meter.print_report(by=args.by, **filters)
        else:
            print(f"wrote {meter.export_csv(args.csv)} calls to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
