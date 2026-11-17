"""The model-and-agent configurations planned in Chapter 3, wired to the token meter.

    python3 planned_models.py list                  what is planned
    python3 planned_models.py describe KEY          versions, hashes, settings for one configuration
    python3 planned_models.py check KEY             one small REAL call, then show what the meter recorded

For a Qwen configuration, add --host http://127.0.0.1:11435 to either of the
last two to use the stand-in server in examples/local_test_server.py instead
of a real Ollama.

In code:

    from tokenmeter import TokenMeter
    from planned_models import ask, start_run

    meter = TokenMeter("token_ledger.db")
    budget = start_run(meter, "qwen3-30b-a3b", run="sqli-01/loop/seed-42")
    reply = ask(meter, "qwen3-30b-a3b", prompt, seed=42, budgets=[budget],
                run="sqli-01/loop/seed-42", task="SQLI-01", round=1, step="patch")
    print(reply.text)

Access paths, as in the plan: GPT through Codex, Claude through Claude Code,
Qwen through a local Ollama service.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any, NamedTuple, Optional, Sequence

import tokenmeter as tm
from tokenmeter import Budget, TokenMeter

# ---- fixed settings from Chapter 3, section 4 --------------------------------

# Local (Qwen) requests: 16,384-token context window, at most 8,192 input and
# 8,192 output tokens per request.
LOCAL_LIMITS = {"context_window": 16_384, "max_input_tokens": 8_192, "max_output_tokens": 8_192}
# 120,000 input + output tokens across all repair and prompt-refinement
# requests in one run.
LOCAL_RUN_TOKEN_BUDGET = 120_000
LOCAL_SEEDS = (42, 43, 44)
EXPECTED_QUANTIZATION = "Q4_K_M"
OLLAMA_HOST = "http://localhost:11434"

NOT_CONFIGURABLE = "not configurable (managed by the client)"
CLIENT_DEFAULT = "client-managed default"

_QWEN_14B_SAMPLING = {"temperature": 0.2, "top_p": 0.9, "top_k": 40}
_QWEN_30B_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

MODELS: dict = {
    # GPT through the Codex CLI. TODO: confirm these two model ids with
    # `codex` itself; they are written here as the plan names them.
    "gpt-6-astra": {"client": "codex", "model": "gpt-6-astra", "effort": "xhigh"},
    "gpt-6-luna": {"client": "codex", "model": "gpt-6-luna", "effort": "xhigh"},
    # Claude through Claude Code. "Extra High" effort is `xhigh`.
    "claude-opus-5-5": {"client": "claude-code", "model": "claude-opus-5-5", "effort": "xhigh"},
    "claude-sonnet-5-5": {"client": "claude-code", "model": "claude-sonnet-5-5", "effort": "xhigh"},
    "claude-haiku-5-5": {"client": "claude-code", "model": "claude-haiku-5-5", "effort": "xhigh"},
    # Qwen through Ollama, Q4_K_M files. Tags checked on ollama.com (2026-11-17).
    "qwen2.5-14b-instruct": {
        "client": "ollama", "model": "qwen2.5:14b-instruct-q4_K_M", "think": None, "sampling": _QWEN_14B_SAMPLING,
    },
    "qwen2.5-coder-14b-instruct": {
        "client": "ollama", "model": "qwen2.5-coder:14b-instruct-q4_K_M", "think": None, "sampling": _QWEN_14B_SAMPLING,
    },
    # Note: plain `qwen3:30b` is a DIFFERENT file (the later "thinking-2507"
    # release). This tag is the original Qwen3-30B-A3B at Q4_K_M.
    "qwen3-30b-a3b": {
        "client": "ollama", "model": "qwen3:30b-a3b-q4_K_M", "think": True, "sampling": _QWEN_30B_SAMPLING,
    },
}

_CLI_PROGRAM = {"codex": "codex", "claude-code": "claude"}


class Reply(NamedTuple):
    text: str  # what the model answered
    raw: Any  # the provider's full reply


def _plan(key: str) -> dict:
    if key not in MODELS:
        raise KeyError(f"unknown configuration {key!r}; choose one of: {', '.join(MODELS)}")
    return MODELS[key]


# ---- running a request -------------------------------------------------------

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
    """Send one request to a planned configuration, through the meter.

    budgets      checked before the request and charged after it. For Qwen the
                 output limit sent to the model is 8,192 or whatever the
                 budgets have left, whichever is smaller.
    seed         used for Qwen (42, 43, 44 in the plan). Codex and Claude Code
                 have no seed control; it is stored as a label only.
    workdir      folder the coding agent works in (Codex / Claude Code)
    allow_edits  let the coding agent change files in workdir
    labels       experiment, run, task, round, step, and anything else to store

    If the model was downgraded after hitting a usage limit, call this with
    the downgraded configuration's key and add `downgraded_from="<key>"`.
    """
    plan = _plan(key)
    model, client = plan["model"], plan["client"]
    labels = {"config": key, "seed": seed, **labels}

    if client == "ollama":
        num_predict = tm.output_allowance(budgets, LOCAL_LIMITS["max_output_tokens"])
        options = {**plan["sampling"], "num_ctx": LOCAL_LIMITS["context_window"], "num_predict": max(num_predict, 1)}
        if seed is not None:
            options["seed"] = seed

        def call_ollama() -> dict:
            reply = tm.ollama_chat(model, prompt, system=system, host=host, think=plan["think"],
                                   timeout=timeout, **options)
            gpu = tm.ollama_gpu_percent(model, host)  # how much of the model is on the GPU right now
            reply["_gpu_percent"] = gpu if gpu is not None else tm.UNAVAILABLE
            return reply

        def read_ollama(reply: dict) -> Optional[tm.Usage]:
            usage = tm.from_ollama(reply)
            if usage is not None:
                usage.tags.update(gpu_percent=reply["_gpu_percent"], num_predict=options["num_predict"],
                                  thinking=bool(plan["think"]))
            return usage

        raw = meter.measure(model, call_ollama, extract=read_ollama, prompt=[system, prompt], provider="ollama",
                            budgets=budgets, limits=LOCAL_LIMITS, **labels)
        return Reply(tm.guess_text(raw), raw)

    # Coding-agent clients get the system text and the prompt as one input on
    # standard input, so every model receives the same words.
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    labels["effort"] = plan["effort"]

    if client == "claude-code":
        command = ["claude", "-p", "--output-format", "json", "--model", model, "--effort", plan["effort"],
                   "--permission-mode", "acceptEdits" if allow_edits else "plan", "--no-session-persistence"]
        raw = meter.measure(
            model, lambda: json.loads(tm.run_cli(command, full_prompt, cwd=workdir, timeout=timeout)),
            extract=tm.from_claude_code, prompt=full_prompt, provider="claude-code", budgets=budgets, **labels,
        )
        if raw.get("is_error"):  # the tokens are already recorded above
            raise RuntimeError(f"Claude Code reported an error: {str(raw.get('result'))[:500]}")
        return Reply(str(raw.get("result") or ""), raw)

    if client == "codex":
        # NOT VERIFIED against a live Codex CLI (flags and output format).
        command = ["codex", "exec", "--json", "--model", model, "-c", f"model_reasoning_effort={plan['effort']}",
                   "--sandbox", "workspace-write" if allow_edits else "read-only", "-"]
        raw = meter.measure(
            model, lambda: tm.run_cli(command, full_prompt, cwd=workdir, timeout=timeout),
            extract=tm.from_codex_jsonl, text=tm.codex_final_message, prompt=full_prompt, provider="codex",
            budgets=budgets, **labels,
        )
        return Reply(tm.codex_final_message(raw), raw)

    raise ValueError(f"no access path for client {client!r}")


# ---- recording the configuration ---------------------------------------------

def describe(key: str, host: str = OLLAMA_HOST) -> dict:
    """Everything the plan says to record about one configuration: exact model
    identifier, model-file hash, client and inference-service versions, and
    each setting, with controls the client does not expose marked as such."""
    plan = _plan(key)
    record: dict = {
        "config": key,
        "client": plan["client"],
        "model": plan["model"],
        "tokenmeter_version": tm.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if plan["client"] == "ollama":
        info = tm.ollama_model_info(plan["model"], host)
        quantization = info.get("quantization_level")
        record.update(info)
        record.update({
            "expected_quantization": EXPECTED_QUANTIZATION,
            "quantization_matches": (quantization == EXPECTED_QUANTIZATION) if quantization != tm.UNAVAILABLE else tm.UNAVAILABLE,
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
            "client_version": tm.cli_version(_CLI_PROGRAM[plan["client"]]),
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
    for local Qwen models, returns the run's 120,000-token budget (None for
    cloud models, whose usage is recorded but not limited)."""
    meter.save_config(key, {"run": run, **describe(key, host)})
    if _plan(key)["client"] == "ollama":
        return meter.budget(f"{run}/repair", LOCAL_RUN_TOKEN_BUDGET)
    return None


# ---- command line ------------------------------------------------------------

def _check(key: str, ledger: str, seed: Optional[int], host: str) -> int:
    """One small real call, to confirm the connection reports tokens."""
    with TokenMeter(ledger, experiment="meter-check") as meter:
        meter.save_config(key, describe(key, host))
        try:
            reply = ask(meter, key, "Reply with the single word: ready", seed=seed, host=host, step="check")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The planned model-and-agent configurations.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show the planned configurations")
    describe_cmd = commands.add_parser("describe", help="print the configuration record for one")
    describe_cmd.add_argument("key", choices=list(MODELS))
    describe_cmd.add_argument("--host", default=OLLAMA_HOST, help="Ollama address (Qwen configurations)")
    check = commands.add_parser("check", help="make one small real call and show what was recorded")
    check.add_argument("key", choices=list(MODELS))
    check.add_argument("--ledger", default="token_ledger.db")
    check.add_argument("--seed", type=int, default=LOCAL_SEEDS[0])
    check.add_argument("--host", default=OLLAMA_HOST, help="Ollama address (Qwen configurations)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # a model reply may hold characters a Windows console cannot show
        sys.stdout.reconfigure(errors="replace")

    if args.command == "list":
        for key, plan in MODELS.items():
            extra = f"effort {plan['effort']}" if "effort" in plan else (
                ", ".join(f"{k} {v}" for k, v in plan["sampling"].items()) + (", thinking on" if plan["think"] else ""))
            print(f"{key:<28} {plan['client']:<12} {plan['model']:<36} {extra}")
        return 0
    if args.command == "describe":
        print(json.dumps(describe(args.key, args.host), indent=2))
        return 0
    return _check(args.key, args.ledger, args.seed, args.host)


if __name__ == "__main__":
    sys.exit(main())
