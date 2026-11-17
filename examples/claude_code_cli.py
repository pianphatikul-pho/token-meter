"""Coding agents run from the command line: Claude Code (and, the same way, Codex).

    python3 examples/claude_code_cli.py claude-haiku-5-5

Run this from a normal terminal, not from inside a Claude Code session:
Claude Code refuses to start inside another Claude Code session.

The tool is started as a normal program with the prompt on standard input;
the meter reads the usage out of what it prints. One run of an agent can be
several model turns, and the agent's own built-in instructions count as
input, so expect larger numbers than a plain API call with the same prompt.

For Codex, run ["codex", "exec", "--json", "-"] the same way and use
`extract=from_codex_jsonl` (not yet verified against a live Codex CLI).
The planned GPT and Claude configurations are ready-made in planned_models.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenmeter import TokenMeter, from_claude_code, run_cli

model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
meter = TokenMeter("token_ledger.db", experiment="claude-code-demo")
prompt = "Reply with the single word: ready"
command = ["claude", "-p", "--output-format", "json", "--model", model]

reply = meter.measure(model, lambda: json.loads(run_cli(command, prompt)), extract=from_claude_code,
                      prompt=prompt, provider="claude-code", round=1)
print(reply.get("result"))
meter.print_report(by=["model", "reported_model"], experiment="claude-code-demo")
