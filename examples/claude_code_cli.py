"""Coding agents run from the command line: Claude Code (and, the same way, Codex).

    python3 examples/claude_code_cli.py claude-haiku-5-5

The tool is started as a normal program; the meter reads the usage out of
what it prints. One run of an agent can be several model turns, and the
agent's own built-in instructions count as input, so expect larger numbers
than a plain API call with the same prompt.

For Codex, run ["codex", "exec", "--json", prompt] and use
`extract=from_codex_jsonl` (not yet verified against a live Codex CLI).
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenmeter import TokenMeter, from_claude_code

model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
meter = TokenMeter("token_ledger.db", experiment="claude-code-demo")
prompt = "Reply with the single word: ready"


def run_claude_code() -> dict:
    finished = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json", "--model", model],
        capture_output=True, text=True, check=True, timeout=600,
    )
    return json.loads(finished.stdout)


reply = meter.measure(model, run_claude_code, extract=from_claude_code, prompt=prompt,
                      provider="claude-code", round=1)
print(reply.get("result"))
meter.print_report(by=["model"], experiment="claude-code-demo")
