"""Claude through the official Anthropic Python SDK.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python3 examples/anthropic_claude.py claude-haiku-5-5

Model ids: claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

from tokenmeter import TokenMeter, from_anthropic

model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-5-5"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
meter = TokenMeter("token_ledger.db", experiment="claude-demo")

prompt = "In one sentence, what is broken access control?"

# No automatic fallback to another model is configured here on purpose: if a
# different model answered, its tokens would be recorded under the wrong name.
response = meter.measure(
    model,
    lambda: client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    ),
    extract=from_anthropic,
    prompt=prompt,
    provider="anthropic",
    round=1,
)

if response.stop_reason == "refusal":
    print("The model declined this request.")
for block in response.content:
    if block.type == "text":
        print(block.text)

meter.print_report(by=["model"], experiment="claude-demo")
