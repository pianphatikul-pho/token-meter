"""GPT models, or any server that speaks the OpenAI chat format.

Set the key in your shell first (never write it into a file):

    export OPENAI_API_KEY=...
    python3 examples/openai_compatible.py <model-name>

For another OpenAI-compatible server, also set its address, for example:

    export OPENAI_BASE_URL=http://localhost:1234/v1     (LM Studio)
    export OPENAI_BASE_URL=http://localhost:11434/v1    (Ollama)

If you use the official `openai` Python package instead, nothing else
changes: pass its response object to the meter the same way.

    response = meter.measure(model, lambda: client.chat.completions.create(...),
                             extract=from_openai, prompt=messages)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenmeter import TokenMeter, from_openai, openai_compatible_chat

if len(sys.argv) < 2:
    sys.exit("usage: python3 examples/openai_compatible.py <model-name>")
model = sys.argv[1]
base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
api_key = os.environ.get("OPENAI_API_KEY")

meter = TokenMeter("token_ledger.db", experiment="openai-demo")
messages = [{"role": "user", "content": "In one sentence, what is path traversal?"}]

reply = meter.measure(
    model,
    lambda: openai_compatible_chat(base_url, model, messages, api_key=api_key),
    extract=from_openai,
    prompt=messages,
    provider="openai-compatible",
    round=1,
)
print(reply["choices"][0]["message"]["content"])
meter.print_report(by=["model"], experiment="openai-demo")
