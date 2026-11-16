"""Local models through Ollama (Qwen and others). Nothing to install besides Ollama.

    ollama pull qwen3:14b
    python3 examples/ollama_qwen.py qwen3:14b

Records to token_ledger.db in the current folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenmeter import TokenMeter, from_ollama, ollama_chat

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"
meter = TokenMeter("token_ledger.db", experiment="ollama-demo")

prompts = [
    "In one sentence, what is SQL injection?",
    "In one sentence, how is it prevented?",
]
for round_number, prompt in enumerate(prompts, start=1):
    reply = meter.measure(
        model,
        lambda: ollama_chat(model, prompt),
        extract=from_ollama,
        prompt=prompt,
        provider="ollama",
        round=round_number,
    )
    print(f"[round {round_number}] {reply['message']['content'].strip()}\n")

meter.print_report(by=["model", "round"], experiment="ollama-demo")
