"""Local models through Ollama (Qwen and others). Nothing to install besides Ollama.

    ollama pull qwen2.5:14b-instruct-q4_K_M
    python3 examples/ollama_qwen.py qwen2.5:14b-instruct-q4_K_M

To try it without Ollama, start examples/local_test_server.py and add its
address:

    python3 examples/ollama_qwen.py qwen2.5:14b-instruct-q4_K_M http://127.0.0.1:11435

Records to token_ledger.db in the current folder. For the three Qwen
configurations in the project plan, with their exact settings, budget and
limits already applied, use planned_models.py instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenmeter import TokenMeter, from_ollama, ollama_chat

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b-instruct-q4_K_M"
host = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:11434"
meter = TokenMeter("token_ledger.db", experiment="ollama-demo")
budget = meter.budget("ollama-demo/total", 20_000)  # stop after 20,000 tokens in total

prompts = [
    "In one sentence, what is SQL injection?",
    "In one sentence, how is it prevented?",
]
for round_number, prompt in enumerate(prompts, start=1):
    reply = meter.measure(
        model,
        # For a thinking model such as Qwen3, add think=True here.
        lambda: ollama_chat(model, prompt, host=host, temperature=0.2, top_p=0.9, top_k=40, seed=42,
                            num_ctx=16384, num_predict=budget.max_output(8192)),
        extract=from_ollama,
        prompt=prompt,
        provider="ollama",
        budgets=[budget],
        limits={"max_input_tokens": 8192, "max_output_tokens": 8192, "context_window": 16384},
        round=round_number,
    )
    print(f"[round {round_number}] {reply['message']['content'].strip()}\n")

meter.print_report(by=["model", "round"], experiment="ollama-demo")
print(budget)
