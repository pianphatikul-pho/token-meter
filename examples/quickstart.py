"""Start here. Runs with no internet, no API key and nothing installed:

    python3 examples/quickstart.py

It uses a pretend model so you can see every way of using the meter, then
shows how to plug in a model the meter has never heard of.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find tokenmeter.py

from tokenmeter import TokenMeter, Usage


# A stand-in for "some AI model". It replies with text plus its own usage
# numbers, in a format that is NOT OpenAI, Anthropic or Ollama.
def pretend_model(prompt: str) -> dict:
    return {
        "answer": f"Echo: {prompt}",
        "stats": {"tokens_in": len(prompt.split()) * 2, "tokens_out": 12, "thinking": 5},
    }


# Because the format is unknown, tell the meter how to read it. This is all
# you ever write to support a new model: one small function.
def read_pretend_usage(reply: dict) -> Usage:
    stats = reply["stats"]
    return Usage(
        input_tokens=stats["tokens_in"],
        output_tokens=stats["tokens_out"] + stats["thinking"],  # output includes reasoning
        reasoning_tokens=stats["thinking"],
        raw=stats,
    )


def main() -> None:
    # ":memory:" keeps this demo off your disk. Use a file name such as
    # "token_ledger.db" to keep the records.
    meter = TokenMeter(":memory:", experiment="quickstart", prices={"pretend-1": (1.00, 4.00)})

    # 1. measure(): the meter makes the call for you and counts it.
    for round_number in (1, 2, 3):
        prompt = f"Fix the login bug, attempt {round_number}"
        reply = meter.measure(
            "pretend-1",
            lambda: pretend_model(prompt),
            extract=read_pretend_usage,
            text=lambda r: r["answer"],
            prompt=prompt,
            task="SQLI-01",
            round=round_number,
            step="patch",
        )
        print("model said:", reply["answer"])

    # 2. context(): label a whole block of calls at once.
    with meter.context(task="AUTH-01", round=1):
        meter.measure("pretend-1", lambda: pretend_model("Add a rate limit"),
                      extract=read_pretend_usage, prompt="Add a rate limit", step="patch")

    # 3. wrap(): make a metered version of your own function, then just use it.
    ask = meter.wrap(pretend_model, "pretend-1", extract=read_pretend_usage, step="chat")
    ask("What is a token?")

    # 4. record(): you already have the numbers (for example after streaming).
    meter.record("other-model", Usage(input_tokens=900, output_tokens=150), task="SQLI-01", round=1)

    # 5. A failed call is still written down, then the error reaches you as usual.
    try:
        meter.measure("pretend-1", lambda: 1 / 0, prompt="this call breaks", task="SQLI-01", round=4)
    except ZeroDivisionError:
        pass

    print("\nTotals per model")
    meter.print_report(by=["model"])
    print("\nTotals per task and round")
    meter.print_report(by=["model", "task", "round"])


if __name__ == "__main__":
    main()
