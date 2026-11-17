"""A stand-in model server for trying tokenmeter with no GPU, no download and
no API key.

    python3 examples/local_test_server.py

It answers on http://127.0.0.1:11435 in the same formats as:

  * Ollama's API (/api/chat, /api/tags, /api/ps, /api/show, /api/version),
    for the three planned Qwen models, and
  * the OpenAI chat format (/v1/chat/completions), for any model name.

Its replies are canned text and its token counts are made up (about one token
per four characters). It only exists so you can check that the meter, the
budget, the reports and planned_models.py work end to end before you point
them at a real model. Never use its numbers as results.

Options:
    --port 11435     port to listen on (a real Ollama uses 11434)
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The planned Qwen tags. Only qwen3 supports thinking, as with real Ollama.
MODELS = {
    "qwen2.5:14b-instruct-q4_K_M": {"family": "qwen2", "parameter_size": "14.8B", "thinking": False},
    "qwen2.5-coder:14b-instruct-q4_K_M": {"family": "qwen2", "parameter_size": "14.8B", "thinking": False},
    "qwen3:30b-a3b-q4_K_M": {"family": "qwen3moe", "parameter_size": "30.5B", "thinking": True},
}
VERSION = "0.0.0-stand-in"


def count(text: str) -> int:
    return max(1, len(text) // 4)


def answer_for(prompt: str) -> str:
    if "single word" in prompt.lower():
        return "ready"
    return (f"[stand-in reply] I received {len(prompt)} characters. A real model would answer here. "
            "This server only exists to test the token meter.")


class Handler(BaseHTTPRequestHandler):
    server_version = "tokenmeter-stand-in"

    def send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path == "/api/version":
            return self.send(200, {"version": VERSION})
        if self.path == "/api/tags":
            return self.send(200, {"models": [
                {"name": tag, "model": tag, "digest": f"stand-in-{index:02d}", "size": 9_000_000_000}
                for index, tag in enumerate(MODELS, start=1)
            ]})
        if self.path == "/api/ps":  # pretend every model is fully on the GPU
            return self.send(200, {"models": [
                {"name": tag, "model": tag, "size": 9_000_000_000, "size_vram": 9_000_000_000} for tag in MODELS
            ]})
        self.send(404, {"error": f"no such endpoint: {self.path}"})

    def do_POST(self) -> None:
        try:
            request = self.body()
        except json.JSONDecodeError:
            return self.send(400, {"error": "request body is not JSON"})
        if self.path == "/api/show":
            model = request.get("model", "")
            if model not in MODELS:
                return self.send(404, {"error": f"model '{model}' not found"})
            info = MODELS[model]
            return self.send(200, {"details": {"format": "gguf", "family": info["family"],
                                               "parameter_size": info["parameter_size"],
                                               "quantization_level": "Q4_K_M"}})
        if self.path == "/api/chat":
            return self.ollama_chat(request)
        if self.path == "/v1/chat/completions":
            return self.openai_chat(request)
        self.send(404, {"error": f"no such endpoint: {self.path}"})

    def ollama_chat(self, request: dict) -> None:
        model = request.get("model", "")
        if model not in MODELS:
            return self.send(404, {"error": f"model '{model}' not found, try pulling it first"})
        if request.get("think") and not MODELS[model]["thinking"]:
            return self.send(400, {"error": f"\"{model}\" does not support thinking"})
        prompt = "\n".join(m.get("content", "") for m in request.get("messages", []))
        options = request.get("options") or {}
        thinking = "Let me look at the request first." if request.get("think") else None
        content = answer_for(prompt)
        output = count(content) + (count(thinking) if thinking else 0)
        done_reason = "stop"
        limit = options.get("num_predict")
        if isinstance(limit, int) and 0 < limit < output:  # honour the output limit like Ollama does
            output, done_reason = limit, "length"
            content = content[: limit * 4]
        message = {"role": "assistant", "content": content}
        if thinking:
            message["thinking"] = thinking
        self.send(200, {
            "model": model, "message": message, "done": True, "done_reason": done_reason,
            "prompt_eval_count": count(prompt), "eval_count": output,
            "total_duration": 120_000_000, "load_duration": 10_000_000,
            "prompt_eval_duration": 30_000_000, "eval_duration": 80_000_000,
        })

    def openai_chat(self, request: dict) -> None:
        prompt = "\n".join(str(m.get("content", "")) for m in request.get("messages", []))
        content = answer_for(prompt)
        self.send(200, {
            "object": "chat.completion", "model": request.get("model", "stand-in"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": count(prompt), "completion_tokens": count(content),
                      "total_tokens": count(prompt) + count(content),
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        })

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stand-in model server for testing tokenmeter.")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Stand-in model server on http://127.0.0.1:{args.port}  (press Ctrl+C to stop)")
    print("Its token counts are made up. Use it to test the setup, never for results.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
