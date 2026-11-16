"""Tests for tokenmeter. Standard library only:

    python3 -m unittest -v
"""

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import tokenmeter
from tokenmeter import (
    TokenMeter, Usage, detect, from_anthropic, from_claude_code, from_codex_jsonl, from_gemini,
    from_ollama, from_openai, guess_text,
)

OPENAI_CHAT = {
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165,
              "prompt_tokens_details": {"cached_tokens": 100},
              "completion_tokens_details": {"reasoning_tokens": 30}},
}
OPENAI_RESPONSES = {
    "output_text": "hello",
    "usage": {"input_tokens": 80, "output_tokens": 20,
              "input_tokens_details": {"cached_tokens": 64},
              "output_tokens_details": {"reasoning_tokens": 8}},
}
ANTHROPIC = {
    "type": "message", "stop_reason": "end_turn",
    "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 200,
              "cache_creation_input_tokens": 50, "output_tokens": 40},
}
OLLAMA = {"message": {"content": "hello"}, "prompt_eval_count": 33, "eval_count": 7,
          "total_duration": 1_500_000_000}
GEMINI = {
    "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "thoughtsTokenCount": 25,
                      "cachedContentTokenCount": 5, "totalTokenCount": 85},
}
CLAUDE_CODE = {"result": "ready", "total_cost_usd": 0.0123, "num_turns": 2, "usage": ANTHROPIC["usage"]}
CODEX = "\n".join(json.dumps(e) for e in [
    {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
    {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60,
                                         "output_tokens": 30, "reasoning_output_tokens": 20}},
    {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 5}},
])


def counts(usage):
    return (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens,
            usage.output_tokens, usage.reasoning_tokens)


class Extractors(unittest.TestCase):
    def test_openai_chat_completions(self):
        self.assertEqual(counts(from_openai(OPENAI_CHAT)), (120, 100, 0, 45, 30))

    def test_openai_responses_api(self):
        self.assertEqual(counts(from_openai(OPENAI_RESPONSES)), (80, 64, 0, 20, 8))

    def test_anthropic_adds_cached_input(self):
        usage = from_anthropic(ANTHROPIC)
        self.assertEqual(counts(usage), (260, 200, 50, 40, 0))
        self.assertEqual(usage.total_tokens, 300)

    def test_ollama(self):
        usage = from_ollama(OLLAMA)
        self.assertEqual(counts(usage), (33, 0, 0, 7, 0))
        self.assertEqual(usage.cost_usd, 0.0)

    def test_gemini_counts_thinking_as_output(self):
        self.assertEqual(counts(from_gemini(GEMINI)), (50, 5, 0, 35, 25))

    def test_gemini_when_thinking_is_already_inside_candidates(self):
        reply = {"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 35,
                                   "thoughtsTokenCount": 25, "totalTokenCount": 85}}
        self.assertEqual(counts(from_gemini(reply)), (50, 0, 0, 35, 25))

    def test_claude_code_keeps_reported_cost(self):
        for form in (CLAUDE_CODE, json.dumps(CLAUDE_CODE)):
            usage = from_claude_code(form)
            self.assertEqual(counts(usage), (260, 200, 50, 40, 0))
            self.assertEqual(usage.cost_usd, 0.0123)
            self.assertEqual(usage.raw["num_turns"], 2)

    def test_codex_sums_turns(self):
        self.assertEqual(counts(from_codex_jsonl(CODEX)), (150, 60, 0, 35, 20))

    def test_detect_picks_the_right_provider(self):
        expected = {
            "openai chat": (OPENAI_CHAT, (120, 100, 0, 45, 30)),
            "openai responses": (OPENAI_RESPONSES, (80, 64, 0, 20, 8)),
            "anthropic": (ANTHROPIC, (260, 200, 50, 40, 0)),
            "ollama": (OLLAMA, (33, 0, 0, 7, 0)),
            "gemini": (GEMINI, (50, 5, 0, 35, 25)),
            "claude code": (CLAUDE_CODE, (260, 200, 50, 40, 0)),
            "claude code text": (json.dumps(CLAUDE_CODE), (260, 200, 50, 40, 0)),
            "codex": (CODEX, (150, 60, 0, 35, 20)),
        }
        for name, (reply, want) in expected.items():
            with self.subTest(name):
                self.assertEqual(counts(detect(reply)), want)

    def test_no_usage_means_none(self):
        for reply in ({}, {"choices": []}, "plain text", None, {"usage": None}):
            self.assertIsNone(detect(reply))

    def test_sdk_style_objects_work_like_dicts(self):
        class FakeSdkResponse:  # like a pydantic model from an SDK
            def model_dump(self):
                return ANTHROPIC

        self.assertEqual(counts(detect(FakeSdkResponse())), (260, 200, 50, 40, 0))
        plain = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=9, completion_tokens=1))
        self.assertEqual(counts(from_openai(plain)), (9, 0, 0, 1, 0))

    def test_reply_text_is_found(self):
        for reply in (OPENAI_CHAT, OPENAI_RESPONSES, ANTHROPIC, OLLAMA, GEMINI, "hello"):
            self.assertEqual(guess_text(reply), "hello")
        self.assertEqual(guess_text(CLAUDE_CODE), "ready")


class Meter(unittest.TestCase):
    def setUp(self):
        self.meter = TokenMeter(":memory:", experiment="test", prices={"priced": (2.0, 10.0)})
        self.addCleanup(self.meter.close)

    def test_measure_records_and_returns_the_result_unchanged(self):
        result = self.meter.measure("gpt-x", lambda: OPENAI_CHAT, prompt="hi there", task="T1", round=2, step="patch")
        self.assertIs(result, OPENAI_CHAT)
        (row,) = self.meter.rows()
        self.assertEqual((row["model"], row["experiment"], row["task"], row["round"], row["step"]),
                         ("gpt-x", "test", "T1", 2, "patch"))
        self.assertEqual((row["input_tokens"], row["output_tokens"], row["total_tokens"]), (120, 45, 165))
        self.assertEqual((row["prompt_chars"], row["response_chars"], row["usage_source"]), (8, 5, "reported"))
        self.assertEqual(json.loads(row["raw_usage"])["total_tokens"], 165)

    def test_failed_call_is_recorded_and_raised(self):
        def broken():
            raise TimeoutError("model did not answer")

        with self.assertRaises(TimeoutError):
            self.meter.measure("gpt-x", broken, prompt="hello", round=1)
        (row,) = self.meter.rows()
        self.assertEqual(row["error"], "TimeoutError: model did not answer")
        self.assertEqual((row["usage_source"], row["total_tokens"], row["prompt_chars"]), ("none", 0, 5))

    def test_missing_usage_is_estimated_and_flagged(self):
        self.meter.measure("mystery", lambda: "x" * 400, prompt="y" * 800)
        (row,) = self.meter.rows()
        self.assertEqual((row["input_tokens"], row["output_tokens"], row["usage_source"]), (200, 100, "estimated"))
        self.assertIn("estimates", self.meter.report())

    def test_cost_uses_provider_cost_then_price_table(self):
        self.meter.record("priced", Usage(input_tokens=1_000_000, output_tokens=100_000))
        self.meter.record("priced", Usage(input_tokens=5, output_tokens=5, cost_usd=0.5))
        self.meter.record("unpriced", Usage(input_tokens=5, output_tokens=5))
        costs = [row["cost_usd"] for row in self.meter.rows()]
        self.assertEqual(costs, [3.0, 0.5, None])

    def test_context_labels_nest_and_extra_labels_become_tags(self):
        with self.meter.context(task="A", round=1):
            self.meter.record("m", Usage(1, 1))
            with self.meter.context(round=2, vulnerability="sqli"):
                self.meter.record("m", Usage(1, 1))
        self.meter.record("m", Usage(1, 1))
        rows = self.meter.rows()
        self.assertEqual([(r["task"], r["round"]) for r in rows], [("A", 1), ("A", 2), (None, None)])
        self.assertEqual(json.loads(rows[1]["tags"]), {"vulnerability": "sqli"})

    def test_wrap_meters_an_ordinary_function(self):
        ask = self.meter.wrap(lambda prompt: OLLAMA, "qwen", step="chat")
        self.assertIs(ask("how many tokens?"), OLLAMA)
        (row,) = self.meter.rows()
        self.assertEqual((row["model"], row["step"], row["total_tokens"], row["prompt_chars"]), ("qwen", "chat", 40, 16))

    def test_summary_groups_and_filters(self):
        for model, round_number, tokens in (("a", 1, 10), ("a", 1, 20), ("a", 2, 5), ("b", 1, 100)):
            self.meter.record(model, Usage(input_tokens=tokens, output_tokens=tokens), round=round_number)
        by_model = {row["model"]: row for row in self.meter.summary(by="model")}
        self.assertEqual((by_model["a"]["calls"], by_model["a"]["total_tokens"]), (3, 70))
        self.assertEqual(by_model["b"]["total_tokens"], 200)
        rounds = self.meter.summary(by=["model", "round"], model="a")
        self.assertEqual([(r["round"], r["total_tokens"]) for r in rounds], [(1, 60), (2, 10)])
        (everything,) = self.meter.summary(by=[])
        self.assertEqual(everything["total_tokens"], 270)
        with self.assertRaises(ValueError):
            self.meter.summary(by=["model; DROP TABLE calls"])

    def test_ledger_file_survives_reopening_and_exports_csv(self):
        folder = tempfile.mkdtemp()
        ledger, out = os.path.join(folder, "ledger.db"), os.path.join(folder, "calls.csv")
        with TokenMeter(ledger) as first:
            first.measure("claude", lambda: ANTHROPIC, prompt="hi", round=1)
        with TokenMeter(ledger) as second:
            second.measure("qwen", lambda: OLLAMA, prompt="hi", round=1)
            self.assertEqual(second.export_csv(out), 2)
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(r["model"], r["total_tokens"]) for r in rows], [("claude", "300"), ("qwen", "40")])
        self.assertEqual(tuple(rows[0]), tokenmeter.CSV_COLUMNS)
        self.assertEqual(tokenmeter.main(["report", ledger, "--by", "model", "round", "--where", "round=1"]), 0)


class ReadyMadeCallers(unittest.TestCase):
    """The built-in HTTP helpers, against a stand-in server on this computer."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen = cls.seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("Authorization"), body))
                if self.path == "/broken/chat/completions":
                    self.send_response(429)
                    self.end_headers()
                    self.wfile.write(b'{"error": "slow down"}')
                    return
                reply = OLLAMA if self.path == "/api/chat" else OPENAI_CHAT
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(reply).encode())

            def log_message(self, *_):
                pass

        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.meter = TokenMeter(":memory:")
        self.addCleanup(self.meter.close)

    def test_ollama_chat_is_metered(self):
        reply = self.meter.measure("qwen3:14b", lambda: tokenmeter.ollama_chat(
            "qwen3:14b", "hi", system="be brief", host=self.url), prompt="hi", round=1)
        self.assertEqual(reply["message"]["content"], "hello")
        path, _, body = self.seen[-1]
        self.assertEqual((path, body["model"], body["stream"], len(body["messages"])), ("/api/chat", "qwen3:14b", False, 2))
        (row,) = self.meter.rows()
        self.assertEqual((row["input_tokens"], row["output_tokens"], row["cost_usd"]), (33, 7, 0.0))

    def test_openai_compatible_chat_is_metered_and_sends_the_key(self):
        messages = [{"role": "user", "content": "hi"}]
        self.meter.measure("gpt-x", lambda: tokenmeter.openai_compatible_chat(
            self.url + "/v1", "gpt-x", messages, api_key="test-key"), prompt=messages)
        path, auth, body = self.seen[-1]
        self.assertEqual((path, auth, body["messages"]), ("/v1/chat/completions", "Bearer test-key", messages))
        (row,) = self.meter.rows()
        self.assertEqual((row["total_tokens"], row["cache_read_tokens"], row["reasoning_tokens"]), (165, 100, 30))

    def test_http_error_is_recorded_as_a_failed_call(self):
        with self.assertRaises(RuntimeError):
            self.meter.measure("gpt-x", lambda: tokenmeter.openai_compatible_chat(
                self.url + "/broken", "gpt-x", []), prompt="hi")
        (row,) = self.meter.rows()
        self.assertIn("HTTP 429", row["error"])
        self.assertEqual(row["usage_source"], "none")


if __name__ == "__main__":
    unittest.main()
