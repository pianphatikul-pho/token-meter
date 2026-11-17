"""Tests for tokenmeter. Standard library only:

    python3 -m unittest
"""

import csv
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest import mock

import tokenmeter
from tokenmeter import (
    BudgetExceeded, TokenMeter, Usage, detect, from_anthropic, from_claude_code, from_codex_jsonl,
    from_gemini, from_ollama, from_openai, guess_text,
)

OPENAI_CHAT = {
    "model": "gpt-x-2026-01-01",
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165,
              "prompt_tokens_details": {"cached_tokens": 100},
              "completion_tokens_details": {"reasoning_tokens": 30}},
}
OPENAI_RESPONSES = {
    "output_text": "hello", "status": "completed",
    "usage": {"input_tokens": 80, "output_tokens": 20,
              "input_tokens_details": {"cached_tokens": 64},
              "output_tokens_details": {"reasoning_tokens": 8}},
}
ANTHROPIC = {
    "type": "message", "model": "claude-opus-5-5", "stop_reason": "end_turn",
    "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 200,
              "cache_creation_input_tokens": 50, "output_tokens": 40},
}
OLLAMA = {"model": "qwen3:30b-a3b-q4_K_M", "message": {"content": "hello", "thinking": "let me think"},
          "done_reason": "stop", "prompt_eval_count": 33, "eval_count": 7, "total_duration": 1_500_000_000}
GEMINI = {
    "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "thoughtsTokenCount": 25,
                      "cachedContentTokenCount": 5, "totalTokenCount": 85},
}
CLAUDE_CODE = {
    "result": "ready", "is_error": False, "subtype": "success", "total_cost_usd": 0.0123, "num_turns": 2,
    "usage": ANTHROPIC["usage"],
    "modelUsage": {"claude-opus-5-5": {"inputTokens": 10, "outputTokens": 40}},
}
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
    """None in a result means the provider does not report that number."""

    def test_openai_chat_completions(self):
        usage = from_openai(OPENAI_CHAT)
        self.assertEqual(counts(usage), (120, 100, None, 45, 30))
        self.assertEqual((usage.model, usage.stop_reason), ("gpt-x-2026-01-01", "stop"))

    def test_openai_responses_api(self):
        self.assertEqual(counts(from_openai(OPENAI_RESPONSES)), (80, 64, None, 20, 8))

    def test_anthropic_adds_cached_input(self):
        usage = from_anthropic(ANTHROPIC)
        self.assertEqual(counts(usage), (260, 200, 50, 40, None))
        self.assertEqual((usage.total_tokens, usage.model, usage.stop_reason), (300, "claude-opus-5-5", "end_turn"))

    def test_ollama_keeps_thinking_size_but_not_a_made_up_token_count(self):
        usage = from_ollama(OLLAMA)
        self.assertEqual(counts(usage), (33, None, None, 7, None))
        self.assertEqual((usage.cost_usd, usage.reasoning_chars, usage.stop_reason), (0.0, 12, "stop"))
        self.assertEqual(usage.model, "qwen3:30b-a3b-q4_K_M")

    def test_gemini_counts_thinking_as_output(self):
        self.assertEqual(counts(from_gemini(GEMINI)), (50, 5, None, 35, 25))

    def test_gemini_when_thinking_is_already_inside_candidates(self):
        reply = {"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 35,
                                   "thoughtsTokenCount": 25, "totalTokenCount": 85}}
        self.assertEqual(counts(from_gemini(reply)), (50, None, None, 35, 25))

    def test_claude_code_keeps_cost_turns_and_models_used(self):
        for form in (CLAUDE_CODE, json.dumps(CLAUDE_CODE)):
            usage = from_claude_code(form)
            self.assertEqual(counts(usage), (260, 200, 50, 40, None))
            self.assertEqual((usage.cost_usd, usage.model, usage.tags["turns"]), (0.0123, "claude-opus-5-5", 2))

    def test_codex_sums_turns(self):
        usage = from_codex_jsonl(CODEX)
        self.assertEqual(counts(usage), (150, 60, None, 35, 20))
        self.assertEqual(usage.tags["turns"], 2)

    def test_detect_picks_the_right_provider(self):
        expected = {
            "openai chat": (OPENAI_CHAT, (120, 100, None, 45, 30)),
            "openai responses": (OPENAI_RESPONSES, (80, 64, None, 20, 8)),
            "anthropic": (ANTHROPIC, (260, 200, 50, 40, None)),
            "ollama": (OLLAMA, (33, None, None, 7, None)),
            "gemini": (GEMINI, (50, 5, None, 35, 25)),
            "claude code": (CLAUDE_CODE, (260, 200, 50, 40, None)),
            "claude code text": (json.dumps(CLAUDE_CODE), (260, 200, 50, 40, None)),
            "codex": (CODEX, (150, 60, None, 35, 20)),
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

        self.assertEqual(counts(detect(FakeSdkResponse())), (260, 200, 50, 40, None))
        plain = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=9, completion_tokens=1))
        self.assertEqual(counts(from_openai(plain)), (9, None, None, 1, None))

    def test_reply_text_is_found(self):
        for reply in (OPENAI_CHAT, OPENAI_RESPONSES, ANTHROPIC, OLLAMA, GEMINI, "hello"):
            self.assertEqual(guess_text(reply), "hello")
        self.assertEqual(guess_text(CLAUDE_CODE), "ready")
        self.assertEqual(guess_text(CODEX), "done")


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
        self.assertEqual((row["reported_model"], row["stop_reason"], row["flags"]), ("gpt-x-2026-01-01", "stop", ""))
        self.assertEqual(json.loads(row["raw_usage"])["total_tokens"], 165)

    def test_unreported_numbers_are_stored_empty_not_zero(self):
        self.meter.measure("qwen3:30b-a3b-q4_K_M", lambda: OLLAMA, prompt="hi")
        (row,) = self.meter.rows()
        self.assertIsNone(row["cache_read_tokens"])
        self.assertIsNone(row["reasoning_tokens"])
        self.assertEqual(row["reasoning_chars"], 12)
        (total,) = self.meter.summary(by="model")
        self.assertIsNone(total["reasoning_tokens"])
        self.assertEqual(total["total_tokens"], 40)

    def test_a_different_model_answering_is_flagged(self):
        downgraded = {**OPENAI_CHAT, "model": "gpt-6-luna"}
        self.meter.measure("gpt-6-astra", lambda: downgraded, prompt="hi")
        helper = {**CLAUDE_CODE, "modelUsage": {"claude-opus-5-5": {}, "claude-haiku-5-5": {}}}
        self.meter.measure("claude-opus-5-5", lambda: helper, prompt="hi")
        self.meter.measure("claude-sonnet-5-5", lambda: helper, prompt="hi")
        rows = self.meter.rows()
        self.assertEqual([r["flags"] for r in rows], ["model_mismatch", "multiple_models", "model_mismatch"])
        self.assertEqual(rows[1]["reported_model"], "claude-haiku-5-5, claude-opus-5-5")
        self.assertEqual(self.meter.summary(by="model")[0]["flagged_calls"], 1)

    def test_per_request_limits_are_flagged(self):
        limits = {"max_input_tokens": 8192, "max_output_tokens": 8192, "context_window": 16384}
        self.meter.record("m", Usage(9000, 100), limits=limits)
        self.meter.record("m", Usage(100, 8192, stop_reason="length"), limits=limits)
        self.meter.record("m", Usage(8192, 8192), limits=limits)
        self.meter.record("m", Usage(8192, 100), limits=limits)
        self.assertEqual([r["flags"] for r in self.meter.rows()], [
            "input_over_limit", "output_truncated,output_at_limit", "output_at_limit,context_full", ""])

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
        ask = self.meter.wrap(lambda prompt: OLLAMA, "qwen3:30b-a3b-q4_K_M", step="chat")
        self.assertIs(ask("how many tokens?"), OLLAMA)
        (row,) = self.meter.rows()
        self.assertEqual((row["step"], row["total_tokens"], row["prompt_chars"]), ("chat", 40, 16))

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
            first.measure("claude-opus-5-5", lambda: ANTHROPIC, prompt="hi", round=1)
        with TokenMeter(ledger) as second:
            second.measure("qwen3:30b-a3b-q4_K_M", lambda: OLLAMA, prompt="hi", round=1)
            self.assertEqual(second.export_csv(out), 2)
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(r["total_tokens"], r["reasoning_tokens"]) for r in rows], [("300", ""), ("40", "")])
        self.assertEqual(tuple(rows[0]), tokenmeter.CSV_COLUMNS)
        for command in (["report", ledger, "--by", "model", "round", "--where", "round=1"],
                        ["budgets", ledger], ["configs", ledger]):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(tokenmeter.main(command), 0)
            self.assertTrue(out.getvalue().strip())

    def test_ledger_from_version_1_0_is_upgraded_in_place(self):
        path = os.path.join(tempfile.mkdtemp(), "old.db")
        old = sqlite3.connect(path)
        old.execute("""CREATE TABLE calls (call_id INTEGER PRIMARY KEY AUTOINCREMENT, time_utc TEXT NOT NULL,
            model TEXT NOT NULL, provider TEXT NOT NULL DEFAULT '', experiment TEXT, run TEXT, task TEXT,
            round INTEGER, step TEXT, input_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL, seconds REAL NOT NULL DEFAULT 0,
            prompt_chars INTEGER NOT NULL DEFAULT 0, response_chars INTEGER NOT NULL DEFAULT 0,
            usage_source TEXT NOT NULL DEFAULT 'reported', error TEXT, tags TEXT NOT NULL DEFAULT '{}',
            raw_usage TEXT NOT NULL DEFAULT '{}')""")
        old.execute("INSERT INTO calls (time_utc, model, input_tokens, output_tokens, total_tokens, round) "
                    "VALUES ('2026-11-16T00:00:00+00:00', 'old-model', 7, 3, 10, 4)")
        old.commit()
        old.close()
        with TokenMeter(path) as meter:
            meter.measure("qwen3:30b-a3b-q4_K_M", lambda: OLLAMA, prompt="hi")
            rows = meter.rows()
        self.assertEqual([(r["call_id"], r["model"], r["total_tokens"], r["round"]) for r in rows],
                         [(1, "old-model", 10, 4), (2, "qwen3:30b-a3b-q4_K_M", 40, None)])
        self.assertIsNone(rows[1]["cache_read_tokens"])


class Budgets(unittest.TestCase):
    def setUp(self):
        self.meter = TokenMeter(":memory:")
        self.addCleanup(self.meter.close)

    def test_budget_counts_down_then_refuses_without_calling_the_model(self):
        budget = self.meter.budget("run-1/repair", 100)
        self.assertEqual(budget.max_output(8192), 100)
        self.meter.record("m", Usage(40, 20), budgets=[budget])
        self.assertEqual((budget.used(), budget.remaining(), budget.max_output(8192), budget.max_output(10)),
                         (60, 40, 40, 10))
        self.meter.record("m", Usage(40, 20), budgets=[budget])  # overshoots: 120 of 100
        self.assertTrue(budget.exhausted)

        called = []
        with self.assertRaises(BudgetExceeded) as caught:
            self.meter.measure("m", lambda: called.append(1), prompt="one more", budgets=[budget])
        self.assertEqual(called, [])
        self.assertIn("run-1/repair (used 120 of 100 tokens)", str(caught.exception))
        rows = self.meter.rows()
        self.assertEqual([r["flags"] for r in rows], ["", "over_budget", "budget_refused"])
        self.assertEqual((rows[2]["total_tokens"], rows[2]["usage_source"]), (0, "none"))
        self.assertEqual(budget.used(), 120)  # the refusal itself costs nothing

    def test_agent_and_total_budgets_are_both_charged_and_evaluation_is_separate(self):
        total, blue = self.meter.budget("run/total", 1000), self.meter.budget("run/blue", 300)
        self.meter.record("m", Usage(100, 50), budgets=[blue, total], step="patch")
        self.meter.record("m", Usage(100, 50), budgets=[total], step="red")
        self.meter.record("m", Usage(500, 500), step="final-evaluation")  # separate allowance: not charged
        self.assertEqual((blue.used(), total.used()), (150, 300))
        self.assertEqual(tokenmeter.output_allowance([blue, total], 8192), 150)
        self.assertEqual(tokenmeter.output_allowance([], 8192), 8192)
        listed = {b["name"]: b for b in self.meter.budgets()}
        self.assertEqual((listed["run/blue"]["remaining_tokens"], listed["run/total"]["calls"]), (150, 2))

    def test_budget_persists_and_its_limit_cannot_be_changed(self):
        path = os.path.join(tempfile.mkdtemp(), "ledger.db")
        with TokenMeter(path) as first:
            first.record("m", Usage(30, 30), budgets=[first.budget("run-9/repair", 120_000)])
        with TokenMeter(path) as second:
            self.assertEqual(second.budget("run-9/repair").used(), 60)
            self.assertEqual(second.budget("run-9/repair", 120_000).remaining(), 119_940)
            with self.assertRaises(ValueError):
                second.budget("run-9/repair", 500_000)
            with self.assertRaises(ValueError):
                second.budget("never-created")

    def test_config_records_are_kept_in_order(self):
        self.meter.save_config("qwen", {"digest": "abc", "temperature": 0.2})
        self.meter.save_config("claude", {"seed": "not configurable"})
        self.assertEqual([c["name"] for c in self.meter.configs()], ["qwen", "claude"])
        self.assertEqual(self.meter.configs("qwen")[0]["config"]["digest"], "abc")


class FakeServices(unittest.TestCase):
    """A stand-in Ollama / OpenAI-style server on this computer, real HTTP."""

    @classmethod
    def setUpClass(cls):
        seen = cls.seen = []
        tag = "qwen3:30b-a3b-q4_K_M"
        answers = {
            "/api/version": {"version": "0.99.0"},
            "/api/tags": {"models": [{"name": tag, "model": tag, "digest": "0b28110b7a33", "size": 19_000}]},
            "/api/ps": {"models": [{"name": tag, "model": tag, "size": 20_000, "size_vram": 15_000}]},
            "/api/show": {"details": {"quantization_level": "Q4_K_M", "parameter_size": "30.5B",
                                      "family": "qwen3moe", "format": "gguf"}},
            "/api/chat": OLLAMA,
            "/v1/chat/completions": OPENAI_CHAT,
        }

        class Handler(BaseHTTPRequestHandler):
            def reply(self):
                body = None
                if self.command == "POST":
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("Authorization"), body))
                if self.path not in answers:
                    self.send_response(429)
                    self.end_headers()
                    self.wfile.write(b'{"error": "slow down"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(answers[self.path]).encode())

            do_GET = do_POST = reply

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

    def last_chat_body(self):
        return [body for path, _, body in self.seen if path == "/api/chat"][-1]

    def test_ollama_chat_sends_thinking_and_options(self):
        tokenmeter.ollama_chat("qwen3:30b-a3b-q4_K_M", "hi", system="be brief", host=self.url, think=True,
                               temperature=0.6, num_ctx=16384, seed=42)
        body = self.last_chat_body()
        self.assertEqual((body["think"], body["stream"], len(body["messages"])), (True, False, 2))
        self.assertEqual(body["options"], {"temperature": 0.6, "num_ctx": 16384, "seed": 42})
        tokenmeter.ollama_chat("qwen2.5:14b-instruct-q4_K_M", "hi", host=self.url)
        self.assertNotIn("think", self.last_chat_body())

    def test_ollama_model_info_reads_version_hash_quantization_and_gpu_share(self):
        info = tokenmeter.ollama_model_info("qwen3:30b-a3b-q4_K_M", self.url)
        self.assertEqual((info["ollama_version"], info["digest"], info["quantization_level"], info["gpu_percent"]),
                         ("0.99.0", "0b28110b7a33", "Q4_K_M", 75.0))
        missing = tokenmeter.ollama_model_info("not-pulled:1b", self.url)
        self.assertEqual(missing["digest"], "unavailable")
        offline = tokenmeter.ollama_model_info("x", "http://127.0.0.1:9")
        self.assertEqual(offline["ollama_version"], "unavailable")

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

    # ---- the planned configurations, local path

    def test_planned_qwen3_request_uses_the_chapter_3_settings(self):
        budget = tokenmeter.start_run(self.meter, "qwen3-30b-a3b", run="sqli-01/loop/seed-42", host=self.url)
        self.assertEqual((budget.name, budget.limit), ("sqli-01/loop/seed-42/repair", 120_000))
        reply = tokenmeter.ask(self.meter, "qwen3-30b-a3b", "fix it", system="you repair code", seed=42,
                         budgets=[budget], host=self.url, task="SQLI-01", round=1, step="patch")
        self.assertEqual(reply.text, "hello")
        body = self.last_chat_body()
        self.assertEqual(body["model"], "qwen3:30b-a3b-q4_K_M")
        self.assertIs(body["think"], True)
        self.assertEqual(body["options"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                                           "num_ctx": 16384, "num_predict": 8192, "seed": 42})
        (row,) = self.meter.rows()
        tags = json.loads(row["tags"])
        self.assertEqual((row["provider"], row["task"], row["round"], row["flags"]), ("ollama", "SQLI-01", 1, ""))
        self.assertEqual((tags["config"], tags["seed"], tags["gpu_percent"], tags["thinking"]),
                         ("qwen3-30b-a3b", 42, 75.0, True))
        self.assertEqual(budget.used(), 40)
        (saved,) = self.meter.configs("qwen3-30b-a3b")
        self.assertEqual((saved["config"]["digest"], saved["config"]["quantization_matches"],
                          saved["config"]["run"]), ("0b28110b7a33", True, "sqli-01/loop/seed-42"))

    def test_planned_qwen_output_limit_shrinks_with_the_budget_and_14b_has_no_thinking(self):
        budget = self.meter.budget("r/repair", 500)
        self.meter.record("x", Usage(200, 100), budgets=[budget])
        tokenmeter.ask(self.meter, "qwen2.5-coder-14b-instruct", "fix it", seed=43, budgets=[budget], host=self.url)
        body = self.last_chat_body()
        self.assertEqual(body["model"], "qwen2.5-coder:14b-instruct-q4_K_M")
        self.assertNotIn("think", body)
        self.assertEqual(body["options"], {"temperature": 0.2, "top_p": 0.9, "top_k": 40,
                                           "num_ctx": 16384, "num_predict": 200, "seed": 43})
        # the stand-in server answers as a different model, which must be noticed
        self.assertEqual(self.meter.rows()[-1]["flags"], "model_mismatch")


class PlannedCloudClients(unittest.TestCase):
    """Codex and Claude Code, with the command-line tool replaced by a recording."""

    def setUp(self):
        self.meter = TokenMeter(":memory:")
        self.addCleanup(self.meter.close)

    def test_every_planned_configuration_is_present(self):
        self.assertEqual(set(tokenmeter.MODELS), {
            "gpt-6-astra", "gpt-6-luna", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
            "qwen2.5-14b-instruct", "qwen2.5-coder-14b-instruct", "qwen3-30b-a3b"})
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(tokenmeter.main(["list"]), 0)
        self.assertIn("qwen3:30b-a3b-q4_K_M", out.getvalue())

    def test_claude_code_request(self):
        with mock.patch.object(tokenmeter, "run_cli", return_value=json.dumps(CLAUDE_CODE)) as run:
            reply = tokenmeter.ask(self.meter, "claude-opus-5-5", "fix it", system="you repair code",
                             workdir="/work", allow_edits=True, task="SQLI-01", round=2)
        command, sent = run.call_args.args
        self.assertEqual(command[:2], ["claude", "-p"])
        self.assertEqual(command[command.index("--model") + 1], "claude-opus-5-5")
        self.assertEqual(command[command.index("--effort") + 1], "xhigh")
        self.assertEqual(command[command.index("--permission-mode") + 1], "acceptEdits")
        self.assertEqual((sent, run.call_args.kwargs["cwd"]), ("you repair code\n\nfix it", "/work"))
        self.assertNotIn("fix it", command)  # the prompt travels on standard input
        self.assertEqual(reply.text, "ready")
        (row,) = self.meter.rows()
        tags = json.loads(row["tags"])
        self.assertEqual((row["model"], row["reported_model"], row["total_tokens"], row["cost_usd"], row["flags"]),
                         ("claude-opus-5-5", "claude-opus-5-5", 300, 0.0123, ""))
        self.assertEqual((tags["effort"], tags["turns"], row["round"]), ("xhigh", 2, 2))

    def test_claude_code_error_result_still_records_its_tokens(self):
        failed = {**CLAUDE_CODE, "is_error": True, "result": "usage limit reached"}
        with mock.patch.object(tokenmeter, "run_cli", return_value=json.dumps(failed)):
            with self.assertRaises(RuntimeError):
                tokenmeter.ask(self.meter, "claude-haiku-5-5", "fix it")
        self.assertEqual(self.meter.rows()[0]["total_tokens"], 300)

    def test_codex_request(self):
        with mock.patch.object(tokenmeter, "run_cli", return_value=CODEX) as run:
            reply = tokenmeter.ask(self.meter, "gpt-6-astra", "fix it")
        command, sent = run.call_args.args
        self.assertEqual(command[:3], ["codex", "exec", "--json"])
        self.assertIn("model_reasoning_effort=xhigh", command)
        self.assertEqual((command[command.index("--sandbox") + 1], command[-1], sent), ("read-only", "-", "fix it"))
        self.assertEqual(reply.text, "done")
        (row,) = self.meter.rows()
        self.assertEqual((row["input_tokens"], row["cache_read_tokens"], row["reasoning_tokens"],
                          row["response_chars"], row["reported_model"]), (150, 60, 20, 4, None))

    def test_cloud_configuration_marks_what_cannot_be_set(self):
        with mock.patch.object(tokenmeter, "cli_version", return_value="9.9.9 (Claude Code)"):
            record = tokenmeter.describe("claude-sonnet-5-5")
            self.assertIsNone(tokenmeter.start_run(self.meter, "claude-sonnet-5-5", run="r1"))
        self.assertEqual((record["client_version"], record["reasoning_effort"]), ("9.9.9 (Claude Code)", "xhigh"))
        for control in ("temperature", "top_p", "top_k", "seed"):
            self.assertEqual(record[control], tokenmeter.NOT_CONFIGURABLE)
        self.assertIn("recorded only", record["token_limit_enforcement"])


class StandInServer(unittest.TestCase):
    """examples/local_test_server.py, driven through the planned-model commands."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("local_test_server",
                                                      os.path.join(here, "examples", "local_test_server.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
        cls.server.RequestHandlerClass.log_message = lambda *args: None
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.ledger = os.path.join(tempfile.mkdtemp(), "test_ledger.db")

    def check(self, key, host=None):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            code = tokenmeter.main(["check", key, "--host", host or self.url, "--ledger", self.ledger])
        return code, out.getvalue()

    def test_every_planned_qwen_configuration_passes_its_check(self):
        for key in ("qwen2.5-14b-instruct", "qwen2.5-coder-14b-instruct", "qwen3-30b-a3b"):
            with self.subTest(key):
                code, printed = self.check(key)
                self.assertEqual(code, 0, printed)
                self.assertIn("reply: ready", printed)
        with TokenMeter(self.ledger) as meter:
            rows = meter.rows()
            self.assertEqual([r["usage_source"] for r in rows], ["reported"] * 3)
            self.assertEqual([r["flags"] for r in rows], ["", "", ""])
            self.assertIsNone(rows[0]["reasoning_chars"])  # Qwen2.5 does not think
            self.assertGreater(rows[2]["reasoning_chars"], 0)  # Qwen3 does
            saved = meter.configs("qwen3-30b-a3b")[0]["config"]
        self.assertEqual((saved["quantization_matches"], saved["gpu_percent"]), (True, 100.0))

    def test_the_stand_in_refuses_thinking_for_a_model_without_it_like_ollama(self):
        with self.assertRaises(RuntimeError) as refused:
            tokenmeter.ollama_chat("qwen2.5:14b-instruct-q4_K_M", "hi", host=self.url, think=True)
        self.assertIn("does not support thinking", str(refused.exception))

    def test_output_limit_is_honoured_and_flagged(self):
        with TokenMeter(":memory:") as meter:
            meter.measure("qwen3:30b-a3b-q4_K_M", lambda: tokenmeter.ollama_chat(
                "qwen3:30b-a3b-q4_K_M", "explain " * 50, host=self.url, think=True, num_predict=5),
                limits={"max_output_tokens": 5})
            (row,) = meter.rows()
        self.assertEqual((row["output_tokens"], row["stop_reason"]), (5, "length"))
        self.assertEqual(row["flags"], "output_truncated,output_at_limit")

    def run_main(self, *args):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            code = tokenmeter.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_ask_command_counts_tokens_and_applies_the_run_budget(self):
        common = ("--host", self.url, "--ledger", self.ledger, "--run", "r1", "--task", "SQLI-01", "--seed", "42")
        code, out, err = self.run_main("ask", "qwen3-30b-a3b", "Fix the login query.", *common,
                                       "--round", "1", "--step", "patch")
        self.assertEqual(code, 0, err)
        self.assertIn("[stand-in reply]", out)
        self.assertIn("[tokenmeter] qwen3-30b-a3b: input", err)
        self.assertIn("budget left 119,9", err)
        code, _, _ = self.run_main("ask", "qwen3-30b-a3b", "Try again.", *common, "--round", "2", "--step", "patch")
        self.assertEqual(code, 0)
        with TokenMeter(self.ledger) as meter:
            rows = meter.rows()
            self.assertEqual([(r["run"], r["task"], r["round"], r["step"]) for r in rows],
                             [("r1", "SQLI-01", 1, "patch"), ("r1", "SQLI-01", 2, "patch")])
            self.assertEqual(len(meter.configs("qwen3-30b-a3b")), 1)  # the run is set up only once
            self.assertEqual(meter.budget("r1/repair").used(), sum(r["total_tokens"] for r in rows))

    def test_ask_command_stops_when_the_run_budget_is_spent(self):
        common = ("qwen2.5-14b-instruct", "hello", "--host", self.url, "--ledger", self.ledger, "--run", "tiny")
        with mock.patch.object(tokenmeter, "LOCAL_RUN_TOKEN_BUDGET", 10):
            first = self.run_main("ask", *common)
            second = self.run_main("ask", *common)
        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], 2)
        self.assertIn("Not sent: budget exhausted: tiny/repair", second[2])

    def test_ask_command_reads_files_and_writes_the_reply_to_a_file(self):
        folder = os.path.dirname(self.ledger)
        prompt, system, reply = (os.path.join(folder, name) for name in ("p.txt", "s.txt", "reply.txt"))
        with open(prompt, "w", encoding="utf-8") as handle:
            handle.write("Reply with the single word: ready")
        with open(system, "w", encoding="utf-8") as handle:
            handle.write("You repair code.")
        code, out, err = self.run_main("ask", "qwen2.5-coder-14b-instruct", "--prompt-file", prompt,
                                       "--system-file", system, "--out", reply, "--host", self.url,
                                       "--ledger", self.ledger)
        self.assertEqual((code, out), (0, ""))
        with open(reply, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "ready")
        with TokenMeter(self.ledger) as meter:
            (row,) = meter.rows()
        self.assertEqual(row["prompt_chars"], len(json.dumps(["You repair code.", "Reply with the single word: ready"])))

    def test_a_model_added_to_the_settings_works_in_every_command(self):
        added = {"my-qwen": {"client": "ollama", "model": "qwen2.5:14b-instruct-q4_K_M", "think": None,
                             "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1}}}
        with mock.patch.dict(tokenmeter.MODELS, added):
            code, listed, _ = self.run_main("list")
            self.assertIn("my-qwen", listed)
            code, printed = self.check("my-qwen")
        self.assertEqual(code, 0, printed)
        with TokenMeter(self.ledger) as meter:
            self.assertEqual(json.loads(meter.rows()[0]["tags"])["config"], "my-qwen")

    def test_unreachable_server_gives_a_clear_error_and_exit_code(self):
        code, printed = self.check("qwen3-30b-a3b", host="http://127.0.0.1:9")
        self.assertEqual(code, 1)
        self.assertIn("cannot reach", printed)
        self.assertIn("Is the model server running", printed)


class CommandLineTools(unittest.TestCase):
    def test_prompt_goes_through_standard_input_at_any_length(self):
        long_prompt = "ก" * 50_000 + " end"  # far past the Windows command-line limit, and not ASCII
        echo = "import sys; sys.stdin.reconfigure(encoding='utf-8'); sys.stdout.reconfigure(encoding='utf-8'); " \
               "print(len(sys.stdin.read()))"
        self.assertEqual(tokenmeter.run_cli([sys.executable, "-c", echo], long_prompt).strip(), "50004")

    def test_failures_are_reported(self):
        with self.assertRaises(RuntimeError) as failed:
            tokenmeter.run_cli([sys.executable, "-c", "import sys; sys.exit('no access')"], "hi")
        self.assertIn("no access", str(failed.exception))
        with self.assertRaises(RuntimeError):
            tokenmeter.run_cli(["definitely-not-installed-tool"], "hi")
        self.assertEqual(tokenmeter.cli_version("definitely-not-installed-tool"), "unavailable")
        self.assertIn("Python", tokenmeter.cli_version(sys.executable))


if __name__ == "__main__":
    unittest.main()
