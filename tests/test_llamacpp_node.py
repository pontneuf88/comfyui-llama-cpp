from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module():
    module_name = "llamacpp_node_under_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "LlamaCppNode.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LlamaCppNodeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.meta = {
            "connectivity": {
                "url": "http://localhost:8080",
                "model": "test-model",
                "request_timeout": 30,
            },
            "options": None,
        }

    def test_generate_caches_latest_response_after_forced_resend(self):
        node = self.module.LlamaCppGenerate()
        responses = [
            ("first", None, {"choices": []}),
            ("second", None, {"choices": []}),
        ]

        with mock.patch.object(self.module, "_run_chat", side_effect=responses) as run_chat:
            first = node.llamacpp_generate(
                "system",
                "prompt",
                False,
                False,
                "text",
                64,
                meta=self.meta,
            )
            forced = node.llamacpp_generate(
                "system",
                "prompt",
                False,
                False,
                "text",
                64,
                meta=self.meta,
                send_on_change_only=False,
            )
            cached = node.llamacpp_generate(
                "system",
                "prompt",
                False,
                False,
                "text",
                64,
                meta=self.meta,
            )

        self.assertEqual(first[0], "first")
        self.assertEqual(forced[0], "second")
        self.assertEqual(cached[0], "second")
        self.assertEqual(run_chat.call_count, 2)

    def test_chat_caches_latest_response_after_forced_resend(self):
        responses = [
            ("first", None, {"choices": []}),
            ("second", None, {"choices": []}),
        ]

        with mock.patch.object(self.module, "_run_chat", side_effect=responses) as run_chat:
            first = self.module.LlamaCppChat().llamacpp_chat(
                "system",
                "prompt",
                False,
                "node-1",
                "text",
                64,
                meta=self.meta,
            )
            forced = self.module.LlamaCppChat().llamacpp_chat(
                "system",
                "prompt",
                False,
                "node-1",
                "text",
                64,
                meta=self.meta,
                send_on_change_only=False,
            )
            cached = self.module.LlamaCppChat().llamacpp_chat(
                "system",
                "prompt",
                False,
                "node-1",
                "text",
                64,
                meta=self.meta,
            )

        self.assertEqual(first[0], "first")
        self.assertEqual(forced[0], "second")
        self.assertEqual(cached[0], "second")
        self.assertEqual(run_chat.call_count, 2)

    def test_generate_uses_input_fingerprint_for_cache_hits(self):
        node = self.module.LlamaCppGenerate()

        with mock.patch.object(self.module, "_run_chat", return_value=("same", None, {"choices": []})) as run_chat:
            node.llamacpp_generate("system", "prompt", False, False, "text", 64, meta=self.meta)
            cached = node.llamacpp_generate("system", "prompt", False, False, "text", 64, meta=self.meta)

        self.assertEqual(cached[0], "same")
        self.assertEqual(run_chat.call_count, 1)

    def test_generate_is_changed_respects_send_on_change_only(self):
        first = self.module.LlamaCppGenerate.IS_CHANGED(
            "system",
            "prompt",
            False,
            False,
            "text",
            64,
            meta=self.meta,
        )
        second = self.module.LlamaCppGenerate.IS_CHANGED(
            "system",
            "prompt",
            False,
            False,
            "text",
            64,
            meta=self.meta,
        )
        forced = self.module.LlamaCppGenerate.IS_CHANGED(
            "system",
            "prompt",
            False,
            False,
            "text",
            64,
            meta=self.meta,
            send_on_change_only=False,
        )

        self.assertEqual(first, second)
        self.assertTrue(math.isnan(forced))

    def test_chat_is_changed_respects_send_on_change_only(self):
        first = self.module.LlamaCppChat.IS_CHANGED(
            "system",
            "prompt",
            False,
            "node-1",
            "text",
            64,
            meta=self.meta,
        )
        second = self.module.LlamaCppChat.IS_CHANGED(
            "system",
            "prompt",
            False,
            "node-1",
            "text",
            64,
            meta=self.meta,
        )
        forced = self.module.LlamaCppChat.IS_CHANGED(
            "system",
            "prompt",
            False,
            "node-1",
            "text",
            64,
            meta=self.meta,
            send_on_change_only=False,
        )

        self.assertEqual(first, second)
        self.assertTrue(math.isnan(forced))


if __name__ == "__main__":
    unittest.main()
