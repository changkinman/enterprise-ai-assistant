import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=object))
sys.modules.setdefault("redis", SimpleNamespace())

from memory.conversation_memory import MemoryManager


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        self.values.pop(key, None)

    def lrange(self, key, start, end):
        return self.values.get(key, [])[start:end + 1]


class FakeMemoryStore:
    def __init__(self):
        self.profile = {"department": "财务"}
        self.history = ["上次报销记录"]
        self.saved = []

    async def search_episodic(self, user_id, query, top_k):
        return self.history if user_id == "u1" and query == "报销" else []

    async def get_profile(self, user_id):
        return self.profile if user_id == "u1" else {}

    async def store_episodic(self, user_id, conv_id, text, summary):
        self.saved.append((user_id, conv_id, text, summary))

    async def upsert_profile(self, user_id, profile):
        self.profile = profile


class ClarificationMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_combines_vector_history_profile_and_redis_summary(self):
        redis_client = FakeRedis()
        redis_client.values["summary:u1:c1"] = "本轮摘要"
        store = FakeMemoryStore()
        with patch("memory.conversation_memory.redis.from_url", return_value=redis_client, create=True), \
             patch("memory.conversation_memory.AsyncAnthropic", return_value=SimpleNamespace()):
            memory = MemoryManager(vector_store=store, api_key="fake")

        context = await memory.get_context("u1", "c1", query="报销")
        self.assertEqual(context.relevant_history, ["上次报销记录"])
        self.assertEqual(context.user_profile, {"department": "财务"})
        self.assertEqual(context.summary, "本轮摘要")
        self.assertIn("上次报销记录", context.to_prompt_text())

    async def test_pending_clarification_round_trips_and_can_be_cleared(self):
        memory = MemoryManager.__new__(MemoryManager)
        memory._redis = FakeRedis()
        state = {
            "original_message": "报销系统退回了，怎么办？",
            "clarification_question": "是系统报错还是审核退回？",
        }

        await memory.save_pending_clarification("u1", "c1", state)
        restored = await memory.get_pending_clarification("u1", "c1")

        self.assertEqual(restored, state)
        self.assertEqual(memory._redis.ttls["pending_clarification:u1:c1"], 600)

        await memory.clear_pending_clarification("u1", "c1")
        self.assertIsNone(await memory.get_pending_clarification("u1", "c1"))


if __name__ == "__main__":
    unittest.main()
