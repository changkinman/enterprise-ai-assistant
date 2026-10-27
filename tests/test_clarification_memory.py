import json
import sys
import unittest
from types import SimpleNamespace


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=object))
sys.modules.setdefault("chromadb", SimpleNamespace())
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


class ClarificationMemoryTests(unittest.IsolatedAsyncioTestCase):
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
