import asyncio
import sys
from contextlib import asynccontextmanager
import threading
import types
import unittest
from unittest.mock import patch

from api.main import lifespan
from core.database import DatabaseSettings
from core.embeddings import SentenceTransformerEmbeddingProvider
from core.vector_store import VectorStore
from mcp.knowledge_base import KnowledgeBase


class FakeEmbeddingProvider:
    dimension = 3

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeConnection:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.fetches = []
        self.writes = []
        self.profiles = {}

    async def fetch(self, sql, *args):
        self.fetches.append((" ".join(sql.split()), args))
        return self.rows

    async def execute(self, sql, *args):
        self.writes.append((" ".join(sql.split()), args))
        if "INSERT INTO user_profiles" in sql:
            self.profiles[args[0]] = args[1]

    async def fetchrow(self, sql, *args):
        self.fetches.append((" ".join(sql.split()), args))
        profile = self.profiles.get(args[0])
        return None if profile is None else {"profile": profile}

    async def fetchval(self, sql, *args):
        self.fetches.append((" ".join(sql.split()), args))
        return self.count


class FakeDatabase:
    def __init__(self, connection):
        self.connection = connection
        self.transactions = 0

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        yield self.connection


class VectorStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = FakeEmbeddingProvider()
        self.conn = FakeConnection()
        self.db = FakeDatabase(self.conn)
        self.store = VectorStore(self.db, self.provider)

    async def test_knowledge_search_uses_cosine_and_maps_response(self):
        self.conn.rows = [{"content": "住宿上限", "metadata": {
            "title": "差旅", "chunk_index": 2}, "distance": 0.123456}]
        self.assertEqual(await self.store.search_knowledge("住宿'", 3), [{
            "title": "差旅", "content": "住宿上限", "score": 0.8765, "chunk": 2}])
        sql, args = self.conn.fetches[0]
        self.assertIn("embedding <=> $1 AS distance", sql)
        self.assertIn("ORDER BY embedding <=> $1", sql)
        self.assertIn("LIMIT $2", sql)
        self.assertNotIn("住宿'", sql)
        self.assertEqual(args, ([3.0, 0.0, 1.0], 3))

    async def test_knowledge_count_reads_database(self):
        self.conn.count = 7
        self.assertEqual(await self.store.count_knowledge(), 7)
        sql, args = self.conn.fetches[-1]
        self.assertIn("COUNT(*)", sql)
        self.assertEqual(args, ())

    async def test_episodic_search_scopes_owner_and_returns_summaries(self):
        self.conn.rows = [{"summary": "旧报销记录", "distance": 0.1}]
        self.assertEqual(await self.store.search_episodic("emp-'1", "报销", 3),
                         ["旧报销记录"])
        sql, args = self.conn.fetches[0]
        self.assertIn("WHERE user_id = $1", sql)
        self.assertIn("ORDER BY embedding <=> $2", sql)
        self.assertIn("LIMIT $3", sql)
        self.assertNotIn("emp-'1", sql)
        self.assertEqual(args, ("emp-'1", [2.0, 0.0, 1.0], 3))

    async def test_empty_search_and_zero_limit_skip_embedding_and_database(self):
        for query, limit in [("", 5), ("  \n", 5), ("制度", 0), ("制度", -1)]:
            self.assertEqual(await self.store.search_knowledge(query, limit), [])
            self.assertEqual(await self.store.search_episodic("u", query, limit), [])
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.db.transactions, 0)

    async def test_add_knowledge_embeds_each_chunk_and_preserves_metadata(self):
        count = await self.store.add_knowledge([
            {"id": "k-1", "title": "差旅制度", "content": "住宿上限",
             "metadata": {"chunk": 0, "source": "制度.pdf"}},
            {"id": "k-2", "title": "差旅制度", "content": "交通",
             "metadata": {"chunk_index": 1}},
        ])
        self.assertEqual(count, 2)
        self.assertEqual(self.provider.calls, [["住宿上限", "交通"]])
        self.assertEqual(self.db.transactions, 1)
        sql, args = self.conn.writes[0]
        self.assertIn("INSERT INTO knowledge_chunks", sql)
        self.assertIn("$5", sql)
        self.assertEqual(args, ("k-1", "差旅制度", "住宿上限", {
            "title": "差旅制度", "chunk": 0, "chunk_index": 0, "source": "制度.pdf"},
            [4.0, 0.0, 1.0]))
        self.assertEqual(self.conn.writes[1][1][-1], [2.0, 0.0, 1.0])

    async def test_empty_knowledge_batch_does_no_work(self):
        self.assertEqual(await self.store.add_knowledge([]), 0)
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.db.transactions, 0)

    async def test_episodic_write_embeds_summary_and_keeps_full_text(self):
        await self.store.store_episodic("u", "c", "完整对话", "摘要")
        self.assertEqual(self.provider.calls, [["摘要"]])
        sql, args = self.conn.writes[0]
        self.assertIn("INSERT INTO episodic_memories", sql)
        self.assertTrue(args[0])
        self.assertEqual(args[1:], ("u", "c", "摘要", "完整对话", [2.0, 0.0, 1.0]))

    async def test_profile_upsert_replaces_instead_of_merging(self):
        await self.store.upsert_profile("u", {"department": "IT", "old": True})
        await self.store.upsert_profile("u", {"department": "财务"})
        self.assertEqual(await self.store.get_profile("u"), {"department": "财务"})
        sql, args = self.conn.writes[-1]
        self.assertIn("ON CONFLICT (user_id) DO UPDATE", sql)
        self.assertIn("profile = EXCLUDED.profile", sql)
        self.assertNotIn("||", sql)
        self.assertEqual(args, ("u", {"department": "财务"}))
        self.assertEqual(self.provider.calls, [])

    async def test_missing_profile_returns_empty_dict(self):
        self.assertEqual(await self.store.get_profile("missing'"), {})
        sql, args = self.conn.fetches[0]
        self.assertIn("WHERE user_id = $1", sql)
        self.assertEqual(args, ("missing'",))

    async def test_bad_embedding_batch_is_rejected_before_write(self):
        async def malformed(texts):
            return [[1.0, 2.0]]
        self.provider.embed = malformed
        with self.assertRaises(ValueError):
            await self.store.add_knowledge([{"id": "k", "content": "内容"}])
        self.assertEqual(self.conn.writes, [])


class EmbeddingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_load_is_lazy_and_encoding_runs_off_event_loop(self):
        main_thread = threading.get_ident()
        threads = []
        constructions = []
        class FakeModel:
            def __init__(self, name):
                constructions.append(name)
                threads.append(threading.get_ident())

            def encode(self, texts, **kwargs):
                threads.append(threading.get_ident())
                return [[1] * 512 for _ in texts]

        with patch.dict("sys.modules", {"sentence_transformers": types.SimpleNamespace(
                SentenceTransformer=FakeModel)}):
            provider = SentenceTransformerEmbeddingProvider()
            self.assertEqual(constructions, [])
            self.assertEqual(await provider.embed([]), [])
            self.assertEqual(constructions, [])
            first, second = await asyncio.gather(provider.embed(["甲"]), provider.embed(["乙"]))
        self.assertEqual(constructions, ["BAAI/bge-small-zh-v1.5"])
        self.assertTrue(all(t != main_thread for t in threads))
        self.assertEqual(first, [[1.0] * 512])
        self.assertEqual(second, [[1.0] * 512])
        self.assertIs(type(first[0][0]), float)

    async def test_runtime_dimension_cannot_diverge_from_schema(self):
        with self.assertRaises(ValueError):
            SentenceTransformerEmbeddingProvider(DatabaseSettings(embedding_dimension=3))

    async def test_wrong_model_dimension_is_rejected(self):
        class FakeModel:
            def __init__(self, name):
                pass

            def encode(self, texts, **kwargs):
                return [[1, 2, 3]]

        with patch.dict("sys.modules", {"sentence_transformers": types.SimpleNamespace(
                SentenceTransformer=FakeModel)}):
            with self.assertRaises(ValueError):
                await SentenceTransformerEmbeddingProvider().embed(["甲"])


class FakeKnowledgeStore:
    def __init__(self, count=0):
        self.count = count
        self.saved = []

    async def count_knowledge(self):
        return self.count

    async def add_knowledge(self, chunks):
        self.saved.extend(chunks)
        self.count += len(chunks)
        return len(chunks)

    async def search_knowledge(self, query, top_k):
        return [{"title": "差旅制度", "content": "住宿上限 500 元", "score": 0.8765, "chunk": 2}]


class KnowledgeBaseCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_handler_preserves_chunk_result_shape(self):
        store = FakeKnowledgeStore()
        kb = KnowledgeBase(store)
        self.assertEqual(await kb.search_handler({"query": "住宿", "top_k": 3}, None), [{
            "title": "差旅制度", "content": "住宿上限 500 元", "score": 0.8765, "chunk": 2,
        }])

    async def test_default_documents_seed_once_when_storage_is_empty(self):
        store = FakeKnowledgeStore()
        kb = KnowledgeBase(store)
        self.assertEqual(store.saved, [])
        await kb.ensure_default_documents()
        first_count = store.count
        self.assertGreater(first_count, 0)
        await kb.ensure_default_documents()
        self.assertEqual(store.count, first_count)

        populated = FakeKnowledgeStore(count=2)
        await KnowledgeBase(populated).ensure_default_documents()
        self.assertEqual(populated.saved, [])


class UniqueKnowledgeConnection(FakeConnection):
    def __init__(self):
        super().__init__()
        self.knowledge = {}
        self.count_calls = 0
        self.both_counted = asyncio.Event()

    async def fetchval(self, sql, *args):
        observed_count = len(self.knowledge)
        self.count_calls += 1
        if self.count_calls <= 2:
            if self.count_calls == 2:
                self.both_counted.set()
            await self.both_counted.wait()
        return observed_count

    async def execute(self, sql, *args):
        if "INSERT INTO knowledge_chunks" not in sql:
            return await super().execute(sql, *args)
        if args[0] in self.knowledge and "ON CONFLICT" not in sql:
            raise ValueError("duplicate primary key")
        self.knowledge[args[0]] = args[1:]


class KnowledgeSeedConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_empty_observations_seed_without_duplicate_key_failure(self):
        connection = UniqueKnowledgeConnection()
        store = VectorStore(FakeDatabase(connection), FakeEmbeddingProvider())
        await asyncio.gather(
            KnowledgeBase(store).ensure_default_documents(),
            KnowledgeBase(store).ensure_default_documents(),
        )
        self.assertEqual(len(connection.knowledge), 6)

    async def test_reimport_updates_existing_chunk_content(self):
        connection = UniqueKnowledgeConnection()
        connection.count_calls = 2
        store = VectorStore(FakeDatabase(connection), FakeEmbeddingProvider())
        kb = KnowledgeBase(store)
        self.assertEqual(await kb.add_documents([{"title": "制度", "content": "A" * 60}]), 1)
        self.assertEqual(await kb.add_documents([{"title": "制度", "content": "A" * 50 + "B" * 10}]), 1)
        self.assertEqual(len(connection.knowledge), 1)
        self.assertEqual(next(iter(connection.knowledge.values()))[1], "A" * 50 + "B" * 10)


class LifespanCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_closes_for_each_startup_failure_and_only_started_monitor_stops(self):
        for failure_stage in ("provider", "vector", "seed", "monitor_start", "evaluator"):
            with self.subTest(failure_stage=failure_stage):
                databases = []
                monitors = []

                class FakeDatabase:
                    def __init__(self):
                        self.settings = object()
                        self.connected = False
                        self.closed = False
                        databases.append(self)

                    async def connect(self):
                        self.connected = True

                    async def close(self):
                        self.closed = True

                class Noop:
                    def __init__(self, *args, **kwargs):
                        pass

                    def register(self, tool):
                        pass

                class FakeProvider(Noop):
                    def __init__(self, *args, **kwargs):
                        if failure_stage == "provider":
                            raise RuntimeError("provider failed")

                class FakeVector(Noop):
                    def __init__(self, *args, **kwargs):
                        if failure_stage == "vector":
                            raise RuntimeError("vector failed")

                class FakeKnowledgeBase(Noop):
                    async def ensure_default_documents(self):
                        if failure_stage == "seed":
                            raise RuntimeError("seed failed")

                    async def doc_count(self):
                        return 1

                    async def search_handler(self, params, context):
                        return []

                class FakeMonitor(Noop):
                    def __init__(self, *args, **kwargs):
                        self.started = False
                        self.stopped = False
                        monitors.append(self)

                    async def start(self):
                        if failure_stage == "monitor_start":
                            raise RuntimeError("monitor failed")
                        self.started = True

                    async def stop(self):
                        self.stopped = True

                class FakeEvaluator(Noop):
                    def __init__(self, *args, **kwargs):
                        if failure_stage == "evaluator":
                            raise RuntimeError("evaluator failed")

                modules = {
                    "agents.agent_orchestrator": types.SimpleNamespace(AgentOrchestrator=Noop, Request=Noop),
                    "core.database": types.SimpleNamespace(Database=FakeDatabase),
                    "core.embeddings": types.SimpleNamespace(SentenceTransformerEmbeddingProvider=FakeProvider),
                    "core.intent_recognizer": types.SimpleNamespace(IntentRecognizer=Noop),
                    "core.vector_store": types.SimpleNamespace(VectorStore=FakeVector),
                    "evaluation.evaluator": types.SimpleNamespace(EndToEndEvaluator=FakeEvaluator),
                    "mcp.knowledge_base": types.SimpleNamespace(KnowledgeBase=FakeKnowledgeBase),
                    "mcp.tool_manager": types.SimpleNamespace(MCPToolManager=Noop, Tool=Noop),
                    "memory.conversation_memory": types.SimpleNamespace(MemoryManager=Noop),
                    "monitor.performance_monitor": types.SimpleNamespace(PerformanceMonitor=FakeMonitor),
                    "tools.loader": types.SimpleNamespace(discover_agent_tools=lambda *a, **k: []),
                    "core.reimbursement_repository": types.SimpleNamespace(InMemoryReimbursementRepository=Noop),
                    "core.reimbursement_harness": types.SimpleNamespace(ReimbursementCaseHarness=Noop),
                }
                with patch.dict(sys.modules, modules), patch("api.main.BANNER", "banner"), \
                     patch("api.main._anthropic_cfg", return_value={
                         "api_key": "fake", "model": "fake"}):
                    with self.assertRaises(RuntimeError):
                        async with lifespan(None):
                            pass

                self.assertTrue(databases[0].connected)
                self.assertTrue(databases[0].closed)
                if monitors:
                    self.assertEqual(monitors[0].stopped, monitors[0].started)
