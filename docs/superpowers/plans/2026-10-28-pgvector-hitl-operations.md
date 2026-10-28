# PostgreSQL/pgvector 与通用副作用操作协议 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace ChromaDB with PostgreSQL + pgvector for long-term retrieval data, and deliver a persisted four-stage HITL executor used by reimbursement submission and an IT access-request operation.

**Architecture:** Redis remains the short-lived conversation store. An async PostgreSQL repository layer owns vector retrieval, business records, pending operations, idempotency records and audit events. An explicit EmbeddingProvider generates vectors; SideEffectExecutor owns precheck, preview, confirmation and commit, while domain adapters own deterministic business rules.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, pgvector, sentence-transformers, PyJWT, PostgreSQL with pgvector, Redis, Docker Compose, unittest.IsolatedAsyncioTestCase.

**Spec:** docs/superpowers/specs/2026-10-28-pgvector-hitl-operations-design.md

## Global Constraints

- Keep Redis for session messages, summaries and pending clarifications. Replace only ChromaDB-backed knowledge, episodic memory and profile storage.
- Install every new Python dependency in .venv only. Never install globally.
- Use focused unittest modules. Unit tests must not need Docker Compose, Redis, PostgreSQL, a model download or an LLM API.
- Do not implement external ERP/IAM integrations, a frontend, multi-step approval flows, historical Chroma migration, full JSON Schema validation or distributed exactly-once semantics.
- Final write tools remain behind MCPToolManager.call(), but ordinary Agent tool-use must be denied; only SideEffectExecutor may call them.
- Actor identity comes only from a signed JWT dependency. Do not trust user_id, reviewer_id or finance_user_id from an HTTP body.
- Persist actor, normalized parameters, preview, resource version, snapshot hash, expiry, idempotency response and audit events in PostgreSQL.
- Preview TTL is 600 seconds. Idempotency-Key is required for confirmation and unique by actor_id, action_type and key.
- The required test gate is the six safety cases in the design spec plus affected existing unit tests. Full service startup is optional verification only.

---

## File Structure

| File | Responsibility |
| --- | --- |
| config/postgres/init.sql | pgvector extension plus vector, domain, operation, idempotency and audit schema. |
| core/database.py | asyncpg pool lifecycle, transaction helper and pgvector codec registration. |
| core/embeddings.py | EmbeddingProvider and lazily loaded sentence-transformer provider. |
| core/vector_store.py | PostgreSQL knowledge, episodic-memory and profile repository. |
| core/auth.py | Actor plus JWT creation, parsing and FastAPI dependency. |
| core/operations.py | HITL enums, values and the SideEffectOperation protocol. |
| core/operation_repository.py | PostgreSQL operations/idempotency/audit persistence and test-only memory store. |
| core/side_effect_executor.py | Generic precheck → preview → confirm → commit authority. |
| core/reimbursement_repository.py | PostgreSQL reimbursement records, versions and atomic budget mutation. |
| core/reimbursement_harness.py | Existing fact collection/validation flow; produces reimbursement previews. |
| core/access_request.py | Local deterministic IT access-request operation adapter. |
| memory/conversation_memory.py | Redis short-term memory plus VectorStore long-term memory. |
| mcp/knowledge_base.py | Chunking and retrieval through VectorStore. |
| mcp/tool_manager.py | executor_only Tool metadata and enforcement. |
| tools/reimbursement_tool.py | Executor-only final reimbursement submit handler. |
| api/main.py | Shared construction, JWT-safe operation APIs and pending-operation serialization. |
| docker-compose.yml, Dockerfile, requirements.txt, .env.example | PostgreSQL and embedding runtime configuration; no ChromaDB. |
| tests/test_vector_store.py | Fake-pool vector behavior. |
| tests/test_side_effect_executor.py | HITL safety boundary coverage. |
| tests/test_access_request.py | Second operation adapter coverage. |
| tests/test_operation_api.py | JWT-scoped operation endpoint coverage. |

## Shared Interfaces

~~~python
# core/auth.py
@dataclass(frozen=True)
class Actor:
    id: str
    role: str

async def get_current_actor(...) -> Actor: ...

# core/embeddings.py
class EmbeddingProvider(Protocol):
    dimension: int
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

# core/operations.py
class OperationState(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    params: dict[str, Any]

@dataclass(frozen=True)
class OperationPreview:
    operation_id: str
    action_type: str
    summary: str
    expires_at: datetime
    state: OperationState

class SideEffectOperation(Protocol):
    action_type: str
    async def precheck(self, actor: Actor, params: dict[str, Any]) -> ValidationResult: ...
    async def build_preview(self, actor: Actor, normalized: dict[str, Any],
                            validation: ValidationResult) -> PreviewPayload: ...
    async def commit(self, actor: Actor, normalized: dict[str, Any],
                     idempotency_key: str) -> dict[str, Any]: ...

# core/side_effect_executor.py
class SideEffectExecutor:
    async def prepare(self, actor: Actor, proposal: ActionProposal) -> OperationPreview: ...
    async def get_preview(self, actor: Actor, operation_id: str) -> OperationPreview: ...
    async def confirm(self, actor: Actor, operation_id: str,
                      idempotency_key: str) -> dict[str, Any]: ...
~~~

ValidationResult has allowed, normalized_params, resource_version, snapshot_hash and reason fields. PreviewPayload has a user-visible summary and JSON-serializable details. The executor, never a model boolean, moves operations through pending, confirmed and completed.

### Task 1: Add PostgreSQL/pgvector infrastructure and runtime settings

**Files:**
- Create: config/postgres/init.sql
- Create: core/database.py
- Modify: requirements.txt, docker-compose.yml, Dockerfile, .env.example
- Test: tests/test_database_config.py

**Interfaces:**
- Produces DatabaseSettings.from_env(), Database.connect(), Database.close() and Database.transaction().
- Consumes DATABASE_URL, EMBEDDING_MODEL and EMBEDDING_DIMENSION.

- [ ] **Step 1: Write the failing pure-configuration test.**

~~~python
class DatabaseSettingsTests(unittest.TestCase):
    def test_reads_dsn_and_dimension(self):
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://echo:pass@localhost:5432/echomind",
            "EMBEDDING_DIMENSION": "512",
        }, clear=False):
            settings = DatabaseSettings.from_env()
        self.assertEqual(settings.embedding_dimension, 512)

    def test_rejects_non_positive_dimension(self):
        with patch.dict(os.environ, {"EMBEDDING_DIMENSION": "0"}, clear=False):
            with self.assertRaises(ValueError):
                DatabaseSettings.from_env()
~~~

- [ ] **Step 2: Run it to prove the new module is absent.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_database_config -v

Expected: FAIL because core.database does not exist.

- [ ] **Step 3: Install dependencies inside .venv and implement the pool.**

~~~powershell
.\.venv\Scripts\python.exe -m pip install asyncpg pgvector sentence-transformers PyJWT
~~~

Pin the installed versions in requirements.txt. Implement Database with a lazy asyncpg pool and an init callback calling pgvector.asyncpg.register_vector(connection). Do not create a connection at import time; close() must be safe after failed startup.

Create init.sql with CREATE EXTENSION IF NOT EXISTS vector; and tables knowledge_chunks, episodic_memories, user_profiles, reimbursement_cases, access_requests, operations, idempotency_records and operation_audit_events. Use vector(512), JSONB for structured facts/details, and a uniqueness constraint on actor_id, action_type and idempotency_key.

- [ ] **Step 4: Replace the Compose Chroma service.**

Add a pgvector-enabled PostgreSQL service with postgres-data volume and init.sql mounted into docker-entrypoint-initdb.d. Set the application DATABASE_URL and health dependency. Remove chromadb, CHROMA_* settings, Chroma data volumes and Dockerfile ONNX model download/cache work. Document POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, DATABASE_URL, EMBEDDING_MODEL and EMBEDDING_DIMENSION=512 in .env.example.

- [ ] **Step 5: Verify only configuration and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_database_config -v

Expected: PASS without PostgreSQL.

Run: docker compose config

Expected: postgres service exists and chromadb service does not.

~~~bash
git add requirements.txt docker-compose.yml Dockerfile .env.example config/postgres/init.sql core/database.py tests/test_database_config.py
git commit -m "feat: add postgres pgvector infrastructure"
~~~

### Task 2: Implement explicit embeddings and vector repositories

**Files:**
- Create: core/embeddings.py, core/vector_store.py, tests/test_vector_store.py

**Interfaces:**
- Consumes Database and EmbeddingProvider.
- Produces VectorStore.add_knowledge(), search_knowledge(), store_episodic(), search_episodic(), upsert_profile() and get_profile().

- [ ] **Step 1: Write fake-provider/fake-connection tests.**

~~~python
class FakeEmbeddingProvider:
    dimension = 3
    async def embed(self, texts):
        self.calls.append(texts)
        return [[float(len(text)), 0.0, 1.0] for text in texts]

async def test_episodic_search_scopes_to_owner_and_orders_by_distance(self):
    store, conn = make_store(rows=[{"summary": "旧报销记录", "distance": 0.1}])
    self.assertEqual(await store.search_episodic("emp-001", "报销", 3), ["旧报销记录"])
    self.assertIn("user_id", conn.last_fetch_args)
    self.assertIn("<=>", conn.last_fetch_sql)
~~~

- [ ] **Step 2: Run the module before implementation.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_vector_store -v

Expected: FAIL because core.embeddings and core.vector_store do not exist.

- [ ] **Step 3: Implement provider and parameterized SQL.**

SentenceTransformerEmbeddingProvider loads EMBEDDING_MODEL only at first embed call and calls the synchronous encoder through asyncio.to_thread. It normalizes output to list[float].

VectorStore must use parameterized SQL. Knowledge search uses:

~~~sql
SELECT content, metadata, embedding <=> $1 AS distance
FROM knowledge_chunks
ORDER BY embedding <=> $1
LIMIT $2;
~~~

Episodic search adds WHERE user_id = $1. Knowledge result maps to title, content, score and chunk, while episodic search returns summaries. Profile upsert replaces a user profile by user_id and does not require a vector.

- [ ] **Step 4: Add write-path tests.**

~~~python
async def test_add_knowledge_embeds_content_and_writes_metadata(self):
    count = await store.add_knowledge([
        {"id": "k-1", "title": "差旅制度", "content": "住宿上限", "metadata": {"chunk": 0}},
    ])
    self.assertEqual(count, 1)
    self.assertEqual(provider.calls, [["住宿上限"]])
    self.assertIn("INSERT INTO knowledge_chunks", conn.executed_sql[0])
~~~

Also test empty queries skip embedding and profile upsert returns only the newest profile.

- [ ] **Step 5: Run focused tests and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_vector_store -v

Expected: PASS using fakes only.

~~~bash
git add core/embeddings.py core/vector_store.py tests/test_vector_store.py
git commit -m "feat: add explicit pgvector retrieval store"
~~~

### Task 3: Migrate knowledge and long-term memory off ChromaDB

**Files:**
- Modify: mcp/knowledge_base.py, memory/conversation_memory.py, api/main.py, docs/architecture-and-flow.md
- Modify: tests/test_vector_store.py, tests/test_clarification_memory.py

**Interfaces:**
- Consumes VectorStore and EmbeddingProvider.
- Produces KnowledgeBase(vector_store) and MemoryManager(..., vector_store=...) without changing their public query/context methods.

- [ ] **Step 1: Write compatibility tests for current response shapes.**

~~~python
async def test_knowledge_search_keeps_title_content_score_shape(self):
    kb = KnowledgeBase(vector_store=fake_vector_store)
    result = await kb.search_handler({"query": "住宿标准", "top_k": 3}, None)
    self.assertEqual(result[0], {
        "title": "差旅", "content": "上限 800", "score": 0.9, "chunk": 0,
    })

async def test_memory_context_combines_vector_history_and_profile(self):
    manager = memory_manager(redis=fake_redis, vector_store=fake_vector_store)
    context = await manager.get_context("emp-001", "conv-1", "报销")
    self.assertEqual(context.relevant_history, ["旧报销记录"])
    self.assertEqual(context.user_profile["preferences"], ["简洁"])
~~~

- [ ] **Step 2: Run focused tests to prove constructor compatibility is absent.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_vector_store tests.test_clarification_memory -v

Expected: FAIL because old classes have no vector_store injection.

- [ ] **Step 3: Replace Chroma calls while retaining Redis behavior.**

KnowledgeBase retains chunking/default documents/search_handler but delegates writes and searches to VectorStore. Add async ensure_default_documents() and call it from lifespan only when the store is empty.

MemoryManager retains all Redis keys, TTLs and compression. Replace episodic/profile collections and all Chroma query/add/delete logic with VectorStore methods. Remove chromadb imports and chroma_host/chroma_port/chroma_path constructor arguments.

FastAPI lifespan creates one Database, one embedding provider and one vector store, injects them into MemoryManager/KnowledgeBase, awaits default seeding and closes Database on shutdown. Update architecture documentation to say PostgreSQL + pgvector plus explicit embedding generation.

- [ ] **Step 4: Add one seed idempotency test and scan for stale runtime code.**

~~~python
async def test_default_seed_runs_once_when_store_is_empty(self):
    await kb.ensure_default_documents()
    await kb.ensure_default_documents()
    self.assertEqual(fake_vector_store.add_knowledge_calls, 1)
~~~

Run: rg -n "chromadb|CHROMA_" memory mcp api Dockerfile docker-compose.yml requirements.txt .env.example

Expected: no matches.

- [ ] **Step 5: Run focused tests and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_vector_store tests.test_clarification_memory -v

Expected: PASS without live PostgreSQL.

~~~bash
git add mcp/knowledge_base.py memory/conversation_memory.py api/main.py docs/architecture-and-flow.md tests/test_vector_store.py tests/test_clarification_memory.py
git commit -m "feat: migrate memory and knowledge to pgvector"
~~~

### Task 4: Define JWT actor identity and persisted operation primitives

**Files:**
- Create: core/auth.py, core/operations.py, core/operation_repository.py
- Modify: tests/test_side_effect_executor.py

**Interfaces:**
- Produces Actor, create_access_token(), decode_actor(), OperationRepository, PostgresOperationRepository and InMemoryOperationRepository.
- Consumes JWT_SECRET_KEY and Task 1 transaction helper.

- [ ] **Step 1: Write failing token and operation-store tests.**

~~~python
def test_valid_signed_token_is_the_only_actor_source(self):
    token = create_access_token(Actor("emp-001", "employee"), secret="test")
    self.assertEqual(decode_actor(token, secret="test"), Actor("emp-001", "employee"))
    with self.assertRaises(AuthenticationError):
        decode_actor(token + "x", secret="test")

async def test_same_key_with_changed_request_hash_is_rejected(self):
    await repo.save_idempotency(employee, "reimbursement_submit", "k1", "hash-a", {"ok": True})
    with self.assertRaises(IdempotencyConflict):
        await repo.save_idempotency(employee, "reimbursement_submit", "k1", "hash-b", {"ok": True})
~~~

- [ ] **Step 2: Run it before implementation.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_side_effect_executor -v

Expected: FAIL because core.auth and core.operations do not exist.

- [ ] **Step 3: Implement trusted values and repositories.**

JWT claims are sub, role and exp. get_current_actor accepts only an Authorization Bearer token and returns HTTP 401 for missing, expired or malformed tokens. A dev-token helper is allowed only when APP_ENV=development.

OperationRepository has create_pending(), get_for_actor(), lock_for_confirmation(), load_idempotency(), save_completed() and append_audit(). The PostgreSQL locking method uses SELECT FOR UPDATE inside the executor transaction. The in-memory implementation has one asyncio.Lock and is test-only.

- [ ] **Step 4: Add owner and expiry checks.**

~~~python
async def test_other_actor_cannot_load_pending_operation(self):
    operation = await repo.create_pending(employee, "access_request", {}, preview, 1, "hash", future)
    self.assertIsNone(await repo.get_for_actor(operation.operation_id, Actor("emp-002", "employee")))
~~~

Also assert that an expired pending record transitions to EXPIRED and records preview_expired.

- [ ] **Step 5: Run focused tests and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_side_effect_executor -v

Expected: PASS without FastAPI or PostgreSQL.

~~~bash
git add core/auth.py core/operations.py core/operation_repository.py tests/test_side_effect_executor.py
git commit -m "feat: add authenticated operation persistence"
~~~

### Task 5: Implement the generic four-stage executor and executor-only tools

**Files:**
- Create: core/side_effect_executor.py
- Modify: mcp/tool_manager.py, tools/reimbursement_tool.py
- Modify: tests/test_side_effect_executor.py, tests/test_agent_tools.py

**Interfaces:**
- Consumes OperationRepository, Actor and action adapters.
- Produces SideEffectExecutor.prepare(), get_preview(), confirm() and Tool(executor_only=True).

- [ ] **Step 1: Write four-stage behavior tests.**

~~~python
async def test_prepare_creates_preview_without_commit(self):
    operation = RecordingOperation("demo_write", version=3)
    executor = SideEffectExecutor(repo, {"demo_write": operation})
    preview = await executor.prepare(employee, ActionProposal("demo_write", {"amount": 100}))
    self.assertEqual(preview.state, OperationState.PENDING)
    self.assertEqual(operation.commit_calls, 0)

async def test_same_confirmation_key_replays_one_commit(self):
    preview = await executor.prepare(employee, ActionProposal("demo_write", {"amount": 100}))
    first = await executor.confirm(employee, preview.operation_id, "retry-1")
    second = await executor.confirm(employee, preview.operation_id, "retry-1")
    self.assertEqual(first, second)
    self.assertEqual(operation.commit_calls, 1)
~~~

- [ ] **Step 2: Run to verify SideEffectExecutor is absent.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_side_effect_executor -v

Expected: FAIL because core.side_effect_executor does not exist.

- [ ] **Step 3: Implement exact prepare/confirm ordering.**

prepare rejects unknown actions, calls adapter.precheck, audits rejected validation, creates pending only for allowed results and computes a stable SHA-256 snapshot over action_type, actor ID, normalized parameters and resource version.

confirm runs one transaction:

~~~text
lock actor-owned operation
→ replay matching idempotency response
→ reject non-pending or expired operation
→ run adapter precheck again
→ reject changed version/hash
→ adapter.commit once
→ save idempotency response, mark completed and audit
~~~

Add executor_only: bool = False to Tool. Exclude executor-only tools from anthropic_tools_for_agent. MCPToolManager.call gets executor_authorized: bool = False and rejects an executor-only call unless true. Mark submit_claim executor-only; adapters call it with caller_agent="billing", executor_authorized=True and cache disabled.

- [ ] **Step 4: Add the required safety tests.**

~~~python
async def test_two_concurrent_confirms_have_one_business_effect(self):
    preview = await executor.prepare(employee, ActionProposal("demo_write", {"amount": 100}))
    results = await asyncio.gather(
        executor.confirm(employee, preview.operation_id, "a"),
        executor.confirm(employee, preview.operation_id, "b"),
        return_exceptions=True,
    )
    self.assertEqual(operation.commit_calls, 1)
    self.assertEqual(sum(isinstance(item, dict) for item in results), 1)
~~~

Also cover 601-second expiry, different actor, changed resource version and direct tool-manager submit denied without executor_authorized=True.

- [ ] **Step 5: Run focused tests and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_side_effect_executor tests.test_agent_tools -v

Expected: PASS; existing read tools remain callable.

~~~bash
git add core/side_effect_executor.py mcp/tool_manager.py tools/reimbursement_tool.py tests/test_side_effect_executor.py tests/test_agent_tools.py
git commit -m "feat: add four-stage side effect executor"
~~~

### Task 6: Convert reimbursement submission into a persisted preview

**Files:**
- Modify: core/reimbursement_models.py, core/reimbursement_repository.py, core/reimbursement_harness.py
- Modify: tools/reimbursement_tool.py, agents/agent_orchestrator.py
- Modify: tests/test_reimbursement_harness.py, tests/test_reimbursement_tools.py

**Interfaces:**
- Consumes Database, SideEffectExecutor, Actor and OperationPreview.
- Produces PostgresReimbursementRepository, ReimbursementSubmitOperation and HarnessResult.pending_operation.

- [ ] **Step 1: Change the expected billing behavior first.**

~~~python
async def test_valid_claim_creates_preview_not_final_submission(self):
    result = await harness.handle(request_with_actor("报销上海酒店 600"), decision_provider(valid_facts))
    self.assertEqual(result.case.state, ClaimState.AWAITING_CONFIRMATION)
    self.assertEqual(result.pending_operation.action_type, "reimbursement_submit")
    self.assertFalse(any(trace.tool_name == "submit_claim" for trace in result.tool_calls))
~~~

- [ ] **Step 2: Run reimbursement tests before changing the Harness.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_reimbursement_harness tests.test_reimbursement_tools -v

Expected: FAIL because HarnessResult has no pending_operation.

- [ ] **Step 3: Port repository operations and add reimbursement adapter.**

Persist case fields in reimbursement_cases with facts, expenses and validation JSONB plus monotonic version. Keep existing hotel limit, receipt and budget rules. In final submit, lock the case and budget data with SELECT FOR UPDATE, validate ownership/state/budget again and reserve the budget atomically.

Implement ReimbursementSubmitOperation with precheck(owner, awaiting_confirmation, validation, budget), build_preview(items, total, policy reference, budget impact), and commit() that calls the executor-only submit_claim tool through MCPToolManager.

When Harness validation succeeds, it calls executor.prepare(actor, ActionProposal("reimbursement_submit", {"case_id": case.case_id})), stores pending_operation and returns its summary. Remove ordinary chat's direct submit path. A changed fact or expense invalidates any pending operation for that case.

- [ ] **Step 4: Add modification and retry assertions.**

~~~python
async def test_expense_change_invalidates_previous_preview(self):
    first = await ready_with_preview()
    await change_expense_to_700()
    with self.assertRaises(OperationConflict):
        await executor.confirm(employee, first.pending_operation.operation_id, "submit-1")
~~~

Also assert a repeated API confirmation key returns the same claim number and reserves budget exactly once.

- [ ] **Step 5: Run focused reimbursement tests and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_reimbursement_harness tests.test_reimbursement_tools -v

Expected: PASS with in-memory/fake repository variants; no database process is required.

~~~bash
git add core/reimbursement_models.py core/reimbursement_repository.py core/reimbursement_harness.py tools/reimbursement_tool.py agents/agent_orchestrator.py tests/test_reimbursement_harness.py tests/test_reimbursement_tools.py
git commit -m "feat: prepare reimbursement submissions for confirmation"
~~~

### Task 7: Add IT access requests and JWT-scoped operation APIs

**Files:**
- Create: core/access_request.py, tests/test_access_request.py, tests/test_operation_api.py
- Modify: api/main.py, agents/agent_orchestrator.py, core/reimbursement_harness.py, tests/test_reimbursement_api.py

**Interfaces:**
- Consumes SideEffectExecutor, Actor, ActionProposal and OperationPreview.
- Produces AccessRequestOperation, POST /access-requests/prepare, GET /operations/{operation_id} and POST /operations/{operation_id}/confirm.

- [ ] **Step 1: Write failing second-domain and endpoint tests.**

~~~python
async def test_access_request_uses_the_shared_confirm_contract(self):
    preview = await executor.prepare(employee, ActionProposal("access_request", {
        "system_name": "vpn", "permission": "read", "reason": "remote support",
    }))
    result = await executor.confirm(employee, preview.operation_id, "access-1")
    self.assertEqual(result["state"], "submitted")

async def test_different_token_cannot_confirm_operation(self):
    response = await call_confirm_as("emp-002", operation_id, "key-1")
    self.assertEqual(response.status_code, 403)
~~~

- [ ] **Step 2: Run tests before adding domain/routes.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_access_request tests.test_operation_api -v

Expected: FAIL because the module and operation routes are absent.

- [ ] **Step 3: Implement deterministic access rules and HTTP boundary.**

AccessRequestOperation accepts systems vpn, gitlab and finance_portal, permissions read/write, and a nonempty reason under 201 characters. Employees can request read; only managers can request write. commit inserts one local access_requests record with submitted state and never calls a real IAM service.

POST /access-requests/prepare, GET /operations/{operation_id} and POST /operations/{operation_id}/confirm all require get_current_actor. Confirm reads a mandatory Idempotency-Key header and maps missing header to 422, other actor to 403 and expiry/version/key conflicts to 409. Add POST /auth/dev-token only when APP_ENV=development; production returns 404.

Extend OrchestratorResult and ChatResponse with optional pending_operation. Billing preview responses populate it; other Agent replies remain unchanged.

- [ ] **Step 4: Add replay and validation tests.**

~~~python
async def test_same_api_confirmation_header_replays_first_response(self):
    first = await call_confirm_as("emp-001", operation_id, "browser-retry")
    second = await call_confirm_as("emp-001", operation_id, "browser-retry")
    self.assertEqual(first.json(), second.json())
~~~

Also cover missing JWT (401), unsupported system precheck rejection, employee write request denial and ChatResponse pending-operation serialization.

- [ ] **Step 5: Run focused API tests and commit.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_access_request tests.test_operation_api tests.test_reimbursement_api -v

Expected: PASS with injected fakes; no live server required.

~~~bash
git add core/access_request.py api/main.py agents/agent_orchestrator.py core/reimbursement_harness.py tests/test_access_request.py tests/test_operation_api.py tests/test_reimbursement_api.py
git commit -m "feat: expose confirmed access request operations"
~~~

### Task 8: Update documentation and run the deliberately narrow verification set

**Files:**
- Modify: README.md, PROJECT_DETAILED_INTRODUCTION.md, docs/architecture-and-flow.md, tools/README.md, .gitignore
- Test: test modules from Tasks 2, 5, 6 and 7

**Interfaces:**
- Produces accurate deployment and Swagger demonstration instructions and resume-safe capability boundaries.

- [ ] **Step 1: Prepare a documentation checklist.**

Every user-facing document must say PostgreSQL + pgvector, not Redis + ChromaDB. Describe mcp/ as project-internal tool governance, not standard MCP transport. Describe the protocol as applied to reimbursement submission and local IT access request, never external ERP/IAM integration.

- [ ] **Step 2: Update deployment and Swagger walkthrough.**

Document required PostgreSQL variables, docker compose startup, development JWT token, Bearer authentication and Idempotency-Key. Add this three-call demo:

~~~text
1. POST /access-requests/prepare with a development JWT → operation_id and preview.
2. POST /operations/{operation_id}/confirm with same JWT and Idempotency-Key: demo-1 → submitted.
3. Repeat step 2 with demo-1 → identical response and no second request row.
~~~

Remove obsolete Chroma comments/paths from .gitignore and docs.

- [ ] **Step 3: Run the required, small verification gate.**

Run: .\.venv\Scripts\python.exe -m unittest tests.test_vector_store tests.test_side_effect_executor tests.test_access_request tests.test_operation_api tests.test_reimbursement_harness -v

Expected: PASS. Do not block this iteration on unrelated monitor, model or full-service test failures.

Run: rg -n "chromadb|CHROMA_" --glob "!docs/superpowers/**" --glob "!*.lock" .

Expected: no source/runtime/deployment matches.

- [ ] **Step 4: Run Docker only when local credentials and Docker are available.**

Run: docker compose up --build

Expected: PostgreSQL, Redis and EchoMind pass health checks. Perform the documented three-call access-request demo once. If Docker, a model download or LLM credentials are unavailable, record this optional check as deferred without changing the unit-test result.

- [ ] **Step 5: Commit documentation.**

~~~bash
git add README.md PROJECT_DETAILED_INTRODUCTION.md docs/architecture-and-flow.md tools/README.md .gitignore
git commit -m "docs: document pgvector hitl operations"
~~~

## Plan Self-Review

- **Spec coverage:** Tasks 1-3 replace ChromaDB with PostgreSQL + pgvector and explicit embeddings. Tasks 4-7 add JWT identity, persisted preview/confirmation/idempotency/audit, reimbursement submission and IT access requests. Task 8 documents and verifies the result.
- **Testing scope:** Each task has a focused unittest. The only Docker command is explicitly optional; the required gate is the small safety suite.
- **Type consistency:** Actor, ActionProposal, OperationPreview, SideEffectOperation and SideEffectExecutor are defined before consumers. Executor-only writes use MCPToolManager.call with executor_authorized=True.
- **Scope discipline:** No task creates a frontend, real external integration, distributed workflow engine or a requirement to run every existing test.

