# 差旅报销 Agent Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, in-memory travel-reimbursement sandbox in which the Billing Agent is constrained by a recoverable case-state Harness and every business action goes through the existing tool-governance layer.

**Architecture:** Add a small reimbursement domain module (models plus in-memory repository), then place `ReimbursementCaseHarness` between billing task routing and tool execution. The Billing Agent returns a constrained decision—facts and an intent—while the Harness owns state transitions, confirmation gates, idempotency and traces. Reimbursement tools are registered by the existing discovery mechanism and execute only through `MCPToolManager.call()`.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, `unittest.IsolatedAsyncioTestCase`, Anthropic-compatible SDK, existing `MCPToolManager`, existing Prometheus monitor.

**Spec:** `docs/superpowers/specs/2026-10-27-reimbursement-agent-harness-design.md`

## Global Constraints

- Keep the scope to a travel-reimbursement demonstration; do not build ERP/OA integrations, OCR, real payments, accounting, tax, multi-currency or a management frontend.
- Add no third-party runtime dependency; the demonstration data repository is process-local and seeded in code.
- Do not store case state in `ConversationMemory` free text; use a dedicated repository keyed by `case_id` and active `(user_id, conv_id)`.
- All business tools, including Harness-triggered calls, must use `MCPToolManager.call()` with `caller_agent="billing"`; do not invoke handlers directly.
- Keep the existing Tool Manager permission, schema, timeout, cache, circuit-breaker and fallback behavior intact.
- Treat only an explicit confirmation phrase as authorization for `submit_claim`; ambiguous acknowledgements must remain in `awaiting_confirmation`.
- Make `submit_claim` atomically reserve the mock budget and submit the mock claim. Do not expose a separate `reserve_budget` tool.
- Preserve existing APIs, tests, route-monitor feedback and unrelated working-tree changes.
- Use `python -m unittest`; do not install dependencies outside `.venv`.

---

## File Structure

| File | Responsibility |
|---|---|
| `core/reimbursement_models.py` | Immutable-enough domain types: state/action enums, expense items, claims, events, Agent decisions and Harness results. |
| `core/reimbursement_repository.py` | Async process-local case store, active-conversation index, seeded employee/department budgets and idempotency-result store. |
| `core/reimbursement_harness.py` | State-transition authority: merge facts, request a constrained Agent decision, permit/deny actions, call governed tools, append events and render safe user responses. |
| `tools/reimbursement_tool.py` | Tool definitions and async handlers which delegate to the repository; no direct HTTP/ERP dependency. |
| `agents/agent_orchestrator.py` | Billing-specific structured decision method and Harness routing, while retaining normal Agent handling for general/technical paths. |
| `api/main.py` | Construct and share the repository/Harness at lifespan; expose read-only case and role-scoped simulation endpoints. |
| `monitor/performance_monitor.py` | Include Harness snapshot metrics in `/monitor` output and optional Prometheus gauges without changing routing penalties. |
| `evaluation/run_reimbursement_harness_benchmark.py` | Run deterministic, offline Harness scenarios and print per-scenario outcomes plus aggregate rates. |
| `evaluation/fixtures/reimbursement_harness_cases.json` | Synthetic normal, validation, recovery and idempotency cases used by the benchmark. |
| `tests/test_reimbursement_models_repository.py` | Unit coverage for case lookup, seeded budgets, versioning and idempotency persistence. |
| `tests/test_reimbursement_tools.py` | Tool discovery, permission and governed-call integration coverage. |
| `tests/test_reimbursement_harness.py` | State machine, confirmation, error/recovery and trace coverage using a deterministic fake decision provider. |
| `tests/test_reimbursement_orchestration.py` | Billing route integration and structured-decision fallback coverage. |
| `tests/test_reimbursement_api.py` | Minimal FastAPI endpoint behavior with injected in-memory application globals. |

## Shared Interfaces

Create the following names exactly; later tasks depend on them.

```python
# core/reimbursement_models.py
class ClaimState(str, Enum):
    COLLECTING_INFO = "collecting_info"
    DRAFT_CREATED = "draft_created"
    VALIDATING = "validating"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUBMITTED = "submitted"
    REWORK_REQUIRED = "rework_required"
    PAID = "paid"
    CLOSED = "closed"

class ReimbursementAction(str, Enum):
    COLLECT_FACTS = "collect_facts"
    VALIDATE = "validate"
    CONFIRM_SUBMIT = "confirm_submit"
    GET_STATUS = "get_status"

@dataclass
class ReimbursementDecision:
    facts: dict[str, Any]
    action: ReimbursementAction
    explicit_confirmation: bool = False
    user_message: str = ""

@dataclass
class HarnessToolTrace:
    round: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0

@dataclass
class HarnessResult:
    content: str
    case: ReimbursementCase
    tool_calls: list[HarnessToolTrace]
    success: bool

# core/reimbursement_repository.py
class InMemoryReimbursementRepository:
    def __init__(self, budgets: dict[str, float] | None = None) -> None: ...
    async def get_or_create_active_case(self, user_id: str, conv_id: str) -> ReimbursementCase: ...
    async def get_case(self, case_id: str) -> ReimbursementCase | None: ...
    async def save_case(self, case: ReimbursementCase) -> ReimbursementCase: ...
    async def remaining_budget(self, department: str) -> float: ...
    async def get_idempotent_result(self, key: str) -> dict[str, Any] | None: ...
    async def save_idempotent_result(self, key: str, result: dict[str, Any]) -> None: ...

# core/reimbursement_harness.py
class ReimbursementCaseHarness:
    async def handle(self, req: Any, agent: Any) -> HarnessResult: ...
    async def get_case_view(self, case_id: str) -> dict[str, Any] | None: ...
    async def review_claim(self, case_id: str, reviewer_id: str, approved: bool, comment: str = "") -> dict[str, Any]: ...
    async def simulate_payment(self, case_id: str, finance_user_id: str) -> dict[str, Any]: ...

# agents/agent_orchestrator.py
class BillingAgent(BaseAgent):
    async def decide_reimbursement(self, req: Request, case: ReimbursementCase) -> ReimbursementDecision: ...
```

`ReimbursementCase` contains at least `case_id`, `user_id`, `conv_id`, `department`, `state`, `facts`, `expenses`, `validation`, `budget_reserved`, `claim_no`, `version`, `events` and `created_at`. Each state-changing save increments `version` exactly once; the Harness derives idempotency keys from `case_id`, `action.value` and the pre-transition version.

### Task 1: Define domain models and the in-memory repository

**Files:**
- Create: `core/reimbursement_models.py`
- Create: `core/reimbursement_repository.py`
- Test: `tests/test_reimbursement_models_repository.py`

**Interfaces:**
- Consumes: standard-library `dataclasses`, `enum`, `datetime`, `asyncio` and `typing` only.
- Produces: all shared model and repository interfaces above, plus `append_event(case, event_type, detail)` and `seeded_budget_for(department)` helpers used by the tools and Harness.

- [ ] **Step 1: Write failing repository tests for a stable active case and seeded budget.**

```python
class ReimbursementRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_conversation_reuses_one_open_case(self):
        repo = InMemoryReimbursementRepository()
        first = await repo.get_or_create_active_case("emp-001", "conv-1")
        second = await repo.get_or_create_active_case("emp-001", "conv-1")
        self.assertEqual(first.case_id, second.case_id)
        self.assertEqual(first.state, ClaimState.COLLECTING_INFO)

    async def test_idempotency_result_is_stored_by_exact_key(self):
        repo = InMemoryReimbursementRepository()
        await repo.save_idempotent_result("case:submit:0", {"claim_no": "CLM-001"})
        self.assertEqual(
            await repo.get_idempotent_result("case:submit:0"),
            {"claim_no": "CLM-001"},
        )
```

- [ ] **Step 2: Run the new test module and verify import failure.**

Run: `python -m unittest tests.test_reimbursement_models_repository -v`
Expected: FAIL because `core.reimbursement_models` and `core.reimbursement_repository` do not exist.

- [ ] **Step 3: Implement the smallest complete domain model and repository.**

```python
@dataclass
class ExecutionEvent:
    at: str
    version: int
    event_type: str
    detail: dict[str, Any]

@dataclass
class ReimbursementCase:
    case_id: str
    user_id: str
    conv_id: str
    department: str = "engineering"
    state: ClaimState = ClaimState.COLLECTING_INFO
    facts: dict[str, Any] = field(default_factory=dict)
    expenses: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    budget_reserved: bool = False
    claim_no: str | None = None
    version: int = 0
    events: list[ExecutionEvent] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

Use `asyncio.Lock` in repository mutation methods. Seed a single `engineering` departmental budget and employees `emp-001` (employee), `manager-001` (manager), and `finance-001` (finance) in code. Return a defensive `deepcopy` from reads and saves so a caller cannot mutate stored state without `save_case`.

- [ ] **Step 4: Extend the tests for state versioning, defensive copying and terminal-case exclusion.**

```python
async def test_closed_case_is_not_reused_for_a_new_active_conversation(self):
    case = await repo.get_or_create_active_case("emp-001", "conv-1")
    case.state = ClaimState.CLOSED
    await repo.save_case(case)
    replacement = await repo.get_or_create_active_case("emp-001", "conv-1")
    self.assertNotEqual(replacement.case_id, case.case_id)
```

- [ ] **Step 5: Run the module and commit the independently working persistence core.**

Run: `python -m unittest tests.test_reimbursement_models_repository -v`
Expected: PASS.

```bash
git add core/reimbursement_models.py core/reimbursement_repository.py tests/test_reimbursement_models_repository.py
git commit -m "feat: add reimbursement case repository"
```

### Task 2: Add governed reimbursement tools with deterministic policy and budget rules

**Files:**
- Create: `tools/reimbursement_tool.py`
- Modify: `tools/loader.py` only if dependency injection is required by its current `build_tools()` contract
- Test: `tests/test_reimbursement_tools.py`

**Interfaces:**
- Consumes: `InMemoryReimbursementRepository`, `ReimbursementCase`, `ClaimState`, `Tool`, and trusted context fields `user_id`, `conv_id`, `agent_type`.
- Produces: `build_tools(repository: InMemoryReimbursementRepository) -> list[Tool]` and handlers named `search_policy`, `get_case`, `check_budget`, `create_draft`, `add_expense_items`, `validate_claim`, `submit_claim`, `review_claim`, `simulate_payment`.

- [ ] **Step 1: Write failing tests that execute tools through `MCPToolManager.call`, not handlers.**

```python
async def test_submit_is_denied_until_harness_confirmation_context(self):
    manager, repo = manager_with_reimbursement_tools()
    case = await repo.get_or_create_active_case("emp-001", "conv-1")
    result = await manager.call(
        "submit_claim",
        {"case_id": case.case_id, "idempotency_key": "k1"},
        {"user_id": "emp-001", "conv_id": "conv-1", "confirmed": False},
        caller_agent=AgentType.BILLING,
    )
    self.assertFalse(result.success)
    self.assertIn("确认", result.error)
```

- [ ] **Step 2: Run the tool tests and verify discovery/import failure.**

Run: `python -m unittest tests.test_reimbursement_tools -v`
Expected: FAIL because the reimbursement tool module and helpers are absent.

- [ ] **Step 3: Implement tool factories and exact rule behavior.**

Implement `build_tools(repository)` as an explicit dependency-injected factory for unit tests. Update `tools.loader.discover_agent_tools()` to accept an optional `dependencies: dict[str, Any]`; it must pass `repository=dependencies["reimbursement_repository"]` only to the reimbursement module and keep calling no-argument `build_tools()` for existing modules. In `api.main.lifespan`, pass the repository to discovery.

Implement these fixed demo rules in the handlers:

```python
HOTEL_LIMITS = {"上海": 800.0, "北京": 800.0, "其他": 600.0}
REQUIRED_FACTS = ("trip_city", "trip_date", "project_code")

# validate_claim result shape
{
    "valid": bool,
    "issues": [{"code": "hotel_over_limit", "message": "..."}],
    "total_amount": 2060.0,
    "required_budget": 2060.0,
}
```

`search_policy` returns the fixed policy limits and a policy-reference string for the requested city; it is a read-only explanation tool, not a second RAG system. `submit_claim` must check `context["confirmed"] is True`, case state is `AWAITING_CONFIRMATION`, validation is valid and available budget covers the total. In one repository-locked mutation it deducts the budget, assigns one `CLM-` number, sets `budget_reserved=True`, sets state to `SUBMITTED`, persists the idempotent result and returns the previously persisted result when the same key is used again. Mark `search_policy`, `get_case` and `check_budget` cacheable; leave write tools uncacheable.

- [ ] **Step 4: Add tool-level rejection and idempotency tests.**

```python
async def test_second_submit_with_same_key_does_not_deduct_budget_twice(self):
    first = await manager.call("submit_claim", params, confirmed_context, caller_agent=AgentType.BILLING)
    second = await manager.call("submit_claim", params, confirmed_context, caller_agent=AgentType.BILLING)
    self.assertTrue(first.success and second.success)
    self.assertEqual(first.data["claim_no"], second.data["claim_no"])
    self.assertEqual(await repo.remaining_budget("engineering"), original_budget - total)
```

Also cover: technical Agent denied by `allowed_agents`, missing `case_id` rejected by schema, hotel-over-limit validation and insufficient-budget submission rejection.

- [ ] **Step 5: Run focused and existing tool tests, then commit.**

Run: `python -m unittest tests.test_reimbursement_tools tests.test_agent_tools -v`
Expected: PASS; existing discovery and permission tests continue to pass.

```bash
git add tools/reimbursement_tool.py tools/loader.py api/main.py tests/test_reimbursement_tools.py
git commit -m "feat: add governed reimbursement tools"
```

### Task 3: Implement the deterministic case Harness and its recovery state machine

**Files:**
- Create: `core/reimbursement_harness.py`
- Test: `tests/test_reimbursement_harness.py`

**Interfaces:**
- Consumes: the Task 1 domain/repository APIs, the Task 2 registered `MCPToolManager`, and a provider exposing `async decide_reimbursement(req, case) -> ReimbursementDecision`.
- Produces: `ReimbursementCaseHarness.handle`, `get_case_view`, `review_claim`, `simulate_payment`, plus `stats()` returning counters `completed`, `clarifications`, `state_rejections`, `tool_failure_recoveries`, `confirmations`.

- [ ] **Step 1: Write Harness tests with a deterministic decision provider.**

```python
class DecisionProvider:
    def __init__(self, *decisions):
        self.decisions = list(decisions)

    async def decide_reimbursement(self, _req, _case):
        return self.decisions.pop(0)

async def test_missing_project_code_keeps_case_and_returns_one_question(self):
    result = await harness.handle(request("上海出差，酒店 600"), provider_with(
        ReimbursementDecision({"trip_city": "上海", "expenses": [{"type": "hotel", "amount": 600}]}, ReimbursementAction.COLLECT_FACTS)
    ))
    self.assertIn("项目", result.content)
    self.assertEqual(result.case.state, ClaimState.COLLECTING_INFO)
    self.assertEqual(result.case.facts["trip_city"], "上海")
```

- [ ] **Step 2: Run the Harness test module and verify import failure.**

Run: `python -m unittest tests.test_reimbursement_harness -v`
Expected: FAIL because `ReimbursementCaseHarness` does not exist.

- [ ] **Step 3: Implement explicit transitions and governed calls.**

Use a state-to-allowed-action mapping rather than nested prompt-dependent behavior:

```python
ALLOWED_ACTIONS = {
    ClaimState.COLLECTING_INFO: {ReimbursementAction.COLLECT_FACTS, ReimbursementAction.GET_STATUS},
    ClaimState.DRAFT_CREATED: {ReimbursementAction.COLLECT_FACTS, ReimbursementAction.VALIDATE, ReimbursementAction.GET_STATUS},
    ClaimState.VALIDATING: {ReimbursementAction.VALIDATE, ReimbursementAction.GET_STATUS},
    ClaimState.AWAITING_CONFIRMATION: {ReimbursementAction.CONFIRM_SUBMIT, ReimbursementAction.COLLECT_FACTS, ReimbursementAction.GET_STATUS},
    ClaimState.SUBMITTED: {ReimbursementAction.GET_STATUS},
    ClaimState.REWORK_REQUIRED: {ReimbursementAction.COLLECT_FACTS, ReimbursementAction.VALIDATE, ReimbursementAction.GET_STATUS},
}
```

`handle` must merge sanitized decision facts, issue exactly one missing-field question for the first absent required fact, call `create_draft`/`add_expense_items`/`search_policy`/`validate_claim` through the manager with `caller_agent="billing"`, and store a `HarnessToolTrace` for each. A valid claim transitions to `AWAITING_CONFIRMATION`; an invalid claim transitions to `REWORK_REQUIRED`; only `explicit_confirmation=True` permits `submit_claim`. On Tool Manager failure, append an event, increment `tool_failure_recoveries`, keep the prior state, and return a retry-safe response.

- [ ] **Step 4: Add complete state-machine and recovery cases.**

```python
async def test_ambiguous_acknowledgement_does_not_submit(self):
    result = await harness.handle(request("好的"), provider_with(
        ReimbursementDecision({}, ReimbursementAction.CONFIRM_SUBMIT, explicit_confirmation=False)
    ))
    self.assertEqual(result.case.state, ClaimState.AWAITING_CONFIRMATION)
    self.assertFalse(any(trace.tool_name == "submit_claim" for trace in result.tool_calls))
```

Also cover complete valid flow to `SUBMITTED`, invalid hotel limit to `REWORK_REQUIRED`, manager rejection retaining the prior state, a manager rejection after a `submitted` review reverting to `REWORK_REQUIRED`, payment reaching `PAID`, terminal case refusing modifications, and an event for every state transition.

- [ ] **Step 5: Run the Harness tests and commit.**

Run: `python -m unittest tests.test_reimbursement_harness -v`
Expected: PASS.

```bash
git add core/reimbursement_harness.py tests/test_reimbursement_harness.py
git commit -m "feat: add reimbursement case harness"
```

### Task 4: Constrain Billing Agent decisions and route billing tasks through the Harness

**Files:**
- Modify: `agents/agent_orchestrator.py`
- Test: `tests/test_reimbursement_orchestration.py`
- Test: `tests/test_task_orchestration.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `ReimbursementCaseHarness`, `ReimbursementCase`, `ReimbursementDecision`, and current `Request`, `AgentResponse`, `ToolCallTrace` types.
- Produces: `BillingAgent.decide_reimbursement`, `AgentOrchestrator(..., reimbursement_harness: ReimbursementCaseHarness | None = None)`, and Billing task execution through `harness.handle`.

- [ ] **Step 1: Write failing tests for billing interception and malformed model output fallback.**

```python
async def test_billing_task_uses_harness_not_base_agent_tool_loop(self):
    harness = AsyncMock()
    harness.handle.return_value = HarnessResult("已创建草稿", open_case(), [], True)
    orchestrator = make_orchestrator(harness=harness)
    await orchestrator.run_plan(request, billing_single_task_plan())
    harness.handle.assert_awaited_once()

async def test_bad_decision_json_becomes_safe_collect_facts_decision(self):
    agent = BillingAgent(FakeClient([[{"type": "text", "text": "not json"}]]), "fake")
    decision = await agent.decide_reimbursement(request("好的"), open_case())
    self.assertEqual(decision.action, ReimbursementAction.COLLECT_FACTS)
    self.assertFalse(decision.explicit_confirmation)
```

- [ ] **Step 2: Run the orchestration tests and verify the new expectations fail.**

Run: `python -m unittest tests.test_reimbursement_orchestration -v`
Expected: FAIL because the Harness constructor argument and structured decision method do not exist.

- [ ] **Step 3: Implement a no-tool structured decision path in `BillingAgent`.**

Add `decide_reimbursement` without changing `BaseAgent._call_llm` behavior for other domains. It must call the existing client with no `tools` argument and require a JSON object of this exact shape:

```json
{
  "facts": {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "PRJ-001", "expenses": [{"type": "hotel", "amount": 600}]},
  "action": "validate",
  "explicit_confirmation": false
}
```

Parse only known fact keys; reject non-object facts, unsupported actions and non-list expenses. The fallback is `ReimbursementDecision({}, ReimbursementAction.COLLECT_FACTS, False, req.message)`. Set the Billing system prompt to the employee travel-reimbursement role and state that it proposes data/actions only; it must not claim submission occurred until it receives Harness results.

- [ ] **Step 4: Route only billing tasks through the Harness without breaking generic routing.**

Pass the initialized Harness to `AgentOrchestrator`. In `_execute`, after selecting the best Agent, when the requested type is `AgentType.BILLING` and a Harness exists, call `await harness.handle(req, agent)` and adapt the result into `AgentResponse(AgentType.BILLING, harness_result.content, harness_result.success, tool_calls=[ToolCallTrace(...)])`. This single interception point also covers `run_plan`, sequential tasks and explicit parallel routing; no billing path may use `BaseAgent._call_llm` with reimbursement tools. Keep normal `_execute` behavior for `GENERAL`, `TECHNICAL`, account mapping and the no-Harness test configuration. If the Harness returns failure, retain the existing General-Agent text fallback but do not retry a write action. Copy every field from `HarnessToolTrace` into `ToolCallTrace` so `/chat` serialization remains unchanged.

- [ ] **Step 5: Run billing and regression suites, then commit.**

Run: `python -m unittest tests.test_reimbursement_orchestration tests.test_task_orchestration tests.test_agent_tools -v`
Expected: PASS; technical/general task behavior and existing ToolCallTrace tests remain unchanged.

```bash
git add agents/agent_orchestrator.py tests/test_reimbursement_orchestration.py tests/test_task_orchestration.py tests/test_agent_tools.py
git commit -m "feat: route billing tasks through harness"
```

### Task 5: Wire the in-memory sandbox into FastAPI and expose minimal demonstration endpoints

**Files:**
- Modify: `api/main.py`
- Create: `tests/test_reimbursement_api.py`

**Interfaces:**
- Consumes: `InMemoryReimbursementRepository`, `ReimbursementCaseHarness`, `AgentOrchestrator`, and FastAPI `HTTPException`.
- Produces: `GET /reimbursements/{case_id}`, `POST /reimbursements/{case_id}/review`, and `POST /reimbursements/{case_id}/payment`.

- [ ] **Step 1: Write endpoint tests using injected module globals; do not start Redis, ChromaDB or a live model.**

```python
async def test_get_case_returns_a_serializable_case_view(self):
    api._reimbursement_harness = harness_with_case("CASE-1")
    payload = await api.get_reimbursement_case("CASE-1")
    self.assertEqual(payload["case_id"], "CASE-1")

async def test_review_requires_manager_role(self):
    with self.assertRaises(HTTPException) as caught:
        await api.review_reimbursement_claim("CASE-1", ReviewInput(reviewer_id="emp-001", approved=True))
    self.assertEqual(caught.exception.status_code, 403)
```

- [ ] **Step 2: Run the endpoint tests and verify they fail because endpoints and globals are absent.**

Run: `python -m unittest tests.test_reimbursement_api -v`
Expected: FAIL with missing endpoint/model names.

- [ ] **Step 3: Initialize and expose the demo components.**

In lifespan, construct one `InMemoryReimbursementRepository`, pass it to discovery, construct `ReimbursementCaseHarness(repository, _tool_manager)`, and pass that Harness to `AgentOrchestrator`. Add Pydantic bodies:

```python
class ReviewInput(BaseModel):
    reviewer_id: str
    approved: bool
    comment: str = ""

class PaymentInput(BaseModel):
    finance_user_id: str
```

`GET` returns a sanitized Harness view or 404. `POST /review` accepts only `manager-001` and passes the decision to the Harness; `POST /payment` accepts only `finance-001`, requires a submitted approved claim and delegates to `simulate_payment`. Convert missing case to 404 and forbidden role/state to 403/409; never access repository internals from routes.

- [ ] **Step 4: Add a chat-level smoke test that verifies a billing response still serializes ordinary `ToolCallResponse` values.**

```python
payload = ChatResponse(..., tool_calls=[ToolCallResponse(tool_name="validate_claim", ...)]).model_dump()
self.assertEqual(payload["tool_calls"][0]["tool_name"], "validate_claim")
```

- [ ] **Step 5: Run API and existing serialization tests, then commit.**

Run: `python -m unittest tests.test_reimbursement_api tests.test_agent_tools -v`
Expected: PASS.

```bash
git add api/main.py tests/test_reimbursement_api.py tests/test_agent_tools.py
git commit -m "feat: expose reimbursement demo endpoints"
```

### Task 6: Add Harness observability and an offline scenario benchmark

**Files:**
- Modify: `monitor/performance_monitor.py`
- Create: `evaluation/fixtures/reimbursement_harness_cases.json`
- Create: `evaluation/run_reimbursement_harness_benchmark.py`
- Create: `tests/test_reimbursement_benchmark.py`

**Interfaces:**
- Consumes: `ReimbursementCaseHarness.stats()`, synthetic JSON fixtures and the deterministic repository/Harness; no model API key, Redis or ChromaDB.
- Produces: `ReimbursementHarnessBenchmark.run() -> dict[str, Any]` with `total`, `passed`, `task_completion_rate`, `confirmation_safety_rate`, `recovery_rate`, `policy_consistency_rate`, `trace_completeness_rate` and per-case rows.

- [ ] **Step 1: Write failing benchmark tests against a complete and an unsafe-confirmation fixture.**

```python
async def test_benchmark_reports_confirmation_safety(self):
    report = await ReimbursementHarnessBenchmark(fixtures_path).run()
    self.assertEqual(report["total"], 2)
    self.assertEqual(report["confirmation_safety_rate"], 1.0)
    self.assertEqual(report["passed"], 2)
```

- [ ] **Step 2: Run the benchmark tests and verify they fail because the runner is absent.**

Run: `python -m unittest tests.test_reimbursement_benchmark -v`
Expected: FAIL with `ModuleNotFoundError` for the benchmark runner.

- [ ] **Step 3: Implement fixtures and the offline runner.**

Add eight synthetic cases: compliant complete submission, missing project code, hotel over limit, insufficient budget, ambiguous acknowledgement, manager rejection then corrected supplement, simulated tool failure then retry, and duplicate confirmation. The runner must use a queue-based fake decision provider rather than an LLM, assert each expected final state/tool absence or presence/event count, and calculate rates from those assertions. Print `json.dumps(report, ensure_ascii=False, indent=2)` from `main()`.

In `PerformanceMonitor.summary()`, append a `reimbursement_harness` field when the orchestrator exposes a Harness. In `_collect`, publish optional gauges for the five count/rate metrics only; do not feed them into `_routing_penalty` or alter current alerts.

- [ ] **Step 4: Add monitor-summary coverage for an orchestrator without a Harness and one with `stats()`.**

```python
summary = monitor.summary()
self.assertEqual(summary["reimbursement_harness"]["confirmations"], 2)
```

Also assert that existing agent/tool statistics remain present and that the no-Harness branch returns `{}` rather than raising.

- [ ] **Step 5: Run the benchmark, Harness and full unit-test suite, then commit.**

Run: `python -m unittest tests.test_reimbursement_benchmark tests.test_reimbursement_harness -v`
Expected: PASS.

Run: `python evaluation/run_reimbursement_harness_benchmark.py`
Expected: JSON report with eight rows and all specified rates.

Run: `python -m unittest discover -s tests -v`
Expected: PASS with all existing and reimbursement tests green.

```bash
git add monitor/performance_monitor.py evaluation/fixtures/reimbursement_harness_cases.json evaluation/run_reimbursement_harness_benchmark.py tests/test_reimbursement_benchmark.py
git commit -m "feat: evaluate reimbursement harness reliability"
```

## Final Verification

- [ ] Run `python -m unittest discover -s tests -v` in the active virtual environment and retain the passing output.
- [ ] Run `python evaluation/run_reimbursement_harness_benchmark.py` and confirm all eight synthetic scenarios report their expected state and safety assertions.
- [ ] Start the API with configured local dependencies, submit one compliant travel claim through `/chat`, explicitly confirm it, call the review endpoint as `manager-001`, call payment as `finance-001`, and verify `GET /reimbursements/{case_id}` shows `paid` plus an event trail.
- [ ] Repeat with “好的” at the confirmation step and verify there is no `submit_claim` trace and the case remains `awaiting_confirmation`.
- [ ] Inspect `GET /monitor` and confirm existing `agent_stats`/`tool_stats` coexist with `reimbursement_harness` metrics.
