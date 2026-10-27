# Multi-Intent Task Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add task-level multi-intent planning that safely selects single, parallel, sequential, or clarification execution for four business domains.

**Architecture:** Add a focused `TaskPlanner` that validates task evidence and dependencies independently of model calls. Extend the intent recognizer with domain candidates, then make the orchestrator execute a `TaskPlan`; API and memory persist only the pending clarification state needed for the next turn.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, Anthropic Async SDK, Redis, unittest.

**Spec:** `docs/superpowers/specs/2026-09-24-multi-intent-task-planning-design.md`

## Global Constraints

- Preserve asynchronous API, model, tool and memory call chains.
- Do not add dependencies or alter existing public fields destructively.
- Only `general`, `technical`, `billing`, and `account` are executable task domains.
- Execute in the current `main` worktree with the user's explicit approval.
- On planning uncertainty, return clarification rather than speculative parallel work.

---

### Task 1: Pure task planning model

**Files:**
- Create: `core/task_planner.py`
- Create: `tests/test_task_planner.py`

**Interfaces:**
- Produces: `BusinessDomain`, `ExecutionMode`, `DomainCandidate`, `PlannedTask`, `TaskPlan`, and `TaskPlanner.decide()`.
- Consumes: literal candidates and tasks, without an LLM dependency in unit tests.

- [ ] **Step 1: Write failing decision tests**

```python
plan = TaskPlanner().decide(
    message="VPN 登录不上，同时报销单被退回了",
    candidates=[DomainCandidate(BusinessDomain.TECHNICAL, 0.91), DomainCandidate(BusinessDomain.BILLING, 0.88)],
    tasks=[technical_task, billing_task],
)
assert plan.mode is ExecutionMode.PARALLEL
```

- [ ] **Step 2: Run the focused test module and observe ImportError because `core.task_planner` is absent**

Run: `python -m unittest tests/test_task_planner.py`

- [ ] **Step 3: Implement validation and mode decisions**

```python
if not credible_candidates:
    return clarify_plan(...)
if invalid_or_overlapping_tasks:
    return clarify_plan(...)
if any(task.depends_on for task in tasks):
    return sequential_plan(...)
if independent_cross_domain_tasks:
    return parallel_plan(...)
return single_plan(...)
```

- [ ] **Step 4: Re-run focused tests**

Run: `python -m unittest tests/test_task_planner.py`

### Task 2: Domain candidates and decomposition

**Files:**
- Modify: `core/intent_recognizer.py`
- Modify: `core/task_planner.py`
- Test: `tests/test_task_planner.py`

**Interfaces:**
- Produces: `IntentRecognizer.analyze_domains(message, history)` returning domain candidates and modifiers.
- Consumes: `TaskPlanner.plan(message, analysis)` and validates LLM-produced task evidence before calling `decide()`.

- [ ] **Step 1: Write a failing test for overlapping evidence producing clarification**

```python
plan = TaskPlanner().decide(message="报销系统退回了", candidates=two_candidates, tasks=overlapping_tasks)
assert plan.mode is ExecutionMode.CLARIFY
```

- [ ] **Step 2: Run focused tests and observe the missing validation branch**

Run: `python -m unittest tests/test_task_planner.py`

- [ ] **Step 3: Add four-domain candidate prompt, deterministic keyword support, task decomposition prompt, and evidence/dependency validation**

```python
analysis = await recognizer.analyze_domains(message, history)
plan = await planner.plan(message, analysis)
```

- [ ] **Step 4: Re-run focused tests**

Run: `python -m unittest tests/test_task_planner.py`

### Task 3: Orchestrator and clarification state

**Files:**
- Modify: `agents/agent_orchestrator.py`
- Modify: `memory/conversation_memory.py`
- Modify: `api/main.py`
- Test: `tests/test_task_planner.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Produces: `AgentOrchestrator.plan_request()`, `AgentOrchestrator.run_plan()`, and `OrchestratorResult.execution_mode`.
- Consumes: `MemoryManager.get_pending_clarification()`, `save_pending_clarification()`, and `clear_pending_clarification()`.

- [ ] **Step 1: Write failing orchestrator tests for parallel task execution and clarification response**

```python
result = await orchestrator.run_plan(request, parallel_plan)
assert result.execution_mode == "parallel"
assert "1." in result.response
```

- [ ] **Step 2: Run focused tests and observe missing `run_plan`**

Run: `python -m unittest tests/test_task_planner.py tests/test_agent_tools.py`

- [ ] **Step 3: Execute task plans and persist/restore clarification state through `/chat`**

```python
pending = await memory.get_pending_clarification(user_id, conv_id)
plan = await orchestrator.plan_request(request, pending=pending)
result = await orchestrator.run_plan(request, plan)
```

- [ ] **Step 4: Re-run focused tests and compile every changed module**

Run: `python -m unittest tests/test_task_planner.py tests/test_agent_tools.py`

Run: `python -m py_compile core/task_planner.py core/intent_recognizer.py agents/agent_orchestrator.py memory/conversation_memory.py api/main.py`

