# Funnel Intent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route simple requests through deterministic or lightweight stages and reserve LLM task planning for ambiguous or composite messages.

**Architecture:** Add a pure `FunnelDecision` layer to `IntentRecognizer`, which produces a rule reply, a lightweight domain analysis, or an escalation signal. `AgentOrchestrator` consumes the signal and uses the existing `TaskPlanner` only on escalation.

**Tech Stack:** Python 3.12, Anthropic Async SDK, unittest.

**Spec:** `docs/superpowers/specs/2026-09-26-funnel-intent-routing-design.md`

## Global Constraints

- Preserve the existing legacy `recognize()` method for legacy evaluation.
- Preserve all four executable domains and the current `TaskPlan` API.
- Do not add dependencies.
- Never use a hard single-domain rule when a message has composite signals.

---

### Task 1: Pure funnel decision tests and model

**Files:**
- Modify: `core/intent_recognizer.py`
- Modify: `tests/test_domain_analysis.py`

**Interfaces:**
- Produces: `FunnelDecision` and `IntentRecognizer.lightweight_domain_analysis()`.
- Consumes: text only; no LLM client is required.

- [ ] **Step 1: Write failing tests for a rule early exit, multi-domain escalation and low-confidence escalation**

```python
assert recognizer.funnel_decision("你好").kind == "rule_reply"
assert recognizer.funnel_decision("VPN 登录失败，同时报销被退回").kind == "escalate"
assert recognizer.funnel_decision("帮我处理一下").kind == "escalate"
```

- [ ] **Step 2: Run tests and observe missing funnel interface**

Run: `python -m unittest tests/test_domain_analysis.py`

- [ ] **Step 3: Add lightweight candidate scoring and deterministic escalation rules**

```python
decision = recognizer.funnel_decision(message)
if decision.kind == "lightweight":
    return decision.analysis
```

- [ ] **Step 4: Re-run focused tests**

Run: `python -m unittest tests/test_domain_analysis.py`

### Task 2: Route orchestration through the funnel

**Files:**
- Modify: `agents/agent_orchestrator.py`
- Modify: `tests/test_task_orchestration.py`

**Interfaces:**
- Produces: `AgentOrchestrator.plan_request()` that skips LLM planning for rule and clear lightweight outcomes.
- Consumes: `FunnelDecision` and `TaskPlanner`.

- [ ] **Step 1: Write a failing test proving a clear lightweight single task skips the task planner**

```python
plan = await orchestrator.plan_request(Request("VPN 登录失败", "u", "c"))
assert planner.called is False
assert plan.mode is ExecutionMode.SINGLE
```

- [ ] **Step 2: Run the focused test and observe the legacy direct `analyze_domains()` call**

Run: `python -m unittest tests/test_task_orchestration.py`

- [ ] **Step 3: Call LLM candidate analysis and TaskPlanner only after a funnel escalation**

```python
decision = recognizer.funnel_decision(message)
if decision.kind == "lightweight":
    return planner.decide(message, decision.analysis.candidates, ())
analysis = await recognizer.analyze_domains(message, history)
return await planner.plan(message, analysis, history)
```

- [ ] **Step 4: Run the complete test suite and compile changed modules**

Run: `python -m unittest discover -s tests`

Run: `python -m py_compile core/intent_recognizer.py agents/agent_orchestrator.py`
