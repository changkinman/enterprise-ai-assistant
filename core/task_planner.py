"""任务级多意图规划：把领域候选转换为可执行、可解释的任务计划。"""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class BusinessDomain(str, Enum):
    """当前允许路由到领域 Agent 的业务域。"""

    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"


class ExecutionMode(str, Enum):
    DIRECT = "direct"
    SINGLE = "single"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class DomainCandidate:
    domain: BusinessDomain
    score: float
    evidence: str = ""


@dataclass(frozen=True)
class DomainAnalysis:
    candidates: Tuple[DomainCandidate, ...]
    modifiers: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    domain: BusinessDomain
    evidence: str
    entity: str
    action: str
    confidence: float
    depends_on: Tuple[str, ...] = ()
    modifiers: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    mode: ExecutionMode
    tasks: Tuple[PlannedTask, ...] = ()
    clarification_question: str = ""
    reason: str = ""
    direct_response: str = ""

    def to_pending_state(self, original_message: str) -> Dict[str, Any]:
        """序列化澄清所需的最小状态，供工作记忆短期保存。"""
        return {
            "original_message": original_message,
            "clarification_question": self.clarification_question,
            "reason": self.reason,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "domain": task.domain.value,
                    "evidence": task.evidence,
                    "entity": task.entity,
                    "action": task.action,
                    "confidence": task.confidence,
                    "depends_on": list(task.depends_on),
                    "modifiers": task.modifiers,
                }
                for task in self.tasks
            ],
        }


class TaskPlanner:
    """
    根据候选领域和任务草案决定执行模式。

    分数仅用来判断是否有可执行的领域候选；是否并行取决于可验证的
    任务证据、依赖关系和领域独立性。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        client: Optional[Any] = None,
        candidate_floor: float = 0.55,
        task_floor: float = 0.60,
    ):
        self._client = client
        if self._client is None and api_key:
            from anthropic import AsyncAnthropic

            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = AsyncAnthropic(**kwargs)
        self._model = model
        self._candidate_floor = candidate_floor
        self._task_floor = task_floor

    def decide(
        self,
        message: str,
        candidates: Sequence[DomainCandidate],
        tasks: Sequence[PlannedTask],
    ) -> TaskPlan:
        """在不依赖模型调用的情况下验证任务并决定执行模式。"""
        credible = [candidate for candidate in candidates if candidate.score >= self._candidate_floor]
        if not credible:
            return self._clarify("未找到可信的业务领域", "请说明你需要查询或处理的具体业务问题。")

        if not tasks:
            if len(credible) == 1:
                candidate = credible[0]
                task = PlannedTask(
                    task_id="t1",
                    domain=candidate.domain,
                    evidence=(candidate.evidence or message).strip(),
                    entity=(candidate.evidence or "当前问题").strip(),
                    action="处理请求",
                    confidence=candidate.score,
                )
                return TaskPlan(ExecutionMode.SINGLE, (task,), reason="只有一个可信领域")
            return self._clarify("多个领域候选但无法拆分任务", self._ambiguous_question(message, credible))

        if not self._tasks_are_valid(message, tasks, credible):
            return self._clarify("任务证据、领域或依赖关系无效", self._ambiguous_question(message, credible))

        if self._has_overlapping_evidence(tasks):
            return self._clarify("多个领域共享同一证据片段", self._ambiguous_question(message, credible))

        ordered = tuple(tasks)
        if len(ordered) == 1:
            return TaskPlan(ExecutionMode.SINGLE, ordered, reason="只有一个有效任务")

        if any(task.depends_on for task in ordered):
            return TaskPlan(ExecutionMode.SEQUENTIAL, ordered, reason="任务存在前置依赖")

        domains = {task.domain for task in ordered}
        if len(domains) == len(ordered):
            return TaskPlan(ExecutionMode.PARALLEL, ordered, reason="跨领域任务证据独立且无依赖")

        return TaskPlan(ExecutionMode.SEQUENTIAL, ordered, reason="同领域任务暂不并行")

    async def plan(
        self,
        message: str,
        analysis: DomainAnalysis,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> TaskPlan:
        """复杂候选请求由模型拆分任务；简单单领域请求不额外调用模型。"""
        credible = [candidate for candidate in analysis.candidates if candidate.score >= self._candidate_floor]
        if len(credible) <= 1:
            return self.decide(message, analysis.candidates, ())

        tasks = await self._decompose(message, credible, analysis.modifiers, history)
        return self.decide(message, analysis.candidates, tasks)

    async def _decompose(
        self,
        message: str,
        candidates: Sequence[DomainCandidate],
        modifiers: Dict[str, Any],
        history: Optional[List[Dict[str, str]]],
    ) -> List[PlannedTask]:
        if self._client is None:
            return []

        candidate_text = ", ".join(f"{item.domain.value}:{item.score:.2f}" for item in candidates)
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in (history or [])[-4:]
        )
        prompt = f"""你是企业服务任务拆分器。只可使用领域 general、technical、billing、account。
候选领域：{candidate_text}
最近对话：{history_text}
用户消息：{message}

将消息拆为独立任务。每个 evidence 必须是用户消息中的连续原文；若两个候选只是同一件事的不同解释，不要拆成两个任务，返回空 tasks。
若任务 B 依赖任务 A 的结果，在 depends_on 填 A 的 task_id。
只返回 JSON：
{{"tasks":[{{"task_id":"t1","domain":"technical","evidence":"原文","entity":"主体","action":"动作","confidence":0.0,"depends_on":[]}}]}}"""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            start, end = raw.find("{"), raw.rfind("}") + 1
            payload = json.loads(raw[start:end])
            return self._tasks_from_payload(payload, modifiers)
        except Exception:
            return []

    @staticmethod
    def _tasks_from_payload(payload: Any, modifiers: Dict[str, Any]) -> List[PlannedTask]:
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            return []
        tasks: List[PlannedTask] = []
        for index, item in enumerate(payload["tasks"], start=1):
            if not isinstance(item, dict):
                return []
            try:
                domain = BusinessDomain(str(item.get("domain", "")))
                dependencies = item.get("depends_on", [])
                if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
                    return []
                tasks.append(PlannedTask(
                    task_id=str(item.get("task_id") or f"t{index}"),
                    domain=domain,
                    evidence=str(item.get("evidence", "")).strip(),
                    entity=str(item.get("entity", "")).strip(),
                    action=str(item.get("action", "")).strip(),
                    confidence=float(item.get("confidence", 0.0)),
                    depends_on=tuple(dependencies),
                    modifiers=dict(modifiers),
                ))
            except (TypeError, ValueError):
                return []
        return tasks

    def _tasks_are_valid(
        self,
        message: str,
        tasks: Sequence[PlannedTask],
        credible: Sequence[DomainCandidate],
    ) -> bool:
        credible_domains = {candidate.domain for candidate in credible}
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            return False
        for task in tasks:
            if (
                task.domain not in credible_domains
                or task.confidence < self._task_floor
                or not task.evidence
                or task.evidence not in message
                or not task.entity
                or not task.action
                or task.task_id in task.depends_on
                or any(dependency not in task_ids for dependency in task.depends_on)
            ):
                return False
        return not self._has_dependency_cycle(tasks)

    @staticmethod
    def _has_overlapping_evidence(tasks: Sequence[PlannedTask]) -> bool:
        for index, current in enumerate(tasks):
            for other in tasks[index + 1:]:
                if current.evidence in other.evidence or other.evidence in current.evidence:
                    return True
        return False

    @staticmethod
    def _has_dependency_cycle(tasks: Sequence[PlannedTask]) -> bool:
        graph = {task.task_id: set(task.depends_on) for task in tasks}
        visited: set[str] = set()
        active: set[str] = set()

        def visit(task_id: str) -> bool:
            if task_id in active:
                return True
            if task_id in visited:
                return False
            active.add(task_id)
            cyclic = any(visit(dependency) for dependency in graph[task_id])
            active.remove(task_id)
            visited.add(task_id)
            return cyclic

        return any(visit(task_id) for task_id in graph)

    @staticmethod
    def _clarify(reason: str, question: str) -> TaskPlan:
        return TaskPlan(ExecutionMode.CLARIFY, clarification_question=question, reason=reason)

    @staticmethod
    def _ambiguous_question(message: str, candidates: Iterable[DomainCandidate]) -> str:
        labels = {
            BusinessDomain.GENERAL: "制度或通用服务",
            BusinessDomain.TECHNICAL: "IT 系统问题",
            BusinessDomain.BILLING: "财务或报销问题",
            BusinessDomain.ACCOUNT: "账户管理问题",
        }
        options = "、".join(labels[candidate.domain] for candidate in candidates)
        subject = message.strip()[:80] or "当前问题"
        return f"关于“{subject}”，你希望处理的是哪一类问题：{options}？请补充具体对象和希望完成的操作。"
