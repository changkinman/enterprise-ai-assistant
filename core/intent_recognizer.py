"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from core.task_planner import BusinessDomain, DomainAnalysis, DomainCandidate

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["年假余额在哪里查看？", "会议室最早能提前多久预约？", "新员工入职需要完成哪些流程？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消会议室预约", "我需要修改办公地点", "请协助申领显示器"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["VPN 无法连接", "门户出现401错误", "企业邮箱容量已满"],
    IntentCategory.BILLING:    ["帮我提交餐费报销", "报销审核需要多久", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class FunnelDecisionKind(str, Enum):
    """意图漏斗前两层的处理结果。"""

    DIRECT_REPLY = "direct_reply"
    LIGHTWEIGHT = "lightweight"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class FunnelDecision:
    """无需模型调用的预路由决定。"""

    kind: FunnelDecisionKind
    analysis: Optional[DomainAnalysis] = None
    reply: str = ""
    reason: str = ""


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # 第三方兼容 API（如 DeepSeek）通常不支持 Embedding，禁用该策略。
        # 官方 Anthropic SDK 当前没有 embeddings 资源，因此下面会使用稳定的
        # 本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def funnel_decision(self, message: str) -> FunnelDecision:
        """运行规则层和轻量候选层，决定是否值得升级到 LLM。

        这里的分数只用于决定能否安全地走单域路径；一旦出现多领域、连接词、
        长句或分差过小，都会升级到 ``analyze_domains``，从而保留多意图能力。
        """
        clean_message = self._clean_text(message).strip()
        normalized = clean_message.lower().rstrip("。！？!?~")
        greeting_replies = {
            "你好": "你好！请告诉我需要查询或处理的企业服务问题。",
            "您好": "您好！请告诉我需要查询或处理的企业服务问题。",
            "嗨": "你好！请告诉我需要查询或处理的企业服务问题。",
            "hi": "Hi！请告诉我需要查询或处理的企业服务问题。",
            "hello": "Hello！请告诉我需要查询或处理的企业服务问题。",
        }
        if normalized in greeting_replies:
            return FunnelDecision(
                FunnelDecisionKind.DIRECT_REPLY,
                reply=greeting_replies[normalized],
                reason="精确寒暄规则",
            )

        analysis = self.lightweight_domain_analysis(clean_message)
        credible = [item for item in analysis.candidates if item.score >= 0.55]
        if not credible:
            return FunnelDecision(FunnelDecisionKind.ESCALATE, analysis, reason="轻量候选置信度低")

        # 领域候选一旦超过一个，必须由 LLM 根据原文证据判断是否真的是多个任务。
        if len(credible) > 1:
            return FunnelDecision(FunnelDecisionKind.ESCALATE, analysis, reason="存在多个可信业务领域")

        task_connectors = ("同时", "另外", "并且", "以及", "先", "再", "然后", "之后", "如果", "若")
        if any(connector in clean_message for connector in task_connectors):
            return FunnelDecision(FunnelDecisionKind.ESCALATE, analysis, reason="消息含任务连接或条件关系")
        if len(clean_message) > 80:
            return FunnelDecision(FunnelDecisionKind.ESCALATE, analysis, reason="消息较长，需理解完整上下文")

        top = credible[0]
        runner_up = next((item for item in analysis.candidates if item.domain != top.domain), None)
        if runner_up and top.score - runner_up.score < 0.15:
            return FunnelDecision(FunnelDecisionKind.ESCALATE, analysis, reason="候选领域分差过小")
        return FunnelDecision(FunnelDecisionKind.LIGHTWEIGHT, analysis, reason="单一高置信领域")

    def lightweight_domain_analysis(self, message: str) -> DomainAnalysis:
        """用关键词模板与本地字符 n-gram 返回四个领域的候选。

        该方法完全本地运行，既不访问模型，也不依赖 ``IntentRecognizer`` 的
        初始化状态，因而可作为低成本、可测试的漏斗第二层。
        """
        text = self._clean_text(message).strip()
        msg_lower = text.lower()
        keyword_patterns = {
            BusinessDomain.TECHNICAL: [
                "vpn", "网络", "报错", "错误", "崩溃", "401", "403", "500", "服务器",
                "系统", "终端", "软件", "电脑", "无法连接",
            ],
            BusinessDomain.BILLING: [
                "报销", "发票", "退款", "扣款", "费用", "薪资", "付款", "金额", "审核", "到账",
            ],
            BusinessDomain.ACCOUNT: [
                "密码", "手机号", "身份验证", "验证器", "账号锁定", "注销", "登录名", "账号", "账户", "重置",
            ],
            BusinessDomain.GENERAL: [
                "制度", "年假", "会议室", "入职", "办公地点", "流程", "政策", "福利", "审批",
            ],
        }
        templates = {
            BusinessDomain.TECHNICAL: _TEMPLATES[IntentCategory.TECHNICAL],
            BusinessDomain.BILLING: _TEMPLATES[IntentCategory.BILLING],
            BusinessDomain.ACCOUNT: _TEMPLATES[IntentCategory.ACCOUNT],
            BusinessDomain.GENERAL: _TEMPLATES[IntentCategory.QUERY] + _TEMPLATES[IntentCategory.REQUEST],
        }
        message_vector = self._local_embedding(msg_lower)
        candidates: List[DomainCandidate] = []
        for domain in BusinessDomain:
            keywords = keyword_patterns[domain]
            hits = [keyword for keyword in keywords if keyword in msg_lower]
            keyword_score = min(0.95, 0.55 + 0.10 * len(hits)) if hits else 0.0
            ngram_similarity = max(
                (_cosine(message_vector, self._local_embedding(template)) for template in templates[domain]),
                default=0.0,
            )
            # 仅在与领域模板高度相似时让 n-gram 独立形成候选，避免哈希碰撞放大噪声。
            template_score = 0.40 + 0.60 * ngram_similarity if ngram_similarity >= 0.72 else 0.0
            score = max(keyword_score, template_score)
            # 轻量单域路径会把 evidence 直接传给 Agent，因此必须保留完整原文，
            # 不能只传递命中的几个关键词而丢失问题、对象和限定条件。
            evidence = text if (hits or template_score) else ""
            candidates.append(DomainCandidate(domain, min(1.0, score), evidence))

        candidates.sort(key=lambda item: item.score, reverse=True)
        return DomainAnalysis(
            candidates=tuple(candidates),
            modifiers={
                "source": "lightweight_funnel",
                "urgency": self._urgency(text, IntentCategory.OTHER).name.lower(),
            },
        )

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent = self._vote(llm, emb, pat)
        entities = await self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=llm["confidence"],
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    async def analyze_domains(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> DomainAnalysis:
        """返回四个可执行业务领域的候选，而不是过早收敛为单一意图。"""
        llm_candidates = await self._llm_domain_candidates(message, history)
        pattern_scores = self._pattern_domain_scores(message)

        candidates: List[DomainCandidate] = []
        for domain in BusinessDomain:
            llm = llm_candidates.get(domain, DomainCandidate(domain, 0.0, ""))
            score = max(llm.score, pattern_scores.get(domain, 0.0))
            candidates.append(DomainCandidate(
                domain=domain,
                score=max(0.0, min(1.0, score)),
                evidence=llm.evidence,
            ))

        candidates.sort(key=lambda item: item.score, reverse=True)
        urgency = self._urgency(message, IntentCategory.OTHER)
        message_lower = message.lower()
        return DomainAnalysis(
            candidates=tuple(candidates),
            modifiers={
                "urgency": urgency.name.lower() if urgency else "low",
                "complaint": any(word in message_lower for word in ["投诉", "太差", "糟糕", "不满"]),
                "escalation_requested": any(word in message_lower for word in ["转人工", "找经理", "升级处理"]),
            },
        )

    async def _llm_domain_candidates(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[BusinessDomain, DomainCandidate]:
        """让 LLM 保留多领域候选及其对应的用户原文证据。"""
        safe_message = self._clean_text(message)
        history_text = "\n".join(
            f"{self._clean_text(item.get('role', 'user'))}: {self._clean_text(item.get('content', ''))}"
            for item in (history or [])[-3:]
        )
        prompt = f"""你是企业服务领域识别器。只评估四个可执行领域：
- general：制度、通用流程、非技术非财务非账户服务
- technical：IT 系统、VPN、网络、终端、错误码和技术权限
- billing：报销、发票、费用、付款和财务审批
- account：密码、身份验证、账户资料和账号锁定

最近对话：
{history_text}
用户消息：{safe_message}

返回严格 JSON：
{{"candidates":[{{"domain":"technical","score":0.0,"evidence":"用户原文中的连续片段"}}]}}
必须返回四个领域各一项，score 范围为 0 到 1。evidence 为空字符串表示该领域不相关。不要把投诉、紧急、转人工当作独立领域。"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=384,
                temperature=0.0,
                messages=[{"role": "user", "content": self._clean_text(prompt)}],
            )
            raw = response.content[0].text
            start, end = raw.find("{"), raw.rfind("}") + 1
            payload = json.loads(raw[start:end])
            items = payload.get("candidates", []) if isinstance(payload, dict) else []
            result: Dict[BusinessDomain, DomainCandidate] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    domain = BusinessDomain(str(item.get("domain", "")))
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    continue
                result[domain] = DomainCandidate(
                    domain,
                    max(0.0, min(1.0, score)),
                    self._clean_text(item.get("evidence", "")).strip(),
                )
            return result
        except Exception as ex:
            logger.warning(f"多领域识别失败: {ex}")
            return {}

    @staticmethod
    def _pattern_domain_scores(message: str) -> Dict[BusinessDomain, float]:
        """确定性关键词仅作为候选补充，避免单一关键词直接决定最终路由。"""
        msg = (message or "").lower()
        patterns = {
            BusinessDomain.TECHNICAL: ["vpn", "网络", "报错", "错误", "崩溃", "401", "403", "500", "服务器"],
            BusinessDomain.BILLING: ["报销", "发票", "退款", "扣款", "费用", "薪资", "付款", "金额"],
            BusinessDomain.ACCOUNT: ["密码", "手机号", "身份验证", "验证器", "账号锁定", "注销", "登录名"],
            BusinessDomain.GENERAL: ["制度", "年假", "会议室", "入职", "办公地点", "流程", "政策"],
        }
        return {
            domain: min(0.95, 0.55 + 0.10 * sum(keyword in msg for keyword in keywords))
            if any(keyword in msg for keyword in keywords) else 0.0
            for domain, keywords in patterns.items()
        }

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是企业内部服务的意图分析专家。根据示例和以下分类规则判断用户意图，返回 JSON。

分类规则与优先级：
1. 只要消息涉及报销、发票、退款、扣款、薪资、费用金额、付款或费用审批进度，一律分类为 billing；无论它是查询、请求、投诉、反馈或要求转人工。
2. IT 系统、VPN、网络、终端、企业邮箱、代码仓库、技术权限、登录故障或错误码相关的查询/请求，一律分类为 technical，而不是 query 或 request。密码、手机号、身份验证器、账号锁定等账户资料管理仍归 account。
3. query 仅用于与 billing、technical、account 无关的通用规则、流程或状态咨询；request 仅用于与 billing、technical、account 无关的通用操作请求。
4. escalation 用于明确要求转人工、找主管或升级处理；complaint 用于表达不满但未明确升级；feedback 用于正面评价；greeting 仅用于寒暄；other 用于无关内容。

示例:
{examples}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        # 费用类表达按产品规则拥有最高优先级，避免被通用 request/escalation 关键词截获。
        billing_kws = ["报销", "发票", "退款", "扣款", "费用", "薪资", "付款", "金额", "补贴"]
        billing_hits = sum(1 for kw in billing_kws if kw in msg)
        if billing_hits:
            return {"intent": IntentCategory.BILLING, "confidence": billing_hits / len(billing_kws)}

        account_kws = ["密码", "手机号", "身份验证", "验证器", "账号锁定", "注销", "登录名"]
        if any(kw in msg for kw in account_kws):
            return {"intent": IntentCategory.ACCOUNT, "confidence": 1.0 / len(account_kws)}

        technical_kws = ["崩溃", "报错", "错误", "error", "crash", "vpn", "网络", "401", "403", "500", "代码仓库", "服务器"]
        complaint_or_escalation = ["投诉", "太差", "糟糕", "转人工", "经理"]
        if any(kw in msg for kw in technical_kws) and not any(kw in msg for kw in complaint_or_escalation):
            return {"intent": IntentCategory.TECHNICAL, "confidence": 1.0 / len(technical_kws)}

        patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "转人工", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "错误", "error", "crash", "vpn", "网络", "401", "403", "500", "代码仓库", "服务器"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> IntentCategory:
        """加权投票。embedding 不可用时权重自动转移到 LLM 和 Pattern。"""
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"]
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"]
            return IntentCategory.OTHER

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentCategory.OTHER

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用 LLM 从消息中提取结构化实体。"""
        message = self._clean_text(message)
        prompt = f"""从客服消息中提取实体，返回 JSON（字段值为列表，没有则为空列表）:
消息: "{message}"
格式: {{"order_id":[],"product":[],"date":[],"amount":[],"error_code":[]}}"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {"order_id": [], "product": [], "date": [], "amount": [], "error_code": []}

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str) -> str:
        return self._clean_text(message)[:200]

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
