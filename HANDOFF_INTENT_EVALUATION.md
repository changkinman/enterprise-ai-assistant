# EchoMind 意图识别改造与评测交接文档

## 当前状态

- 已准备脱敏评测材料：
  - `evaluation/fixtures/intent_cases.json`：100 条意图样本。
  - `evaluation/fixtures/rag_documents.json`、`rag_cases.json`：RAG 材料。
  - `evaluation/fixtures/memory_scenarios.json`：三级记忆材料。
  - `evaluation/fixtures/quality_rubric.md`：回答质量评分标准。
- 当前硬编码 Mock 工具不参与任何评测。
- 已完成一次规则改造前的真实意图分类基线：

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 100 |
| 完成数 / 错误数 | 100 / 0 |
| Accuracy | 81.0% |
| Macro-F1 | 0.8089 |
| 意图分类 P50 延迟 | 926ms |
| 意图分类 P95 延迟 | 1443ms |

此基线只测意图分类，不含实体抽取耗时。运行环境配置了 `ANTHROPIC_BASE_URL`，按现有代码逻辑会关闭 Embedding 分支，因此该基线实际为 **LLM + Pattern**，不能表述为完整三路融合结果。

## 已完成的规则改造与最新实测

以下改造已完成：

- `core/intent_recognizer.py` 已更新 Few-shot 示例、LLM 分类规则与 Billing 优先的 Pattern 兜底。
- `evaluation/evaluator.py` 的内置意图样本已替换为企业服务语料。
- `evaluation/fixtures/intent_cases.json` 已完成 Billing/Technical 相关样本的重新标注。
- 已新增 `tests/test_intent_rules.py`，覆盖 Billing 优先、Technical 与通用 Query/Request 的边界及 Account 兜底。

改造后在重标注的 100 条样本上进行真实模型测试（仅意图分类，不含实体抽取）：

| 指标 | 结果 |
| --- | ---: |
| 完成数 / 错误数 | 100 / 0 |
| Accuracy | 88.0% |
| Macro-F1 | 0.8552 |
| 分类 P50 延迟 | 977ms |
| 分类 P95 延迟 | 1755ms |
| Billing F1（18 条） | 0.9444 |
| Technical F1（15 条） | 0.8966 |

由于新规则改变了测试标签，改造后的 88.0% **不能**和旧口径下的 81.0% 直接表述为“提升 7 个百分点”。若需要严格 A/B 对比，必须以当前锁定标签为准，分别运行旧规则和新规则。

实体抽取在 8 条带标注样本上的结果：Micro-Precision 1.0000、Micro-Recall 0.8750、Micro-F1 0.9333；唯一遗漏为 `timeout` 错误码。该实体集规模过小，不建议单独作为简历量化结论。

## 新的分类规则

1. 只要涉及报销、发票、退款、扣款、薪资、费用金额、付款、补贴或费用审批进度，一律归 `billing`。
   - 即使是查询、请求、投诉、正向反馈或要求转人工，也归 `billing`。
2. `technical` 包含 IT 系统、VPN、网络、终端、企业邮箱、代码仓库、技术权限、登录故障和错误码相关的**查询或请求**。
3. `query` 和 `request` 仅处理与 `billing`、`technical`、`account` 无关的通用行政/服务类咨询和操作。
4. `account` 处理密码、手机号、身份验证器、账号锁定、注销和账户资料管理。
5. `complaint`、`escalation`、`feedback` 保持原有语义；技术问题的投诉式表达可保留为 `complaint`。
6. `billing` 具有最高优先级；但“差旅”一词本身不必然是费用问题：
   - “差旅住宿报销”“差旅费用”归 `billing`；
   - “提交下周差旅申请”可归 `request`。

## 待修改文件

### 1. `core/intent_recognizer.py`

#### 更新 Few-shot 模板

```python
IntentCategory.QUERY: [
    "年假余额在哪里查看？",
    "会议室最早能提前多久预约？",
    "新员工入职需要完成哪些流程？",
]
IntentCategory.REQUEST: [
    "帮我取消会议室预约",
    "我需要修改办公地点",
    "请协助申领显示器",
]
IntentCategory.TECHNICAL: [
    "VPN 无法连接",
    "门户出现401错误",
    "企业邮箱容量已满",
]
IntentCategory.BILLING: [
    "帮我提交餐费报销",
    "报销审核需要多久",
    "发票问题",
]
```

#### 更新 LLM Prompt

在 `_llm_recognize()` 的 Prompt 中加入完整分类规则，特别明确：

```text
1. 只要消息涉及报销、发票、退款、扣款、薪资、费用金额、付款或费用审批进度，
   一律分类为 billing；无论它是查询、请求、投诉、反馈或要求转人工。
2. IT 系统、VPN、网络、终端、企业邮箱、代码仓库、技术权限、登录故障或错误码
   相关的查询/请求，一律分类为 technical，而不是 query 或 request。
   密码、手机号、身份验证器、账号锁定等账户资料管理仍归 account。
3. query 仅用于与 billing、technical、account 无关的通用规则、流程或状态咨询；
   request 仅用于与 billing、technical、account 无关的通用操作请求。
```

#### 更新 Pattern 兜底

在 `_pattern_recognize()` 开头优先识别费用类：

```python
billing_kws = ["报销", "发票", "退款", "扣款", "费用", "薪资", "付款", "金额", "补贴"]
billing_hits = sum(1 for kw in billing_kws if kw in msg)
if billing_hits:
    return {
        "intent": IntentCategory.BILLING,
        "confidence": billing_hits / len(billing_kws),
    }
```

不要仅凭“差旅”归入 `billing`。将技术关键词排在 `query` / `request` 之前，建议包含：`崩溃`、`报错`、`错误`、`error`、`crash`、`vpn`、`网络`、`401`、`403`、`500`、`代码仓库`、`服务器`。

### 2. `evaluation/evaluator.py`

将 `DEFAULT_INTENT_CASES` 改为企业服务语料：

```python
DEFAULT_INTENT_CASES = [
    IntentTestCase("年假余额在哪里查看？", "query"),
    IntentTestCase("帮我取消 A302 会议室预约", "request"),
    IntentTestCase("你们服务太差了！", "complaint"),
    IntentTestCase("VPN 登录报 401 错误", "technical"),
    IntentTestCase("帮我提交本周餐费报销", "billing"),
    IntentTestCase("我要投诉，转人工！", "escalation"),
    IntentTestCase("你好", "greeting"),
    IntentTestCase("修改我的邮箱地址", "account"),
]
```

### 3. `evaluation/fixtures/intent_cases.json`

补全 `label_rules`，必须覆盖全部 10 类：`query`、`request`、`complaint`、`greeting`、`escalation`、`technical`、`billing`、`account`、`feedback`、`other`。

重新标注以下样本：

| ID | 原标签 | 新标签 |
| --- | --- | --- |
| I001 | query | billing |
| I002 | query | technical |
| I004 | query | technical |
| I006 | query | billing |
| I009 | query | billing |
| I010 | query | technical |
| I011 | request | billing |
| I012 | request | technical |
| I017 | request | technical |
| I018 | request | billing |
| I022 | complaint | billing |
| I029 | complaint | billing |
| I083 | feedback | billing |

保持以下标签：

- I019“请为下周出差提交差旅申请”仍为 `request`。
- I023“系统反复登录失败，根本没法用”仍为 `complaint`。
- I016“请协助重置我的企业邮箱密码”仍为 `account`。
- I020“麻烦更新我的紧急联系人电话”仍为 `account`。

### 4. 新增单元测试

测试不得调用真实模型，至少覆盖：

- “请转人工处理这笔报销” → `billing`
- “帮我提交餐费报销” → `billing`
- “VPN 权限申请需要找哪个部门？” → `technical`
- “帮我开通项目仓库权限” → `technical`
- “帮我取消 A302 会议室预约” → `request`
- “年假余额在哪里查看？” → `query`
- “系统反复登录失败，根本没法用” → `complaint`
- “忘记邮箱密码” → `account`

## 修改后的验证与测试

1. 校验 `intent_cases.json` 的 JSON 格式。
2. 运行 Python 语法校验和全部单元测试。
3. 运行 100 条真实意图样本的第二轮评测。
   - 意图分类与实体抽取分开执行，避免每样本重复模型调用。
   - 输出 Accuracy、Macro-F1、每类 Precision/Recall/F1、P50/P95。
4. 对比改造前基线，重点报告 `billing`、`query`、`request` 的变化。
5. 只有改造后、标签已锁定的结果才能用于最终简历。
