# EchoMind 项目详细介绍

> 本文依据仓库当前实现整理，面向项目交接、技术复盘和面试讲解。它描述的是已经存在的代码行为；`mcp/` 目录是项目内部的工具治理模块名称，**并非标准 Model Context Protocol（MCP）Server/Client 协议实现**。

## 1. 项目定位

EchoMind 是一个面向企业内部服务场景的异步智能助手。它处理制度与流程咨询、IT 故障支持、费用报销等自然语言请求，并将意图分析、对话记忆、知识检索、领域 Agent、工具调用、监控和评测组合为可观测的服务链路。

系统为知识咨询和受控操作提供自助入口。当前 JWT 只保护操作预览与确认接口，`/chat` 的 `user_id` 仍由请求体提供，适合演示；报销与权限申请的业务数据仍在进程内，未接入外部财务系统或 IAM。操作预览、幂等记录和审计事件由 PostgreSQL 保存；真实企业接入还需完善身份体系、访问控制、数据治理与脱敏。

## 2. 技术架构与模块边界

```text
客户端
  │ HTTP
  ▼
FastAPI（api/main.py）
  │
  ├─ MemoryManager ─────────── Redis：工作记忆、摘要、澄清状态
  │         └──────────────── PostgreSQL/pgvector：情景记忆、用户画像
  │
  ├─ MCPToolManager ────────── KnowledgeBase（PostgreSQL/pgvector）
  │         └──────────────── 工具：权限、校验、缓存、超时、熔断、降级
  │
  ├─ AgentOrchestrator
  │     ├─ IntentRecognizer：漏斗、领域分析、意图识别
  │     ├─ TaskPlanner：任务拆分校验与执行模式决策
  │     └─ General / Technical / Billing Agent
  ├─ SideEffectExecutor ──── 预检、预览、JWT 确认、幂等与审计
  │
  ├─ EndToEndEvaluator：离线与回归评测
  └─ PerformanceMonitor：Prometheus、异常检测、路由反馈
          │
          ├─ Anthropic SDK / 兼容 LLM API
          └─ Prometheus / 可选 Webhook 告警
```

| 模块 | 主要入口 | 职责 | 关键依赖 |
| --- | --- | --- | --- |
| API 与装配 | `api/main.py` | 生命周期、HTTP 校验、上下文拼装、响应序列化 | FastAPI、全部核心组件 |
| 意图与领域分析 | `core/intent_recognizer.py` | 精确寒暄漏斗、轻量领域评分、LLM 多领域候选、传统意图识别 | LLM、本地字符 n-gram |
| 任务规划 | `core/task_planner.py` | LLM 任务草案、证据与依赖校验、单路/并行/串行/澄清决策 | LLM、领域候选 |
| Agent 编排 | `agents/agent_orchestrator.py` | Agent 选择、工具调用循环、失败降级、结果聚合 | 意图分析、任务规划、工具管理器 |
| 对话记忆 | `memory/conversation_memory.py` | 工作记忆、跨会话情景记忆、用户画像、摘要压缩 | Redis、PostgreSQL/pgvector、LLM |
| 知识检索 | `mcp/knowledge_base.py` | 文档切分、入库、向量检索 | PostgreSQL/pgvector |
| 人工确认 | `core/side_effect_executor.py` | 预检、预览、确认、幂等及审计 | JWT、PostgresOperationRepository、操作适配器 |
| 工具治理 | `mcp/tool_manager.py` | 工具注册、权限、Schema 校验、缓存、超时、熔断、检索优化 | LLM、工具 Handler |
| 在线监控 | `monitor/performance_monitor.py` | 统计收集、阈值和 Z-score 异常、告警、路由惩罚 | Prometheus、编排器、工具管理器 |
| 评测 | `evaluation/evaluator.py` | 意图 Accuracy/F1、LLM-as-Judge、多轮对话、回归比较 | 编排器、识别器、LLM |

## 3. 启动与依赖装配

FastAPI 使用 `lifespan` 在进程启动时装配单例组件，而不是在每个请求中重新建立客户端：

1. 从 `.env` 读取模型配置。`ANTHROPIC_API_KEY` 是必需项；`ANTHROPIC_BASE_URL` 非空时，Anthropic SDK 会请求兼容接口，实际模型不一定是 Claude。
2. 连接 PostgreSQL 并构造显式 Embedding provider 与共享 VectorStore；创建使用 Redis 工作记忆和 PostgreSQL/pgvector 情景记忆、画像的 `MemoryManager`。数据库连接失败会阻止启动，没有 ChromaDB 本地回退。
3. 创建 `MCPToolManager` 和 `KnowledgeBase`，注册 `knowledge_search`，再通过 `tools.loader.discover_agent_tools()` 发现扩展工具。
4. 创建 PostgreSQL 操作仓库、`SideEffectExecutor`、报销与权限申请适配器，再创建 `AgentOrchestrator`，让各 Agent 获得已授权工具定义。
5. 创建并启动 `PerformanceMonitor`，然后创建 `EndToEndEvaluator`。
6. 关闭服务时停止监控后台任务并关闭 PostgreSQL 连接池。

`knowledge_search` 被注册为受治理工具：它有 JSON Schema、300 秒缓存、可重排标记和失败降级结果。Agent 不能绕过 `MCPToolManager.call()` 直接调用 Handler。

## 4. 主请求链路：`POST /chat`

```text
ChatRequest(message, user_id, conv_id)
  → 读取三级记忆
  → 预取 RAG 知识上下文
  → 构造 Request(message, history, context)
  → 读取可能存在的待澄清状态
  → 生成 TaskPlan
  → 按 SINGLE / PARALLEL / SEQUENTIAL / CLARIFY 执行
  → 写入用户消息与助手回复
  → 异步更新用户画像
  → 返回 ChatResponse 与工具调用轨迹
```

### 4.1 上下文构造

`/chat` 首先调用 `MemoryManager.get_context(user_id, conv_id, query)`，得到：

- 当前会话最近消息；
- 与当前 query 语义相关的跨会话历史；
- 当前用户画像；
- 已压缩的会话摘要。

这些信息经过 `MemoryContext.to_prompt_text()` 组织成 Prompt 文本。随后 `_build_knowledge_context()` 判断是否需要 RAG：精确寒暄不会检索，其他长度不少于四个字符或包含业务关键词的请求会触发 `search_with_rewrite()`。最终上下文是“记忆上下文 + 知识库 Top-3 片段”。

RAG 预取失败不会中断聊天请求：方法捕获异常并返回空知识上下文，Agent 仍可基于会话上下文和通用能力作答。

### 4.2 澄清状态的续接

当任务规划结论为 `CLARIFY` 时，API 将 `TaskPlan.to_pending_state()` 生成的最小状态写入 Redis，默认有效期 600 秒。下一轮用户补充会将之前的原始问题与当前补充拼接，再重新执行完整领域分析和规划；成功执行后会清除该状态。

### 4.3 响应写回

编排完成后，用户原问题和最终回复都会写入工作记忆。用户画像更新通过 `asyncio.create_task()` 在后台执行，不阻塞 HTTP 响应。API 返回内容包括：会话 ID、最终文本、主要领域映射出的意图、首个 Agent 类型、升级标志、执行模式、知识库使用标志和完整工具调用轨迹。

## 5. 请求理解、任务规划与路由

### 5.1 两条并存的识别能力

`IntentRecognizer` 同时存在两套能力，应避免混为一谈：

| 能力 | 使用场景 | 主要输出 |
| --- | --- | --- |
| `recognize()` | 独立意图识别和评测 | 十类 `IntentCategory`、置信度、实体、紧急度 |
| `funnel_decision()` + `analyze_domains()` | 当前 `/chat` 的任务规划前置路径 | 四类可执行业务领域候选和规划入口 |

`recognize()` 的 LLM Few-shot、Embedding 相似度和关键词 Pattern 会加权投票：Embedding 可用时权重为 0.7 / 0.2 / 0.1；兼容第三方 Base URL 时关闭 Embedding，权重改为 LLM 0.85、规则 0.15。之后还会单独调用 LLM 提取订单号、产品、日期、金额和错误码等实体。

当前 `/chat` 的编排入口则优先走“漏斗 + 领域规划”，不会先调用 `recognize()` 再拆任务。

### 5.2 三段式漏斗

```text
精确寒暄
  └─ DIRECT_REPLY：直接返回固定欢迎语

轻量领域分析（关键词 + 本地 n-gram）
  └─ LIGHTWEIGHT：直接构造单任务

复杂请求
  └─ ESCALATE：进入 LLM 多领域分析与可能的任务拆分
```

第一层只匹配精确的“你好、您好、嗨、hi、hello”，命中后不调用 Agent。第二层为 `general`、`technical`、`billing`、`account` 四个领域打分。关键词命中可形成 0.55 起的候选分数；本地字符 n-gram 与领域模板相似度达到 0.72 时也可以形成候选。

只有同时满足以下条件，才走轻量单任务路径：

- 至少一个候选分数不低于 0.55；
- 可信候选只有一个；
- 文本不含“同时、另外、并且、以及、先、再、然后、之后、如果、若”等连接词；
- 文本长度不超过 80 个字符；
- 第一候选相对第二候选的分差不低于 0.15。

任一条件不满足时，`ESCALATE` 的含义是升级到 LLM 深度分析，并不等同于转人工。

### 5.3 多领域分析与任务草案

复杂请求会调用 LLM，为四个业务领域各输出一项 `{domain, score, evidence}`。`evidence` 被要求为用户原文中的连续片段；关键词评分只作为候选补充，最终每个领域分数取 LLM 分数和规则分数的较大值。

当可信领域不超过一个时，规划器直接生成一个 `PlannedTask`。当有多个可信领域时，`TaskPlanner._decompose()` 才请求 LLM 生成任务列表。任务字段为：

```json
{
  "task_id": "t1",
  "domain": "technical",
  "evidence": "用户原文中的连续片段",
  "entity": "处理对象",
  "action": "要执行的动作",
  "confidence": 0.0,
  "depends_on": []
}
```

### 5.4 为什么由代码决定执行模式

LLM 负责理解语义和提出依赖草案；代码负责将草案变为可安全执行的计划。`TaskPlanner.decide()` 会校验：

- 任务领域是否属于可信候选；
- 任务置信度是否达到 0.60；
- `evidence` 是否非空且是用户原文的子串；
- `entity`、`action` 是否非空；
- 任务 ID 是否重复；
- 依赖是否指向存在的任务、是否自依赖、是否存在环；
- 不同任务的 `evidence` 是否存在包含关系。

校验失败或证据重叠时，系统返回澄清问题，而不是删除可疑任务继续执行。校验成功后按以下规则选择执行模式：

| 条件 | 模式 | 行为 |
| --- | --- | --- |
| 只有一个有效任务 | `SINGLE` | 路由一个 Agent |
| 任一任务具有前置依赖 | `SEQUENTIAL` | 按依赖拓扑顺序执行 |
| 跨领域、无依赖的多个任务 | `PARALLEL` | 使用 `asyncio.gather()` 并发执行 |
| 同领域多任务且无依赖 | `SEQUENTIAL` | 保守串行 |

当前重叠判断是字符串包含关系，不是实体级或语义级相似度判断；这是一项可继续增强的边界。

### 5.5 Agent 选择与结果聚合

每个任务会携带专属的 `[当前任务]` 上下文，避免多个任务的证据互相污染。领域到 Agent 的当前映射是 General → GeneralAgent、Technical → TechnicalAgent、Billing → BillingAgent、Account → GeneralAgent。

同类 Agent 若存在多个实例，编排器按运行评分选取最高者：

```text
routing_score = (成功率 × 0.7 + 延迟得分 × 0.3) × (1 - 监控惩罚)
延迟得分 = 1 / (1 + 平均延迟秒数)
```

专业 Agent 调用失败会降级到 GeneralAgent。并行任务的 `asyncio.gather()` 保留任务计划顺序；串行任务会将前置任务结果加入后置任务上下文。最终多任务文本采用编号拼接，而不是再调用一个汇总 LLM：

```text
1. 第一个任务的 Agent 回复

2. 第二个任务的 Agent 回复
```

工具调用轨迹会被展平合并；只要任一任务要求升级，最终 `escalated` 为真。当前 API 的 `agent_type` 和 `intent` 字段取首个任务的结果，因此多任务的完整执行信息应结合 `execution_mode` 与回复编号理解。

## 6. Agent 与受治理 Tool Calling

每个领域 Agent 继承 `BaseAgent`，使用 Anthropic Messages 风格的工具调用循环：

```text
构造 messages（上下文 + 当前问题）
  → 请求 LLM
  → 若无 tool_use：返回文本
  → 若有 tool_use：并发执行同一轮全部工具
  → 将 tool_result 回填给 LLM
  → 继续下一轮，最多 MAX_TOOL_ROUNDS（默认 5）
```

达到工具调用轮数上限后，系统再次请求模型，并强制其只基于已有结果输出文本。每一次工具调用记录 `round`、调用 ID、工具名、参数、成功状态、结果、错误和耗时，最终通过 `/chat` 返回。

`MCPToolManager` 为工具调用提供以下治理能力：

| 能力 | 实现语义 |
| --- | --- |
| 注册校验 | 名称、描述、Handler、Object JSON Schema、必填字段和 Agent 白名单均需合法 |
| Agent 授权 | 仅暴露/允许已在 `allowed_agents` 中注册的工具 |
| 参数校验 | 根据 Schema 的 `required` 和字段类型做基础校验 |
| 缓存 | 工具可配置 TTL；`knowledge_search` 为 300 秒 |
| 超时 | 用 `asyncio.wait_for()` 包裹 Handler，默认 30 秒 |
| 熔断 | 连续失败 5 次打开，60 秒后进入半开探测 |
| 降级 | 工具失败时返回工具定义的 fallback，而非直接抛出空错误 |

## 7. RAG 知识库与检索优化

### 7.1 写入与集合

`KnowledgeBase` 使用共享 `VectorStore` 与 PostgreSQL/pgvector。知识文档通过 `/knowledge/add` 或 `/knowledge/upload` 写入：

1. `.txt`、`.md` 或 JSON 文档被读取；上传大小限制为 10MB；
2. 长文本按句号/换行切分为最多约 500 字的片段；
3. 每个片段以标题、分片序号和总片数作为 metadata 写入 PostgreSQL；
4. 显式 `SentenceTransformerEmbeddingProvider` 按 `EMBEDDING_MODEL` 生成向量；默认维度为 512，与初始化表结构对应。

普通 `search()` 返回标题、片段内容、由距离转换而来的相似度和分片序号。

### 7.2 增强检索链路

`MCPToolManager.search_with_rewrite()` 用于 `/chat` 的知识预取和 `/search` 接口：

```text
原始查询
  → LLM 改写为 3 个不同角度的子查询，并保留原始查询
  → 对最多 4 个查询并行调用 knowledge_search
  → 按结果对象哈希合并去重
  → LLM 根据原始查询重排候选
  → 返回 Top-K
```

查询改写解决员工措辞与制度文本措辞不一致、或问题包含多个隐含角度的问题；并行召回增加候选覆盖；去重避免重复片段挤占候选；重排解决“向量相似度较高但不一定最能回答当前问题”的排序误差。

该链路的代价是额外 LLM 调用。是否使用增强检索应结合质量、时延和成本评估；仓库中的离线基准是开发回归材料，不代表生产流量表现。

## 8. Redis 与 PostgreSQL/pgvector 三级对话记忆

### 8.1 工作记忆：Redis

当前会话消息保存为 Redis List：

```text
wm:{user_id}:{conv_id}
```

每项是 JSON：

```json
{"role":"user","content":"VPN 无法连接","ts":"ISO-8601 时间","metadata":{}}
```

写入使用 `LPUSH`，因此 Redis 内最新消息在前；读取时反转以恢复时间顺序。列表与会话摘要均设置 24 小时 TTL。额外的 `pending_clarification:{user_id}:{conv_id}` 采用 600 秒 TTL 保存待澄清计划。

### 8.2 摘要压缩

工作记忆读取最多 20 条，但正常情况下在写入后达到 15 条即触发压缩：

1. 保留最近 5 条；
2. 其余旧消息交给 LLM 摘要，提示要求保留地点、日期、金额、单号、错误码、项目名和审批状态等实体；
3. 摘要追加到 Redis 的 `summary:{user_id}:{conv_id}`；
4. 原旧消息文本及摘要存入情景记忆；
5. 删除原 List，仅重建最近 5 条。

因此首次达到 15 条时压缩约 10 条，之后每新增约 10 条再次压缩。摘要 LLM 失败时仍会写入“摘要生成失败”的占位文本，保证工作记忆可收缩。

### 8.3 情景记忆与画像：PostgreSQL/pgvector

| 数据类别 | 写入内容 | 检索方式 |
| --- | --- | --- |
| `episodic_memories` | 旧对话摘要、全文、`user_id`、`conv_id`、向量和时间 | 当前 query 的向量检索，按 `user_id` 过滤 |
| `user_profiles` | LLM 提炼的 `preferences` 与部门、地点、常用业务等实体 JSON | 按 `user_id` 读取一条记录 |

用户画像更新取当前会话最近 10 条消息，并要求 LLM 只保留对后续企业服务有长期价值的信息。情景记忆的查询结果会以“相关历史”形式放入 Prompt。

当前实现已有两个需要关注的边界：情景记忆没有时间衰减或相关度阈值，同一用户相似但实际无关的历史可能被召回；长期数据也没有自动保留期或归档策略。

## 9. 监控、评测与反馈闭环

### 9.1 性能监控

`PerformanceMonitor` 默认每 10 秒读取编排器与工具管理器的运行统计。它维护 60 个样本的滑动窗口，并在累计至少半个窗口后使用 Z-score 识别突变。当前阈值包括：

| 指标 | 告警阈值 |
| --- | --- |
| Agent 成功率 | 小于 0.90 |
| 工具成功率 | 小于 0.95 |
| Agent 平均延迟 | 大于 3000ms |
| 工具平均延迟 | 大于 5000ms |

监控会输出 Prometheus Gauge、Histogram 和 Counter，可选地向 Webhook 发送告警。更重要的是，它会将成功率和平均延迟转换为 0 到 0.9 的路由惩罚，回写给编排器；后续 `_best_agent()` 会自然降低异常实例的路由评分。该机制是“运行数据 → 路由调整”的闭环，而不是单纯的指标展示。

### 9.2 评测

`EndToEndEvaluator` 提供：

- 意图分类 Accuracy、Macro-F1 和每类 Precision/Recall/F1；
- 调用编排器的单轮与多轮对话评测；
- LLM-as-Judge，从相关性、准确性、完整性、有用性四个维度打分；
- 与保存在 `EVAL_BASELINE_PATH` 的历史报告比较，识别回归并生成建议。

Judge 的默认及格线为综合得分 0.75。由于 LLM Judge 本身可能与被评模型同源，且评测材料主要是匿名化、自建数据，结果应作为开发回归信号而非线上业务结论。具体汇总数据见 `evaluation/BENCHMARK_REPORT.md`，评测脚本和逐样本结果位于 `evaluation/`。

## 10. HTTP 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 返回服务状态和 Agent 统计 |
| `POST` | `/chat` | 主对话入口 |
| `POST` | `/auth/dev-token` | 仅 `APP_ENV=development` 时签发演示 JWT |
| `POST` | `/access-requests/prepare` | JWT 认证后创建权限申请预览 |
| `GET` | `/operations/{operation_id}` | JWT 认证后读取本人操作预览 |
| `POST` | `/operations/{operation_id}/confirm` | JWT 与 `Idempotency-Key` 确认并执行 |
| `GET` | `/search` | 执行增强 RAG 检索 |
| `POST` | `/knowledge/add` | 批量导入 `{title, content}` 文档 |
| `POST` | `/knowledge/upload` | 上传 `.txt`、`.md` 或 JSON 文档数组 |
| `GET` | `/knowledge/stats` | 查看知识库分片数 |
| `POST` | `/eval/run` | 运行内置或传入评测用例 |
| `GET` | `/monitor` | 返回 Agent、工具、告警和建议摘要 |
| `GET` | `/metrics` | 暴露 Prometheus 指标 |

`ChatResponse` 中的 `tool_calls` 是本次回复的工具调用轨迹；持久操作审计另存于 PostgreSQL。`knowledge_used` 表示预取 RAG 上下文是否成功构建；`execution_mode` 用于区分单任务、并行、串行、澄清或直接回复。报销对话还可返回 `pending_operation` 预览。

### 10.1 人工确认操作与数据边界

`SideEffectExecutor` 接受操作提案，调用适配器预检角色、参数与业务状态，构造只读预览，再把操作 ID、资源版本和快照哈希存入 PostgreSQL。提交时 JWT 中的 `sub` 必须是预览所有者；执行器以事务锁定操作和幂等键，校验预览未过期、状态仍待确认，再次预检并比对资源版本和快照。只有执行器可以授权 `submit_claim` 工具执行提交；相同幂等键重放已保存响应，冲突、失效和失败会记录审计事件。

报销单经 `/chat` 的演示流程产生预览，但 `/chat` 本身未认证，不能将自然语言“确认”当作已认证提交。权限申请由已认证的 `/access-requests/prepare` 创建预览；确认后只写入本地演示工单，不调用 IAM。正常 FastAPI 生命周期下操作预览、幂等记录与审计事件使用 PostgreSQL 仓库，报销单与权限工单仍是进程内数据，进程重启不会保留业务记录。开发环境的 `/auth/dev-token` 只供本地演示，不是外部身份提供商。

## 11. 部署、配置与安全边界

Docker Compose 运行以下服务：

```text
Nginx → EchoMind (FastAPI)
             ├─ Redis 7（AOF 持久化）
             ├─ PostgreSQL/pgvector（持久化卷；向量与操作记录）
             └─ Prometheus
```

常用配置包括：

| 环境变量 | 含义 | 默认或示例 |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | 模型服务密钥 | 必填 |
| `ANTHROPIC_MODEL` | 模型名称 | `claude-3-5-sonnet-20241022` |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容服务地址 | 可选 |
| `REDIS_URL` | Redis 连接地址 | Compose 中指向 `redis:6379/0` |
| `DATABASE_URL` / `DATABASE_URL_DOCKER` | PostgreSQL 连接地址 | 本地 / Compose 容器内连接串 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | 向量模型与维度 | `BAAI/bge-small-zh-v1.5` / `512` |
| `JWT_SECRET_KEY` | 操作接口 JWT 签名密钥 | 部署时更换示例值 |
| `MAX_TOOL_ROUNDS` | 单次 Agent 的最大工具调用轮数 | `5` |
| `MONITOR_INTERVAL` | 监控采集间隔秒数 | `10` |
| `ALERT_WEBHOOK_URL` | 告警 Webhook | 可选 |

不要提交 `.env`、模型 Key、数据库密码或企业敏感知识。当前 CORS 配置允许全部来源，且 `/chat` 与报销演示管理接口尚无 Bearer 鉴权；真实企业部署前应收紧来源并完善身份与访问控制、租户隔离以及敏感数据审计与脱敏。

## 12. 已知限制与后续改进方向

1. `mcp/` 不是标准 MCP 协议实现；若需对接外部 MCP Server，需要补充协议传输、握手与客户端/服务端能力。
2. 当前多 Agent 回复只做编号拼接，没有统一的语义汇总、冲突消解或跨 Agent 去重。
3. 任务重叠只做原文字符串包含判断，尚未实现实体标准化或语义重叠识别。
4. `account` 当前复用 GeneralAgent；如需独立的账号服务策略，应新增 AccountAgent 与对应工具权限。
5. 长期 PostgreSQL 向量数据没有保留期、归档、分区和容量压测策略。
6. RAG 的查询改写与 LLM 重排提高相关性，但会增加时延和模型成本；可引入简单查询分流、混合检索、轻量 reranker 与查询缓存。
7. 现有评测以离线自建样本为主，尚不能代表真实员工流量、真实业务系统正确性或生产吞吐能力。

## 13. 推荐阅读顺序

1. `api/main.py`：先理解应用如何装配和 `/chat` 如何串联全部组件；
2. `core/intent_recognizer.py` 与 `core/task_planner.py`：理解漏斗、领域分析、任务计划与校验；
3. `agents/agent_orchestrator.py`：理解 Agent 执行、工具循环、串并行与降级；
4. `memory/conversation_memory.py`：理解三级记忆的数据结构和压缩逻辑；
5. `mcp/tool_manager.py` 与 `mcp/knowledge_base.py`：理解工具治理和 RAG；
6. `monitor/performance_monitor.py`、`evaluation/evaluator.py`：理解观测与评测闭环；
7. `tests/`、`evaluation/`：查看可复现的单元测试、基准脚本和离线材料。
