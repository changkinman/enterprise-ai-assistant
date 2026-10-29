# EchoMind 架构与处理流程

> 本文依据当前仓库实现整理，面向参与维护、扩展和部署 EchoMind 的开发者。本文描述的是现状；其中 `mcp/` 目录是项目内部的工具治理与知识检索模块，并非标准 Model Context Protocol（MCP）Server/Client 实现。

## 1. 系统定位与边界

EchoMind 是企业内部智能服务助手。它将员工的自然语言请求，经过意图理解、对话记忆、企业知识检索和领域 Agent 协作，转化为可追溯的自助服务答复。

当前内置的主要服务领域如下：

| 领域 | 典型问题 | 负责组件 |
| --- | --- | --- |
| 通用制度咨询 | 公司制度、流程、政策查询 | General Agent |
| IT 技术支持 | 系统使用、故障排查、账号相关问题 | Technical Agent |
| 财务报销 | 报销政策、票据与流程查询 | Billing Agent |

系统不直接替代企业业务系统，也不在当前实现中提供完整标准 MCP 协议。工具执行统一经过项目内 `MCPToolManager` 治理，知识检索通过内置知识库工具完成。

## 2. 技术架构

```text
客户端
  │ HTTP
  ▼
FastAPI（api/main.py）
  ├── /chat                 主对话入口
  ├── /search               知识检索
  ├── /knowledge*           知识写入与上传
  ├── /eval/run             评测执行
  ├── /access-requests/prepare、/operations/{id}、/operations/{id}/confirm
  └── /monitor、/metrics
  │
  ├── IntentRecognizer      意图、实体、紧急度识别
  ├── MemoryManager         工作记忆、情景记忆、用户画像
  ├── AgentOrchestrator     领域路由、并行协作、结果汇总
  ├── MCPToolManager        工具调用治理与检索优化
  ├── KnowledgeBase         文档切片、向量检索
  ├── SideEffectExecutor    操作预检、预览、确认、幂等和审计
  ├── EndToEndEvaluator     离线/在线评测
  └── PerformanceMonitor    指标、异常、路由惩罚
          │                    │
          ▼                    ▼
  Anthropic 兼容 LLM API    Prometheus
          │
          ├── Redis（会话工作记忆）
          └── PostgreSQL + pgvector（情景记忆、画像、知识库、操作记录）
```

`api/main.py` 是应用装配点。FastAPI 生命周期启动时连接一次 PostgreSQL，构造显式 Embedding provider 与共享 VectorStore，并注入记忆管理器和知识库；HTTP 路由复用这些已初始化的组件处理请求。关闭时释放数据库连接池。

正常生命周期同时注入 `PostgresOperationRepository`。操作预览、确认状态、幂等响应和审计事件持久化在 PostgreSQL；报销单与权限申请工单使用进程内仓库，重启后丢失，不代表持久业务数据。权限申请未连接外部 IAM。

模型调用使用 `anthropic.AsyncAnthropic` 的异步客户端。通过设置 `ANTHROPIC_BASE_URL` 可将请求指向 Anthropic 兼容的第三方接口（例如 DeepSeek 的 Anthropic 兼容端点）；因此 SDK 名称并不等同于实际使用 Claude 模型。

## 3. 核心组件职责

| 模块 | 入口 | 主要职责 | 关键依赖 |
| --- | --- | --- | --- |
| API 层 | `api/main.py` | 生命周期管理、请求校验、HTTP 响应序列化 | FastAPI、全部核心组件 |
| Agent 编排 | `agents/agent_orchestrator.py` | 按意图选择领域 Agent；复合问题并行派发；按成功率、延迟和监控惩罚选择实例 | IntentRecognizer、工具管理器、监控器 |
| 意图识别 | `core/intent_recognizer.py` | LLM Few-shot、Embedding 相似度、关键词规则的综合识别；实体和紧急度提取；缓存 | LLM、可选 Embedding |
| 记忆 | `memory/conversation_memory.py` | 管理短期会话、跨会话历史与长期画像；长对话摘要压缩 | Redis、VectorStore、LLM |
| 工具治理 | `mcp/tool_manager.py` | 工具注册和 Agent 授权；参数校验、缓存、超时、熔断、降级；检索查询改写和重排 | LLM、KnowledgeBase |
| 知识库 | `mcp/knowledge_base.py` | 文档切片、向量写入、检索 | VectorStore、显式 Embedding provider |
| 人工确认操作 | `core/side_effect_executor.py` | 预检、预览、确认时重验、幂等与审计 | JWT、PostgresOperationRepository、业务适配器 |
| 评测 | `evaluation/evaluator.py` | 意图 Accuracy/F1、LLM-as-Judge、端到端多轮和基线回归评测 | 编排器、识别器、LLM |
| 监控 | `monitor/performance_monitor.py` | Prometheus 指标、Z-score 异常检测、告警和路由惩罚反馈 | 编排器、Prometheus |

## 4. 应用启动流程

```text
进程启动
  → 加载 .env 与日志配置
  → FastAPI lifespan 开始
  → 读取 LLM 配置（API Key、模型、可选 Base URL）
  → 连接 PostgreSQL，构造 pgvector VectorStore 和显式 Embedding provider
  → 创建 Redis 工作记忆管理组件、知识库和工具管理器
  → 空知识库时播种默认文档，并注册受治理工具
  → 创建意图识别器、领域 Agent 编排器、评测器与监控器
  → 对外提供 HTTP 接口
  → 应用关闭时释放/关闭相关资源
```

关键配置由环境变量提供。`ANTHROPIC_API_KEY` 是模型调用的必要配置；`ANTHROPIC_MODEL`、`ANTHROPIC_BASE_URL` 和 `LOG_LEVEL` 等用于调整运行行为。真实密钥和连接密码仅应放在本地环境配置中，不应写入代码、文档样例或日志。

## 5. 主对话流程：`POST /chat`

```text
用户消息、会话 ID、用户 ID
  │
  ├─ 1. 读取记忆
  │     Redis 近期消息 + PostgreSQL/pgvector 相关历史 + 用户画像
  │
  ├─ 2. 识别请求
  │     LLM / Embedding / 规则 → 意图、实体、紧急度、置信度
  │
  ├─ 3. 组装上下文
  │     当前问题 + 近期对话 + 历史摘要/片段 + 画像 + 检索上下文
  │
  ├─ 4. 编排并执行 Agent
  │     单领域路由，或为跨领域请求并行选择多个 Agent
  │     Agent 必须通过 MCPToolManager 调用已获授权工具
  │
  ├─ 5. 汇总并返回答案
  │     形成对用户可见的回复及必要元数据
  │
  └─ 6. 写回与观测
        写入工作/情景/画像记忆；记录 Agent 与工具成功率、延迟等指标
        监控异常可更新路由惩罚
```

### 5.1 意图识别与路由

`IntentRecognizer` 覆盖制度查询、服务请求、技术支持、财务报销、账号管理、反馈和人工升级等意图。它以 LLM 语义理解为主，并结合 Embedding 相似度与关键词 Pattern 结果加权投票；低置信度结果回退为 `OTHER`。为减少重复推理，识别结果使用 LRU 缓存。

LLM 与 Embedding 分支可并行运行。为兼容第三方 Anthropic API，当 `ANTHROPIC_BASE_URL` 被设置时，识别器会关闭 Embedding 分支；部署时不应假定三种识别策略始终同时可用。

`AgentOrchestrator` 根据识别结果选取 General、Technical 或 Billing 等领域能力。对于多部门复合问题，多个领域 Agent 可并行执行。候选 Agent 的选择除领域匹配外，还综合运行成功率、平均延迟和监控模块提供的惩罚，以便在异常时倾向健康实例或采取降级路径。

### 5.2 记忆读取与写回

记忆分为三级：

| 层级 | 存储 | 作用 |
| --- | --- | --- |
| 工作记忆 | Redis | 保存当前会话的近期消息，默认 TTL 为 24 小时 |
| 情景记忆 | PostgreSQL + pgvector | 保存可跨会话检索的企业服务历史片段 |
| 用户画像 | PostgreSQL | 保存从对话中提炼的长期偏好、部门和常用业务实体 |

当会话历史达到阈值时，系统会使用 LLM 将较早消息压缩为摘要。执行 Agent 前，记忆管理器将摘要、相似历史、用户画像与最近消息组合为上下文；完成响应后，再将本轮内容和可沉淀信息写回相应层级。

## 6. 知识库与受治理工具流程

### 6.1 知识写入

```text
制度/流程/操作规范文档
  → /knowledge/add 或 /knowledge/upload
  → 文本抽取与切片
  → 显式 Embedding provider 生成向量并写入 PostgreSQL + pgvector
  → 保留可检索的内容及元数据
```

### 6.2 检索与工具调用

```text
Agent 请求工具
  → MCPToolManager.call()
  → Agent 权限检查 + JSON Schema 参数校验
  → 缓存命中？命中则直接返回
  → 超时控制与熔断器状态检查
  → 执行工具（如知识库检索）
  → 成功结果缓存并记录指标；失败时返回降级结果
```

知识检索类工具在执行中采用以下优化链路：

```text
原始查询
  → 查询改写
  → 多子查询并行召回
  → 合并与去重
  → LLM 重排
  → 返回 Top-K 片段给 Agent
```

这条链路既提升了对不同表述的覆盖，也把结果数量控制在模型上下文可用范围内。任何新增工具都应注册到工具管理器，并保留权限、校验、缓存、超时、熔断和降级语义；不要绕过 `MCPToolManager.call()` 直接从 Agent 执行工具。

## 7. 评测、监控与反馈闭环

```text
请求/评测执行
  → 记录意图、Agent、工具的成功率与延迟
  → Prometheus 暴露指标（/metrics）
  → PerformanceMonitor 维护滑动窗口并计算 Z-score
  → 发现异常时触发阈值告警（可选 Webhook）
  → 生成路由惩罚
  → AgentOrchestrator 在后续选择候选 Agent 时纳入惩罚
```

评测模块提供四类能力：意图识别 Accuracy/F1、LLM-as-Judge 回复质量、端到端多轮对话，以及与历史基线的回归检测。监控模块并非只用于展示：其惩罚会反馈至编排器，形成“运行指标 → 路由调整”的闭环。因此，修改路由评分或监控统计时，应同时验证这条反馈链没有被中断。

## 8. API 概览

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 服务健康检查 |
| `POST /chat` | 主对话入口 |
| `POST /auth/dev-token` | 仅 `APP_ENV=development` 签发本地演示 JWT |
| `POST /access-requests/prepare` | JWT 认证后预检并预览本地权限申请 |
| `GET /operations/{operation_id}` | JWT 认证后读取本人预览 |
| `POST /operations/{operation_id}/confirm` | JWT + `Idempotency-Key` 确认操作 |
| `GET /search` | 知识库检索 |
| `POST /knowledge/add` | 写入知识文档 |
| `POST /knowledge/upload` | 上传并写入知识文档 |
| `POST /eval/run` | 运行评测 |
| `GET /monitor` | 获取监控摘要 |
| `GET /metrics` | 暴露 Prometheus 指标 |

### 8.1 人工确认操作

报销对话可以返回 `pending_operation`，权限申请使用已认证的 `/access-requests/prepare` 产生预览。`/chat` 仍以请求体 `user_id` 标识用户且无 Bearer 鉴权，只适合演示对话；自然语言确认不能代替操作 API 的认证。客户端读取预览后，以同一所有者的 JWT 和非空 `Idempotency-Key` 调用确认接口。统一执行器在确认时锁定待处理操作，核对到期时间、资源版本与快照，再由适配器执行一次业务写入；相同幂等键返回已有响应，冲突、失效和失败写入操作审计。报销提交工具要求执行器授权，不能由普通 Agent 工具调用直接提交。开发环境令牌接口仅供本地演示，未实现外部身份系统集成。

## 9. 部署与维护要点

- 推荐通过 Docker Compose 同时启动 EchoMind、Redis、PostgreSQL + pgvector、Prometheus 和 Nginx。
- 请求路径应保持异步：API、模型调用、工具调用与记忆操作均优先使用 `async` / `await`，避免阻塞 FastAPI 事件循环。
- 变更 Agent、工具或评测数据结构时，同时检查 API 响应序列化、监控统计以及测试覆盖。
- 核心行为的测试重点包括工具错误、超时和熔断、路由降级、数据序列化及操作确认。测试分布在 `tests/` 中的多个模块。
- 本项目当前可准确描述为：多 Agent、RAG、LLM Tool Calling、Redis + PostgreSQL/pgvector 三级记忆、LLM-as-Judge、Prometheus 监控和 Docker Compose 部署。不要将其表述为已实现标准 MCP 协议、真实动态学习的意图权重、Grafana 可视化或外部业务系统集成，除非对应能力后续实际完成并验证。
