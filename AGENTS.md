# EchoMind 项目全局说明

## 1. 项目定位

EchoMind 是一个基于 **Python + FastAPI** 的企业内部智能服务助手，面向员工的通用制度咨询、IT 系统使用、财务报销查询等场景，并可扩展至更多内部服务能力。系统通过意图理解、对话记忆、知识检索和专业 Agent 协作，帮助员工从分散的企业知识中获得可追溯的自助服务支持。

项目的核心目标是构建一条完整的 Agent 处理链路：

```text
用户请求
  → 意图识别与紧急度判断
  → 读取三级对话记忆 / RAG 知识上下文
  → 多 Agent 路由、并行协作与工具调用
  → 写回记忆
  → 评测、监控与路由反馈
```

## 2. 技术栈

- 服务框架：Python、FastAPI、Pydantic、Uvicorn
- 大模型：DeepSeek API 或 Anthropic API；代码通过 `anthropic` 的 `AsyncAnthropic` 客户端调用
- 存储：Redis（会话工作记忆）、ChromaDB（情景记忆、用户画像、知识库）
- 可观测性：Prometheus、内置异常检测、可选 Webhook 告警
- 部署：Docker、Docker Compose、Nginx

说明：当配置 `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic` 时，项目使用 Anthropic SDK 以 Anthropic 兼容协议调用 DeepSeek；SDK 名称不代表实际模型一定是 Claude。

## 3. 核心模块

| 模块             | 入口文件                             | 职责                                                                        |
| -------------- | -------------------------------- | ------------------------------------------------------------------------- |
| HTTP API 与应用装配 | `api/main.py`                    | FastAPI 生命周期、聊天/知识库/评测/监控接口，以及组件初始化                                       |
| 多 Agent 编排     | `agents/agent_orchestrator.py`   | 通用制度咨询（General）、IT 运维（Technical）、财务报销（Billing）Agent；意图路由、并行协作、升级/降级、运行时统计 |
| 意图识别           | `core/intent_recognizer.py`      | LLM Few-shot、Embedding 相似度、关键词 Pattern 的多策略识别；实体提取、紧急度判断、缓存               |
| 对话记忆           | `memory/conversation_memory.py`  | Redis 工作记忆、ChromaDB 情景记忆、ChromaDB 用户画像；长对话摘要压缩                            |
| 工具治理与检索优化      | `mcp/tool_manager.py`            | 工具注册、Agent 权限、Tool Calling、缓存、超时、熔断、降级；查询改写、并行召回、重排                       |
| 知识库            | `mcp/knowledge_base.py`          | 文档切片、ChromaDB 检索、知识库工具处理器                                                 |
| 评测             | `evaluation/evaluator.py`        | 意图 Accuracy/F1、LLM-as-Judge、端到端多轮对话、基线回归检测                                |
| 在线监控           | `monitor/performance_monitor.py` | Prometheus 指标、Z-score 异常检测、阈值告警、路由惩罚反馈                                    |

## 4. 关键设计

### 4.1 意图识别与路由

- 意图类别包括制度查询、服务请求、技术支持、财务报销、账号管理、反馈与人工升级等。
- 识别器以 LLM 语义理解为主，结合轻量 Embedding 相似度和规则匹配；结果通过加权投票合并，低置信度会回退到 `OTHER`。
- LLM 与 Embedding 分支可并行执行，并有 LRU 缓存。
- `AgentOrchestrator` 先按意图选择通用服务、IT 运维或财务报销等领域 Agent，再综合成功率、平均延迟和监控惩罚选择可用实例；跨部门复合问题可并行派发多个 Agent。

### 4.2 三级记忆

- **工作记忆**：Redis 保存员工当前会话的近期消息，默认 TTL 为 24 小时。
- **情景记忆**：ChromaDB 保存可跨会话检索的企业服务历史片段。
- **用户画像**：ChromaDB 保存从对话提炼出的长期偏好、部门或常用业务实体。
- 达到阈值时使用 LLM 压缩历史，最终将摘要、相关历史、画像和近期消息合成为 Agent 上下文。

### 4.3 RAG 与工具调用

- 企业知识文档（制度、流程与操作规范等）进入知识库后会被切片并写入 ChromaDB。
- 对检索类工具，链路为：查询改写 → 多子查询并行召回 → 合并去重 → LLM 重排 → Top-K 返回。
- 工具管理器提供 JSON Schema 参数校验、Agent 级权限、缓存、超时、三态熔断器和降级结果。
- `mcp/` 是项目内的工具治理模块命名；当前实现是自研的 LLM Tool Calling 框架，**不是完整的标准 MCP（Model Context Protocol）Server/Client 协议实现**。不要在文档或简历中把它表述为已接入标准 MCP，除非后续补齐 MCP 传输与服务端实现。

### 4.4 评测与监控闭环

- 评测包含意图识别测试、LLM-as-Judge 回复质量评分、端到端多轮对话评测和与历史基线的回归对比。
- 监控采集 Agent/工具成功率、延迟等指标；滑动窗口 Z-score 用于发现异常。
- 监控结果会形成路由惩罚，反馈到编排器的 Agent 选择逻辑，构成运行指标 → 路由调整的闭环。

## 5. API 与运行

常用接口在 `api/main.py` 中定义，主要包括：

- `GET /health`：健康检查
- `POST /chat`：主对话接口
- `GET /search`：知识库检索
- `POST /knowledge`、`POST /knowledge/upload`：知识文档写入与上传
- `POST /evaluation/run`：执行评测
- `GET /monitor/summary`、`GET /metrics`：监控摘要和 Prometheus 指标

本地依赖与部署配置：

- Python 依赖见 `requirements.txt`。
- 环境变量样例见 `.env.example`；不要将真实 API Key、Redis 密码或其他敏感配置提交到仓库或输出到日志。
- 多服务启动配置见 `docker-compose.yml`，包含 Redis、ChromaDB、Prometheus、EchoMind 和 Nginx。通常可使用 `docker compose up --build` 启动。

## 6. 开发约定与注意事项

1. 保持异步调用链路：API、模型调用、工具调用和记忆操作优先使用 `async` / `await`，避免在请求路径中加入阻塞操作。
2. 修改 Agent、工具或评测数据结构时，同步检查 `api/main.py` 的序列化响应、监控统计和测试。
3. 修改工具实现时保留参数校验、权限控制、超时、缓存、熔断和降级语义，避免绕过 `MCPToolManager.call()` 直接执行不受治理的工具。
4. 修改路由评分时，注意 `PerformanceMonitor` 会向编排器写入监控惩罚；不要破坏该反馈闭环。
5. `ANTHROPIC_BASE_URL` 非空时，意图识别器会关闭其 Embedding 分支以兼容第三方接口；不要假设所有部署配置都会同时启用三路识别。
6. 项目当前测试位于 `tests/test_agent_tools.py`；为新增的核心行为补充测试，尤其覆盖工具错误、超时/熔断、路由降级和数据序列化。

## 7. 简历表述边界

可以准确表述为：企业内部智能服务助手、多 Agent、RAG、LLM Tool Calling、Redis + ChromaDB 三级记忆、LLM-as-Judge、Prometheus 监控、Docker Compose 部署。

不应在未实现或未验证时表述为：完整标准 MCP 协议、真实动态学习的意图权重、已量化的性能提升、Grafana 可视化或外部业务系统工具集成。
