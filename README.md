# EchoMind

> 面向企业内部服务场景的多 Agent 智能助手：将漏斗式领域分析、任务规划、三级对话记忆、RAG 知识检索、受治理的工具调用、离线评测与在线监控串联为一条可观测的服务链路。

EchoMind 适用于员工制度咨询、IT 系统支持、财务报销查询等场景。服务使用 Anthropic SDK 调用模型；配置兼容 Anthropic 协议的第三方 `ANTHROPIC_BASE_URL` 后，也可接入相应模型服务。

## 功能概览

- **任务规划与多 Agent 编排**：通过漏斗式领域分析识别简单、复杂及歧义请求；复杂问题由 LLM 拆分为带原文证据和依赖关系的任务，再由代码决定单路、并行、串行或澄清路径。
- **领域分析与意图识别**：主对话链路采用关键词与本地 n-gram 的轻量领域分析，并在复杂场景升级到 LLM 多领域分析；独立意图识别组件支持 LLM Few-shot、规则匹配、可选 Embedding 相似度、实体提取、紧急度判断和低置信度回退。
- **三级记忆**：Redis 保存会话工作记忆，ChromaDB 保存可检索的情景记忆与用户画像；长对话可被压缩为摘要。
- **RAG 知识库**：支持文档切片、导入、查询改写、多子查询并行召回、去重与 LLM 重排。
- **工具治理**：工具的 JSON Schema 校验、Agent 权限、缓存、超时、熔断与降级统一由工具管理器处理；`tools/` 支持按约定自动发现扩展工具。
- **评测与监控**：提供意图、RAG、记忆和端到端回答质量评测；Prometheus 指标、Z-score 异常检测与可选 Webhook 告警会反馈给路由逻辑。

> `mcp/` 是项目内部的工具治理模块命名，当前实现为自研 LLM Tool Calling 框架，并非完整的标准 MCP（Model Context Protocol）Server/Client 实现。

## 架构

```text
用户请求
  │
  ├─ 读取工作记忆 / 情景记忆 / 用户画像
  ├─ RAG：查询改写 → 并行召回 → 去重 → 重排
  ├─ 请求规划
  │    ├─ 精确寒暄 → 直接回复
  │    ├─ 轻量领域分析 → 单任务
  │    └─ LLM 多领域分析 → 任务拆分 → 代码校验
  ├─ 执行计划：DIRECT / SINGLE / PARALLEL / SEQUENTIAL / CLARIFY
  ├─ Agent 编排：General / Technical / Billing
  │      └─ 受治理的工具调用（校验、权限、缓存、超时、熔断、降级）
  ├─ 生成回复并写回三级记忆
  └─ 监控指标、异常检测与路由反馈
```

## 请求规划与多 Agent 路由

`POST /chat` 的主链路不再将用户消息直接归为单一意图后路由，而是先生成可执行的任务计划。

```text
用户消息
  → 精确寒暄规则
  → 轻量领域分析（关键词 + 本地字符 n-gram）
  → 复杂请求的 LLM 多领域分析
  → LLM 任务拆分（领域、原文证据、实体、动作、依赖）
  → 代码校验与执行模式决策
  → 领域 Agent 执行与结果聚合
```

### 漏斗与领域候选

- 精确匹配“你好”“您好”“hi”“hello”等寒暄时，系统直接返回欢迎语，不调用 Agent。
- 其他请求先对 `general`、`technical`、`billing`、`account` 四个业务领域进行本地轻量评分。只有存在唯一的高置信领域、文本不含任务连接词、文本不长且第一、第二候选分差足够大时，系统才直接构造单任务。
- 出现多个可信领域、"同时/另外/并且/先/再"等连接词、长文本或候选分差过小时，请求会升级到 LLM 多领域分析。这里的升级表示进入更完整的分析流程，不等同于人工升级。

### 任务拆分、校验与调度

对于多个可信领域，LLM 只生成任务草案。每项任务包含领域、用户原文中的连续证据、处理实体、动作、置信度和 `depends_on` 依赖关系。代码负责验证并决定是否执行：

| 校验项 | 处理方式 |
| --- | --- |
| 领域不可信、置信度不足、证据不在用户原文中、实体或动作为空 | 返回澄清问题 |
| 任务 ID 重复、依赖不存在、自依赖或依赖成环 | 返回澄清问题 |
| 多任务证据存在包含关系 | 返回澄清问题，避免对同一问题重复执行 |
| 单个有效任务 | `SINGLE`：路由一个领域 Agent |
| 存在前置依赖 | `SEQUENTIAL`：按依赖顺序执行，前置结果传给后置任务 |
| 跨领域且无依赖 | `PARALLEL`：并发执行多个 Agent |
| 同领域多任务且无依赖 | `SEQUENTIAL`：保守串行执行 |

并行任务由 `asyncio.gather()` 调度，结果保持任务计划顺序；串行任务会将前置回复注入后置任务上下文。多任务结果当前采用编号拼接，同时合并全部工具调用轨迹；任一任务需要升级时，最终响应标记为已升级。专业 Agent 执行失败时会降级到 General Agent。

## 技术栈

| 类别 | 组件 |
| --- | --- |
| API 服务 | Python 3.12、FastAPI、Pydantic、Uvicorn |
| 模型调用 | Anthropic Python SDK；可配置 Anthropic 兼容服务 |
| 存储 | Redis、ChromaDB |
| 可观测性 | Prometheus、Z-score 异常检测、Webhook 告警 |
| 部署 | Docker、Docker Compose、Nginx |

## 快速开始

### 方式一：Docker Compose（推荐）

**前置条件**：Docker Desktop（含 Docker Compose）以及可用的模型 API Key。

1. 创建本地配置文件并填写模型密钥：

   ```powershell
   Copy-Item .env.example .env
   ```

   至少修改 `.env` 中的 `ANTHROPIC_API_KEY`。若使用 Anthropic 兼容服务，可同时设置：

   ```dotenv
   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
   ANTHROPIC_MODEL=deepseek-chat
   ```

2. 构建并启动：

   ```powershell
   docker compose up --build
   ```

3. 打开以下地址：

   - 应用健康检查：<http://localhost/health>
   - OpenAPI 文档：<http://localhost/docs>
   - Prometheus：<http://localhost:9090>

Nginx 监听 `80` 端口，应用容器也映射在 `8000` 端口；两者均可用于访问应用。停止服务可执行 `docker compose down`。默认会保留 Docker 数据卷；若需要一并删除数据卷，请先确认其中不含要保留的数据，再使用 `docker compose down -v`。

### 方式二：本地开发

**前置条件**：Python 3.12、Redis；建议启动 ChromaDB 服务（不可用时知识库会降级为本地持久化模式）。所有 Python 依赖均安装在项目虚拟环境中。

```powershell
# 创建并激活虚拟环境（首次使用时）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装项目依赖
python -m pip install -r requirements.txt

# 配置环境变量
Copy-Item .env.example .env

# 启动 API（默认 http://127.0.0.1:8000）
python -m uvicorn api.main:app --reload
```

本地运行时，请将 `.env` 的 Redis 和 ChromaDB 地址改为本机实际地址；例如使用 Compose 单独启动依赖服务后，可将 `CHROMA_HOST=localhost`、`CHROMA_PORT=8001`。`CHROMA_PERSIST_DIRECTORY` 仅用于 ChromaDB 服务不可用时的本地回退。

也可以运行交互式命令行：

```powershell
python api/main.py --cli
```

## 配置说明

配置模板见 [`.env.example`](.env.example)。以下变量最常用：

| 变量 | 作用 | 示例 / 默认值 |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | 模型服务密钥，必填 | 请在 `.env` 中填写 |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容 API 地址，可选 | `https://api.deepseek.com/anthropic` |
| `ANTHROPIC_MODEL` | 模型名称，可选 | `claude-3-5-sonnet-20241022` |
| `REDIS_URL` | Redis 连接地址 | `redis://localhost:6379/0` |
| `REDIS_PASSWORD` | Redis 密码 | 本地开发请按实际 Redis 配置填写 |
| `CHROMA_HOST` / `CHROMA_PORT` | ChromaDB 服务地址 | 本地 Compose 映射端口为 `localhost:8001` |
| `MAX_TOOL_ROUNDS` | 单次 Agent 请求的最大工具调用轮数 | `5` |
| `PROMETHEUS_PORT` | 应用额外暴露 Prometheus HTTP 服务的端口；设 `0` 关闭 | `9091` |
| `EVAL_BASELINE_PATH` | 评测基线文件路径 | `./data/eval/baseline.json` |

不要提交 `.env`，更不要将 API Key、Redis 密码或其他真实凭据写入日志和示例请求。

## API 示例

完整且可交互的接口定义以启动后的 [`/docs`](http://localhost:8000/docs) 为准。

### 对话

```powershell
$body = @{ message = 'VPN 无法连接，提示认证失败'; user_id = 'employee-001' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -ContentType 'application/json' -Body $body
```

响应会包括会话 ID、识别意图、首个执行 Agent、执行模式、是否升级、延迟、知识库是否命中及工具调用轨迹。多任务请求还会以编号方式汇总各任务结果。

### 导入知识文档

```powershell
$body = @{
  documents = @(
    @{ title = 'VPN 使用规范'; content = '员工可通过企业门户申请 VPN 权限……' },
    @{ title = '报销流程'; content = '报销单需附合规发票，并在系统中提交……' }
  )
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:8000/knowledge/add -ContentType 'application/json' -Body $body
```

也可以上传 `.txt`、`.md` 或 JSON 文档数组：

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/knowledge/upload -Form @{ file = Get-Item .\policy.md }
```

### 检索与运维接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 服务状态与 Agent 统计 |
| `POST` | `/chat` | 主对话链路 |
| `POST` | `/search?query=...&top_k=5` | RAG 检索优化链路 |
| `POST` | `/knowledge/add` | 批量导入 JSON 文档 |
| `POST` | `/knowledge/upload` | 上传 `.txt`、`.md` 或 JSON 文档数组 |
| `GET` | `/knowledge/stats` | 知识库片段统计 |
| `POST` | `/eval/run` | 执行内置或自定义评测 |
| `GET` | `/monitor` | Agent / 工具统计、告警与建议 |
| `GET` | `/metrics` | Prometheus 指标 |

## 工具扩展

在 `tools/` 目录新增以 `_tool.py` 结尾的模块即可被自动发现。模块需提供 `build_tools()`，返回一个或多个 `Tool`；同时指定参数 Schema 与可调用 Agent。详细约定见 [`tools/README.md`](tools/README.md)。

当前 `billing_lookup` 和 `fault_diagnose` 是用于演示工具调用链路的模拟工具，不连接真实财务、工单或企业系统。

## 测试与评测

在已激活的虚拟环境中运行：

```powershell
python -m unittest discover -s tests -v
```

项目附带的是匿名化、自建的离线评测材料，覆盖意图识别、RAG、三级记忆与回答质量。扩展集的结果和边界说明见 [`evaluation/BENCHMARK_REPORT.md`](evaluation/BENCHMARK_REPORT.md)，评测材料说明见 [`evaluation/fixtures/README.md`](evaluation/fixtures/README.md)。这些结果用于开发回归，不代表线上真实业务表现。

## 项目结构

```text
EchoMind/
├── api/main.py                    # FastAPI 生命周期与 HTTP 接口
├── agents/agent_orchestrator.py   # 多 Agent 执行、串并行、降级与统计
├── core/intent_recognizer.py      # 漏斗、领域分析、独立意图/实体/紧急度识别
├── core/task_planner.py            # LLM 任务草案校验与执行模式决策
├── memory/conversation_memory.py  # Redis + ChromaDB 三级记忆
├── mcp/knowledge_base.py          # 文档切片与 ChromaDB RAG 检索
├── mcp/tool_manager.py            # 工具治理、检索优化与调用链路
├── tools/                         # 可自动发现的扩展工具
├── evaluation/                    # 评测器、基准脚本与匿名化材料
├── monitor/performance_monitor.py # 指标、异常检测与路由反馈
├── tests/                         # 单元测试
└── docker-compose.yml             # Redis、ChromaDB、Prometheus、Nginx 与应用编排
```

## 许可证与使用边界

当前仓库未包含许可证文件，请在分发或商用前补充明确的许可证和数据治理策略。接入真实企业知识、员工信息或业务系统前，应完成鉴权、访问控制、审计、敏感数据脱敏以及相应的安全评估。
