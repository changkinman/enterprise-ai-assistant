# PostgreSQL/pgvector 与通用副作用操作协议设计

## 目标

本次迭代有两个目标：

1. 用 PostgreSQL + pgvector 替换 ChromaDB，存储知识库、情景记忆和用户画像。
2. 将当前仅用于报销提交的确认流程升级为可复用的 Human-in-the-Loop（HITL）副作用操作协议。

目标是为个人项目提供清晰、可演示、能回答日常实习面试追问的实现，而不是构建完整企业工作流平台。Redis 保留用于短期会话记忆与待澄清状态；不将它迁移到 PostgreSQL。

## 范围与非目标

本迭代会提供两类接入通用协议的写操作：

- 报销提交（改造现有流程）。
- 一个本地模拟的 IT 权限申请工单（用于证明协议并非报销专用）。

以下内容不在范围内：外部业务系统集成、前端确认页面、复杂审批流、多租户隔离、分布式 exactly-once 语义、历史 ChromaDB 数据迁移和全项目端到端测试。知识库默认数据可重新导入。

## 总体架构

```text
FastAPI
  ├── Redis：当前会话消息、摘要、待澄清状态
  ├── PostgreSQL + pgvector
  │     ├── 知识库分片 / 情景记忆 / 用户画像
  │     └── 领域数据、待确认操作、幂等记录、审计事件
  ├── EmbeddingProvider：明确生成文本向量
  ├── MCPToolManager：读工具及实际领域工具的治理边界
  └── SideEffectExecutor：预校验 → 预览 → 确认 → 幂等提交
          ├── ReimbursementSubmitOperation
          └── AccessRequestOperation
```

PostgreSQL 负责关系数据、事务和审计；pgvector 只负责向量相似度检索。向量生成不再由向量库隐式完成，而由 `EmbeddingProvider` 明确负责。

## PostgreSQL 与 pgvector 设计

### 技术选择

- Docker Compose 新增 PostgreSQL 镜像，并在初始化 SQL 中执行 `CREATE EXTENSION IF NOT EXISTS vector`。
- Python 数据访问采用异步 PostgreSQL 驱动和轻量 repository 层；数据库 SQL 保持集中，不引入额外的工作流或消息队列。
- 向量维度由 EmbeddingProvider 配置决定。初始实现采用可配置的中文 embedding 实现；测试使用固定的 FakeEmbeddingProvider，不下载模型、不访问网络。
- 向量检索使用 PostgreSQL 的余弦距离排序。数据量很小的个人项目优先保证正确读写；可在初始化脚本中提供可选的 HNSW 向量索引，但不宣称任何性能指标。

### 表与职责

| 表 | 核心字段 | 用途 |
| --- | --- | --- |
| `knowledge_chunks` | `id`, `title`, `content`, `metadata jsonb`, `embedding vector`, `created_at` | RAG 文档切片与检索 |
| `episodic_memories` | `id`, `user_id`, `conv_id`, `summary`, `full_text`, `embedding vector`, `created_at` | 跨会话记忆检索 |
| `user_profiles` | `user_id`, `profile jsonb`, `updated_at` | 每个用户最新长期画像 |
| `reimbursement_cases` | `case_id`, `owner_id`, `state`, `facts jsonb`, `expenses jsonb`, `validation jsonb`, `version` | 报销领域状态 |
| `access_requests` | `request_id`, `owner_id`, `system_name`, `permission`, `state`, `version` | 本地模拟的 IT 权限申请 |
| `operations` | `operation_id`, `actor_id`, `action_type`, `params jsonb`, `preview jsonb`, `snapshot_hash`, `resource_version`, `state`, `expires_at` | 待确认操作及其不可变预览 |
| `idempotency_records` | `actor_id`, `action_type`, `idempotency_key`, `request_hash`, `response jsonb` | 重试返回首次响应，拒绝同 key 不同载荷 |
| `operation_audit_events` | `id`, `operation_id`, `event_type`, `actor_id`, `detail jsonb`, `created_at` | 可追溯的操作事件 |

`operations`、`idempotency_records` 与领域写入必须在同一数据库事务中处理。幂等记录对 `(actor_id, action_type, idempotency_key)` 设置唯一约束。

## EmbeddingProvider

定义最小异步接口：

```python
class EmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

知识库写入、情景记忆压缩写入和查询检索都只依赖此接口。真实实现从配置读取模型或兼容 Embeddings API；同步本地模型编码必须经线程池调用，避免阻塞 FastAPI 事件循环。测试实现返回稳定的固定向量。

## 通用 HITL 副作用操作协议

### 核心边界

模型只能生成结构化 `ActionProposal`，例如动作类型和候选参数。模型无权调用提交工具，不能决定确认，也不能声称执行已成功。

只有 `SideEffectExecutor` 可以触发领域操作的最终写入；实际领域工具仍经 `MCPToolManager.call()` 执行，因此保留参数校验、权限、超时和审计链路。

每个可写领域动作实现同一最小适配器接口：

```text
precheck(actor, params) -> ValidationResult
build_preview(actor, normalized_params, validation) -> Preview
commit(actor, normalized_params, idempotency_key) -> Result
```

适配器负责确定性的领域规则，例如报销制度/预算校验，或权限申请的系统与权限等级校验；执行器负责通用的身份、预览、确认、版本、过期、幂等和审计规则。

### 四阶段流程

```text
LLM ActionProposal
  → 1. precheck：服务端获得可信 Actor，校验参数、权限、资源归属与业务状态
  → 2. preview：规范化参数，生成摘要、resource_version、snapshot_hash、expires_at，并持久化 Operation
  → 3. confirm：可信用户针对 operation_id 显式确认；校验操作者、未过期、状态仍为 pending
  → 4. commit：事务内再次验证版本/快照，使用 Idempotency-Key 调用领域提交并保存响应与审计
```

预览默认有效期为 10 分钟。数据变更、资源版本变化、操作者不匹配、预览过期或已经完成时，确认必须被拒绝，用户需要重新发起预校验。

### 身份与确认

新增 FastAPI 认证依赖，以签名 JWT 解析 `Actor(id, role)`。业务 API 与执行器只使用该 Actor，不再从请求体信任 `user_id`、`reviewer_id` 或 `finance_user_id`。为保持个人项目可演示性，测试和本地文档提供开发用签发方式，但 API 的信任边界保持不变。

确认 API 为 `POST /operations/{operation_id}/confirm`，客户端通过 `Idempotency-Key` 请求头提供幂等键。该接口同步完成 commit 并返回首次或重放结果。聊天中的确认语句可作为兼容入口，但只能由服务端将其绑定到当前用户唯一有效的 pending operation；不会相信 LLM 的 `explicit_confirmation` 字段。

## 两个接入场景

### 报销提交

保留已有事实收集、制度、票据、金额和预算校验。校验通过后不立即调用 `submit_claim`，而是写入 `reimbursement_submit` 类型的 Operation。预览显示报销单号/草稿标识、费用明细、总额、制度摘要和预算影响。最终提交在事务中重新验证案件版本、校验结果和可用预算，成功后将案件置为 submitted。

草稿与费用明细属于可修订的准备动作；“报销提交”才是本协议保护的高风险最终副作用。修改任一费用后，旧预览失效，必须重新预校验和确认。

### IT 权限申请

新增最小本地 `access_request` 领域。模型可提议系统名、所需权限和理由；服务端校验系统是否受支持、申请者是否可申请该级别权限，再生成预览。确认后创建提交状态的权限申请工单。它不连接真实 IAM 系统，仅用于验证同一执行器可处理第二类领域写操作。

## API 与响应变化

- `POST /chat` 在需要确认时返回可展示的 `pending_operation`（operation_id、摘要、过期时间、状态），而非只返回文本提示。
- `GET /operations/{operation_id}` 仅允许操作者读取自己的预览与状态。
- `POST /operations/{operation_id}/confirm` 使用 JWT Actor 和 `Idempotency-Key` 完成确认/提交。
- 保留现有报销查询接口；审核和付款若仍保留，同样改为从 Actor 获取角色，并复用幂等写入语义。

## 错误处理与审计

预校验失败不创建可确认操作；确认或提交失败记录审计事件但不产生部分最终副作用。提交接口需明确区分：未认证（401）、无权限或非操作所有者（403）、预览过期/版本冲突/错误状态（409）、参数格式错误（422）。

每次 proposal、预校验拒绝、预览生成、确认拒绝、提交成功、幂等重放或工具失败都会写入 `operation_audit_events`。审计数据不包含密钥或原始 JWT。

## 精简测试策略

不要求启动 Docker Compose 或覆盖所有对话路径。测试以 repository、executor 与领域适配器为主，保留以下关键用例：

1. pgvector repository 可写入并返回最相关的知识/记忆记录（FakeEmbeddingProvider）。
2. 校验失败或未确认时不会执行 commit。
3. 非操作所有者、过期预览、已变更版本的预览无法确认。
4. 同一 Idempotency-Key 重试返回首次结果；同 key 不同请求被拒绝。
5. 并发确认只产生一次最终副作用。
6. 报销修改费用后必须重新生成预览；IT 权限申请走同一执行器。

已有普通 Agent、工具调用和报销回归测试按受影响部分运行即可。非关键的全服务启动、监控或模型真实调用失败可留到后续迭代。

## 验收标准与简历边界

验收时应能通过 Swagger 或测试调用演示：知识库/记忆从 PostgreSQL + pgvector 检索；报销提交和 IT 权限申请均显示预览、显式确认、幂等重放和审计事件；模型没有直接调用最终写工具的路径。

完成后可准确写为“设计面向副作用工具的 HITL 执行协议，模型仅生成结构化操作提案，服务端完成预校验、预览确认与幂等提交，并以 PostgreSQL + pgvector 持久化向量检索与操作审计”。不应声称已接入真实外部业务系统、实现分布式 exactly-once 或取得未经测量的性能指标。
