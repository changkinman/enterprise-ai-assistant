# 差旅报销 Agent Harness 设计

**状态：已确认，待实施**
**目标：** 使用最小可演示的差旅报销业务沙盒，展示 EchoMind 在有状态任务中的可靠 Agent Harness，而不是建设完整财务系统。

## 1. 背景与目标

当前 EchoMind 已具备意图识别、任务规划、Agent 编排、RAG、对话记忆、工具治理、评测和监控能力，但财务领域的业务工具仅能查询模拟账单，主要产出仍是对话回答。

本次改造以“员工办理差旅报销”为单一业务场景。员工可用自然语言描述出差费用，Agent 收集缺失信息、解释制度、校验规则与预算、创建报销草稿，并在员工确认后提交模拟审批。主管驳回或员工补件后，任务从既有案件继续推进。

成功标准是可清楚演示以下 Harness 能力：

- Agent 基于案件当前状态选择下一步，而不是自由调用工具；
- 所有工具仍通过 `MCPToolManager.call()` 执行并接受既有治理；
- 有副作用的动作经过显式确认、状态校验和幂等控制；
- 信息不足、校验失败、驳回补件和工具失败均可恢复；
- 每个决策、工具调用与状态变化均留下可评测的轨迹。

## 2. 范围和非目标

### 范围

- 差旅报销案件、费用明细、部门预算、审批记录、模拟付款状态；
- 一个员工、直属主管、财务处理人三角色的模拟流程；
- Agent 信息抽取、澄清、制度解释和下一步动作决策；
- 状态化工具调用、人工确认、断点恢复、审计轨迹与场景评测；
- 用于展示的案件查询、模拟审批/驳回、模拟付款 API。

### 非目标

- ERP、OA、企业微信、真实支付或真实审批系统集成；
- OCR、电子发票验真、复杂会计科目、税务处理或多币种结算；
- 完整管理后台、前端工作台或生产级权限体系；
- 为未来系统集成而预建复杂适配器层；
- 扩大到采购、HR、IT 等其他员工服务场景。

业务数据采用进程内轻量仓储和预置演示数据。服务重启后丢失数据在本期可接受；设计以清晰的仓储接口隔离，使其日后可替换。

## 3. 总体架构

```text
用户消息
  -> IntentRecognizer / TaskPlanner
  -> AgentOrchestrator
  -> ReimbursementCaseHarness（读取或创建案件）
  -> BillingAgent（提取事实、生成澄清或下一步工具意图）
  -> Harness（状态、字段、确认与幂等校验）
  -> MCPToolManager.call()
  -> 报销工具 / 轻量仓储
  -> 更新案件、事件和工具轨迹
  -> 回复用户，或进入等待确认/审批状态
```

`BillingAgent` 负责语言理解和解释，不拥有绕过 Harness 的权限。Harness 是业务执行的权威：它仅允许在当前案件状态中合法的工具和动作。`MCPToolManager` 继续是工具执行的唯一入口，保留参数校验、Agent 权限、缓存、超时、熔断和降级语义。

不新增“为了多 Agent 而多 Agent”的角色。第一期仅使用现有财务 Agent；政策、预算和审批由受治理的工具及确定性业务规则完成。这样可以把复杂性集中在可解释、可测试的 Harness，而非提示词协作。

## 4. 案件模型与状态机

每次报销都有稳定的 `case_id`，并包含：申请人、部门、结构化费用项、已知事实、缺失字段、政策/预算校验结果、确认标记、审批记录、付款状态、幂等键和执行事件。

状态机固定为：

```text
collecting_info
  -> draft_created
  -> validating
  -> awaiting_confirmation
  -> submitted
  -> paid

validating / submitted
  -> rework_required
rework_required
  -> collecting_info

任意未关闭状态 -> closed（员工取消或不可恢复错误）
```

状态含义：

- `collecting_info`：收集城市、日期、费用、项目/客户等必填信息；只允许更新事实和提出澄清。
- `draft_created`：创建了唯一的报销草稿；可继续增加或修正费用项。
- `validating`：可运行制度、票据和预算校验；异常必须被写入案件。
- `awaiting_confirmation`：校验已完成，等待员工明确确认提交。
- `submitted`：模拟主管/财务审批中；可查询状态，不允许重复提交。
- `rework_required`：驳回或校验失败后的补件状态；补齐后返回收集或校验。
- `paid` 与 `closed`：终态，不再允许改变报销内容。

Harness 在每次动作前验证状态、角色和必填字段。工具调用失败不会推进状态；可重试动作使用 `case_id + action + version` 生成幂等键，确保 LLM 重复调用不会创建重复草稿、重复占用预算或重复提交。

## 5. 最小工具集与确认策略

工具统一放在 `tools/reimbursement_tool.py` 并由现有自动发现机制注册。所有写操作接收 `case_id` 与幂等键；工具返回结构化结果，不能自行改变 Harness 以外的状态。

| 工具 | 用途 | 影响等级 |
|---|---|---|
| `search_policy` | 查询差旅制度依据 | 只读 |
| `get_case` | 读取案件、状态和待办 | 只读 |
| `check_budget` | 查询可用预算 | 只读 |
| `create_draft` | 为案件创建唯一草稿 | 可幂等写入 |
| `add_expense_items` | 写入或修正费用明细 | 可幂等写入 |
| `validate_claim` | 执行政策、票据、预算规则校验 | 可幂等写入 |
| `submit_claim` | 原子地占用预算并提交模拟审批 | 副作用，需确认 |
| `review_claim` | 模拟主管/财务审批或驳回 | 副作用，仅审批角色 |
| `simulate_payment` | 模拟付款 | 副作用，仅财务角色 |

为控制规模，预算占用是 `submit_claim` 的内部确定性步骤，而不是独立工具；最终代码暴露九个报销工具。用户确认仅认可明确的提交意图，例如“确认提交这笔报销”，不能由模型根据“好的”之类歧义输入自行推断。

## 6. 一次任务的执行流

1. Harness 用 `user_id + conv_id` 查找未关闭案件；若无匹配案件，创建 `collecting_info` 案件。
2. BillingAgent 从输入中提取候选事实，Harness 做类型、范围和必填字段校验。
3. 缺字段时，Harness 保存已提取事实并输出一个最小澄清问题；后续消息恢复同一案件。
4. 信息足够后创建草稿并运行只读查询与校验，保存制度依据、超标原因和预算结果。
5. Agent 对可继续项生成摘要；Harness 转为 `awaiting_confirmation`。超标、超预算或缺票时转为 `rework_required`，给出可执行的补件/修改建议。
6. 员工明确确认后，Harness 放行预算占用和提交动作，案件转为 `submitted`。
7. 演示 API 或指定角色调用审批、驳回和付款工具。驳回进入 `rework_required`；付款成功进入 `paid`。

每一步都写入 `ExecutionEvent`：时间、案件版本、Agent 建议、允许/拒绝的动作、工具参数摘要、结果、错误和状态迁移。事件是调试、展示和评测的事实来源。

## 7. 代码边界

新增：

- `core/reimbursement_models.py`：枚举和纯数据模型；
- `core/reimbursement_repository.py`：进程内仓储、预置员工/预算/规则数据；
- `core/reimbursement_harness.py`：状态迁移、字段校验、确认门禁、恢复和幂等；
- `tools/reimbursement_tool.py`：工具定义与仓储调用；
- `tests/test_reimbursement_harness.py`、`tests/test_reimbursement_tools.py`；
- `evaluation/run_reimbursement_harness_benchmark.py` 与配套 fixtures。

有针对性地修改：

- `agents/agent_orchestrator.py`：把 billing 任务交给 Harness 驱动，保留现有路由、统计和 `ToolCallTrace` 输出；
- `api/main.py`：增加案件查询、模拟审批/驳回和模拟付款的最小演示接口；
- `evaluation/evaluator.py` 与 `monitor/performance_monitor.py`：接收 Harness 任务指标，避免改变原有评测/监控语义。

不应将案件状态塞入 `ConversationMemory` 的自由文本上下文，也不应绕过 `MCPToolManager.call()` 直接调用工具。

## 8. 错误处理与恢复

- 参数或状态不合法：拒绝动作，保留案件状态，记录拒绝事件，并向用户解释下一可行动作。
- 工具超时、熔断或失败：由 `MCPToolManager` 保持现有降级行为；Harness 不推进状态，标记可重试并提供查询/稍后重试选项。
- 模型输出无效或试图跳过状态：Harness 拒绝该动作，回退为澄清或安全的只读状态查询。
- 用户补件：在 `rework_required` 中合并新事实，清除过期校验结果后重跑校验。
- 重复消息或模型重试：同一幂等键返回先前结果，不产生二次副作用。

## 9. 评测、监控与验收

新增确定性场景：正常提交、缺少项目编号、酒店超标、预算不足、主管驳回后补件、工具超时、重复提交、终态继续修改。

验收指标：

- 任务完成率：给定完整且合规输入能到达 `submitted`，审批付款后到达 `paid`；
- 工具正确率：非法状态或字段缺失时不会调用写工具；
- 确认安全率：会占用预算的 `submit_claim` 只能在显式确认后调用；
- 恢复率：补件、工具失败和重复输入均保持案件一致；
- 政策一致性：超标、超预算和缺票案例被拦截并留下依据；
- 轨迹完整率：每次状态迁移均有关联执行事件和工具轨迹。

现有监控增加 Harness 维度的完成率、澄清轮次、状态拒绝次数、工具失败恢复次数和人工确认次数。它们用于展示可靠性，不宣称未经基准验证的性能提升。

## 10. 分阶段交付

1. **确定性内核**：实现案件模型、内存仓储、状态机、工具和单元测试；无 LLM 也能完整验证流程。
2. **Agent 接入**：让 BillingAgent 输出受约束的事实和下一步建议，接入 Harness 与现有轨迹返回；不得改变工具治理入口。
3. **展示证据**：补充模拟审批 API、评测 fixtures、Harness 指标和运行报告。

完成标准是可通过自动化场景复现完整报销链路，并能在面试中解释：模型负责何种判断，Harness 如何限制副作用，工具治理如何工作，失败和补件如何恢复，以及用什么证据证明这些行为。
