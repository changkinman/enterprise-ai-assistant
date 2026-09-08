# Agent 工具扩展约定

`tools.loader.discover_agent_tools()` 会自动加载本目录下所有以 `_tool.py` 结尾的模块。
新增工具时不需要修改 Agent、API 或中央注册表，只需增加一个模块：

```python
from mcp.tool_manager import Tool

async def handler(params, context):
    return {"ok": True}

def build_tools():
    return [
        Tool(
            name="example_tool",
            description="清晰说明模型应在什么场景调用该工具",
            handler=handler,
            schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            allowed_agents=frozenset({"general"}),
        )
    ]
```

约束：

- 模块必须提供 `build_tools()`，并返回非空的 `Tool` 或 `Tool` 列表。
- 工具名称必须唯一；重复名称会导致启动失败。
- `allowed_agents` 只能包含 `general`、`technical`、`billing`。
- `context` 中的 `user_id`、`conv_id`、`request_id` 和 `agent_type` 来自服务端可信上下文。
