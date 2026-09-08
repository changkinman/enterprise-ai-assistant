"""账单信息查询模拟工具。"""
from typing import Any, Dict, Optional

from mcp.tool_manager import Tool


async def billing_lookup(
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """根据账单号返回确定性的模拟账单数据。"""
    billing_no = str(params.get("billing_no", "")).strip()
    if not billing_no:
        raise ValueError("billing_no 不能为空")

    return {
        "mock": True,
        "billing_no": billing_no,
        "amount": 199.00,
        "currency": "CNY",
        "status": "paid",
        "billing_period": "2026-06",
        "paid_at": "2026-07-01T10:00:00+08:00",
        "items": [
            {"name": "EchoMind 专业版订阅", "quantity": 1, "amount": 199.00},
        ],
    }


def build_tools() -> list[Tool]:
    """创建本模块提供的工具定义。"""
    return [
        Tool(
            name="billing_lookup",
            description=(
                "根据用户提供的账单号查询账单详情。"
                "当用户询问具体账单的金额、状态、账期或明细时调用。"
            ),
            handler=billing_lookup,
            schema={
                "type": "object",
                "properties": {
                    "billing_no": {"type": "string", "description": "需要查询的账单号"},
                },
                "required": ["billing_no"],
                "additionalProperties": False,
            },
            cache_ttl=60.0,
            allowed_agents=frozenset({"billing", "general"}),
        )
    ]
