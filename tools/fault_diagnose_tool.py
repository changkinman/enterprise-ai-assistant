"""后台日志故障诊断模拟工具。"""
from typing import Any, Dict, Optional

from mcp.tool_manager import Tool


async def fault_diagnose(
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """从可信上下文取得用户身份并返回模拟日志诊断。"""
    trusted_context = context or {}
    user_id = str(trusted_context.get("user_id", "anonymous"))
    symptom = str(params.get("symptom", "")).strip()

    return {
        "mock": True,
        "user_id": user_id,
        "symptom": symptom or "用户未提供具体故障现象",
        "error_code": "DB_POOL_EXHAUSTED",
        "severity": "high",
        "cause": "数据库连接池耗尽，导致部分后台请求超时。",
        "log_excerpt": [
            "2026-07-11T09:31:02+08:00 ERROR request timeout after 3000ms",
            "2026-07-11T09:31:02+08:00 WARN db pool active=20 idle=0 waiting=14",
        ],
        "suggestions": [
            "稍后重试当前操作",
            "检查慢查询并确认数据库连接是否及时释放",
            "必要时临时扩容连接池并持续观察等待队列",
        ],
    }


def build_tools() -> list[Tool]:
    """创建本模块提供的工具定义。"""
    return [
        Tool(
            name="fault_diagnose",
            description=(
                "读取当前用户的模拟后台日志并诊断故障原因。"
                "当用户遇到登录失败、接口报错、超时或系统异常并要求排查时调用。"
            ),
            handler=fault_diagnose,
            schema={
                "type": "object",
                "properties": {
                    "symptom": {
                        "type": "string",
                        "description": "用户描述的故障现象；未知时可省略",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            cache_ttl=0.0,
            allowed_agents=frozenset({"technical", "general"}),
        )
    ]
