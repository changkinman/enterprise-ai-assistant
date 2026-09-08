"""按约定自动发现并注册 Agent 工具。"""
import importlib
import pkgutil
from typing import List

from mcp.tool_manager import MCPToolManager, Tool


def discover_agent_tools(
    manager: MCPToolManager,
    package_name: str = "tools",
) -> List[str]:
    """
    加载 package_name 下所有 ``*_tool.py`` 模块。

    每个模块必须暴露 ``build_tools()``，返回 Tool 或 Tool 列表。
    模块加载、定义校验和重名错误都会直接向上传递，使服务启动失败。
    """
    package = importlib.import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        raise RuntimeError(f"工具包不可扫描: {package_name}")

    registered: List[str] = []
    module_names = sorted(
        info.name
        for info in pkgutil.iter_modules(package_paths)
        if info.name.endswith("_tool")
    )

    for module_name in module_names:
        full_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_name)
        factory = getattr(module, "build_tools", None)
        if not callable(factory):
            raise RuntimeError(f"工具模块缺少 build_tools(): {full_name}")

        definitions = factory()
        if isinstance(definitions, Tool):
            definitions = [definitions]
        if not isinstance(definitions, (list, tuple)) or not definitions:
            raise RuntimeError(f"工具模块未返回有效 Tool: {full_name}")

        for tool in definitions:
            if not isinstance(tool, Tool):
                raise RuntimeError(f"工具模块返回了非 Tool 对象: {full_name}")
            manager.register(tool)
            registered.append(tool.name)

    return registered
