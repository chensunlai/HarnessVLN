# 添加 Nexus 组件

具体实现放在对应类别的三级目录，例如：

```text
src/vln/my_vln/__init__.py
src/vln/my_vln/navigator.py
config/vln/my_vln.yaml
```

## 最小组件

```python
from harness import Component, ComponentContext, Function


class Planner(Component):
    name = "planner"
    required_functions = frozenset({"nav.observe"})
    optional_functions = frozenset({"memory.search"})

    def functions(self):
        return (
            Function(
                name="planner.plan",
                description="Build a navigation plan.",
                handler=self._plan,
                input_schema={"type": "object"},
            ),
        )

    async def start(self, context: ComponentContext) -> None:
        self.context = context

    async def _plan(self, call, arguments):
        observation = await self.context.functions.call("nav.observe")
        return {"observation": observation}

    async def close(self, reason: str) -> None:
        pass
```

函数 handler 接收调用上下文和已经通过 JSON Schema 校验的参数。函数结果若声明了
`output_schema`，也会在返回调用方之前校验。

`mutates=True` 表示函数会改变状态；操作相同资源的函数使用相同 `serial_key`。Environment
的移动与 stop 应共享环境串行键，组件内部仍需遵守原生后端自己的状态机。

## 角色契约

- `Agent.run(context)`：一次完整导航过程，必须主动调用 `nav.stop`。
- `Environment.wait_terminal()`：等待显式 stop 或后端原生终止。
- `Environment.result()`：返回终止后可保存的环境事实。
- `Metric.evaluate(terminal, environment)`：读取私有评测能力并返回数值指标。
- 普通 `Component`：提供函数或管理后台 Job，不获得额外运行时分支。

完整 VLN 应把自己的观测、推理、动作频率和内部状态保留在组件内。推荐向 Agent 提供
`start/status/cancel` Job 函数，而不是要求 Runner 或 Agent 每个模型 step 调用一次。

## 输出

组件只能通过 `context.output` 写自己的目录：

```python
context.output.append_jsonl("inference/trace.jsonl", record)
context.output.add_artifact("inference/trace.jsonl", "application/jsonl")
```

Domain 结束后会验证声明的 artifact 是否真实存在。Environment 保存视频和轨迹，Agent
保存模型交互，VLN 保存推理记录；Runner 只汇总 manifest 和 `DomainResult`。

## 配置

```yaml
factory: vln.my_vln:MyVLN
parameters:
  checkpoint: model/my_vln
```

工厂必须返回相应的 `Component` 角色实例。配置加载阶段拒绝未知字段；Domain 启动前完成
函数重名、依赖缺失、组件重名及 `nav.stop` 所有权校验。
