# HarnessVLN Nexus

Nexus 是一个 Agent 主导的视觉语言导航 Harness 底座。Runner 只调度完整导航 Domain，
Domain 内的 Agent、Environment、VLN、Memory 和 Metric 只通过函数调用协作。

```text
Runner -> Bench -> Domain worker processes -> DomainResult -> Bench aggregate

DomainRuntime
|-- FunctionBus
|-- Environment       required, owns world state and nav.stop
|-- Agent             called once, owns the navigation loop
|-- VLN / Memory      optional function components
`-- Metric            selected by Bench, evaluated after terminal
```

没有全局 observe-act Runner 循环，没有发布订阅层，也没有模拟器或模型特例进入核心。

## 运行 Dummy Stack

```bash
./scripts/run_dummy.sh --run-id first-run
```

等价的直接入口：

```bash
PYTHONPATH=src python -m harness.cli \
  --runner-config config/runners/dummy.yaml \
  --run-id first-run
```

Dummy Stack 包含一个异步 VLN Job、空间 Memory、Agent、Environment 和 Metric。它用于验证
框架契约，不代表真实导航算法。运行结果位于 `runs/nexus_dummy/<run-id>/`。

## 配置关系

```text
config/runners/dummy.yaml
|-- config/agents/dummy.yaml
|   |-- config/vln/dummy.yaml
|   `-- config/memory/dummy.yaml
`-- config/benches/dummy.yaml
    |-- config/envs/dummy.yaml
    `-- config/metrics/dummy.yaml
```

Runner 可以引用多个 Bench。每个 Bench 明确引用自己的 Environment 和 Metric；VLN、Memory
及其他导航能力属于 Agent 的组件组合。工厂统一使用 `module:attribute`，没有插件注册表。

每个 worker 可以固定进程环境，例如多 GPU 任务可分别设置
`CUDA_VISIBLE_DEVICES: "0"` 和 `CUDA_VISIBLE_DEVICES: "1"`。环境变量在 worker 创建时注入，
不会在同一个已初始化 CUDA 的进程中切换设备。

## 目录

```text
src/harness/       Domain、函数总线、配置、输出和进程 Runner
src/schemas/       跨组件导航数据
src/agents/*/      Agent 实现
src/envs/*/        模拟器或真机中间件
src/vln/*/         完整 VLN 模型适配器
src/memory/*/      导航记忆
src/metrics/*/     Bench 指定的评分组件
src/benches/*/     Case 枚举与结果聚合
```

核心代码直接位于 `src/harness/`；每个具体实现必须位于自己的三级目录。

## 核心约束

- 一个 Domain 对应一次环境生命周期，不强制对应单个 Goal。
- Environment 必须且只能有一个，并且必须提供 `nav.stop`。
- Agent 的 `run(context)` 每个 Domain 只调用一次。
- 组件只获得 `required_functions` 与当前存在的 `optional_functions`。
- 环境写函数通过相同 `serial_key` 串行，终止后 FunctionBus 拒绝新写入。
- Metric 可以获得 evaluator-only 函数，Agent 和 VLN 不会自动看到它们。
- 每个组件只写自己的输出目录，Runner 不读取组件内部状态。
- 多 Domain 在固定资源槽的 worker 进程中并行，worker 环境变量在进程启动时注入。

架构细节见 [Nexus architecture](docs/architecture/nexus.md)，扩展步骤见
[Adding a component](docs/extending/component.md)。

## 验证

```bash
python -m pytest -q
python -m mypy src
python -m compileall -q src tests
git diff --check
```

基础环境定义在 `config/conda/harnessvln.yaml`，Python 要求为 3.10 及以上。
