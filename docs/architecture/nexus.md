# HarnessVLN Nexus

Nexus 是 HarnessVLN 的最小导航运行时。它不实现某一种 VLN 方法，而是为 Agent、
Environment、VLN、Memory、Metric 以及传统导航模块提供稳定的组合和执行边界。

## 设计目标

- Agent 主导完整导航过程，Runner 不执行 `observe -> act` 循环。
- 一个 Domain 表示一段独立的环境生命周期，多个 Domain 以进程为默认并行单位。
- Environment 是 Domain 中唯一必选组件，也是环境状态的唯一写入者。
- 所有业务协作只经过具名函数调用，不引入发布订阅、消息 Actor 或全局依赖容器。
- 组件依赖函数能力，而不依赖其他组件的 Python 类型或具体实现。
- 固定工作流 Agent 与自由 Agent Loop 实现同一个 `Agent.run(context)` 契约。
- 上层适配器自行保存模型输出、视频和调试产物，Runner 不读取组件内部状态。

## 控制面与 Domain

```text
Runner -> Bench.cases() -> DomainProcess x N -> DomainResult -> Bench.aggregate()

DomainRuntime
|-- FunctionBus
|-- Environment       exactly one
|-- Agent             one driver in benchmark mode
|-- VLN               optional component
|-- Memory            optional component
|-- Metric            zero or more, selected by Bench
`-- other components  optional
```

Runner 只枚举任务、分配资源、启动 Domain、收集结果和汇总进度。Bench 只定义 Case、
选择环境与 Metric、保存私有评测数据并聚合结果。两者都不调用观测或动作函数。

一个 Domain 对应一次环境 session，而不强制对应一个 Goal。R2R 的一个 episode 通常是
一个 Domain；GOAT 的连续 Goal 可以共享同一个 Domain，完成单个 Goal 不会 reset 环境；
交互模式的 Domain 则一直存在到显式 stop。

## 组件契约

普通组件只需要声明依赖、提供函数并实现生命周期：

```python
class Component(ABC):
    name: str
    required_functions: frozenset[str]

    def functions(self) -> Sequence[Function]: ...
    async def start(self, context: ComponentContext) -> None: ...
    async def close(self, reason: str) -> None: ...
```

`functions()` 只描述并绑定函数，不启动后台工作。Domain 先注册所有函数并校验依赖，
再启动组件，因此启动顺序不会决定函数是否可发现。组件可以是被动函数服务，也可以在
`start()` 内建立自己的异步任务。框架不要求所有组件具有循环或独立线程。

Agent 额外实现一次完整的 `run(context)`。Environment 额外提供终止等待和最终事实；
Metric 额外提供最终计算。这些是明确的角色契约，不在 Runner 中通过类型分支模拟。

## FunctionBus

FunctionBus 是 Domain 内唯一的组件协作通道，负责：

1. 函数名与 schema 注册；
2. 调用方的 `required_functions` 白名单；
3. 输入输出校验；
4. 每个资源键的串行写入；
5. 调用编号、耗时、结果和错误审计；
6. Domain 终止后的写入屏障。

函数按稳定导航能力命名，而不携带模拟器名称：

```text
nav.observe
nav.move
nav.stop
vln.navigate.start
vln.navigate.status
vln.navigate.cancel
memory.search
memory.remember
```

OpenAI Responses 等原生 function calling 只需把 Agent 可见的 Function schema 交给模型，
再把原生 function call 交给同一个 FunctionBus。固定工作流也使用同一个 client，避免
模型工具调用和 Python 直调形成两套协议。

## Environment 与终止

Environment 必须提供 `nav.stop`，但公开名称属于导航能力，不属于 Habitat 或 Isaac 等
后端。动作函数共享环境串行键，Environment 在内部再次保护原生状态机。

`nav.stop` 与 `Environment.close()` 语义不同：前者结束导航，后者释放资源。stop 必须
幂等，第一次终止声明生效；调用开始后拒绝新动作，排空已经进入原生环境的动作，然后
冻结最终状态。环境原生终止、Agent 异常、超时和外部取消最终都收敛到同一 terminal。

DomainRuntime 负责停止其他组件和关闭 FunctionBus，但不替 Environment 决定环境事实。
若 Agent 正常返回却没有调用 stop，Domain 将其视为协议失败并主动终止环境。

## 并行

并行包含两个层次：

- Runner 在多个进程中运行完整 Domain；Domain 之间没有共享 FunctionBus 或可变环境状态。
- 单个 Domain 内，Agent、VLN Job 和组件后台任务可按各自频率并发；函数 handler 不因此
  自动获得并发安全性。

Environment 写操作始终串行。完整 VLN Job 若会自主移动，应在组件内部或环境适配器中
持有运动控制权；Agent 可以查询和取消 Job，但不能与其同时提交冲突动作。第一版只提供
实现该策略所需的函数与串行边界，不在核心中建立复杂调度器。

## Metric 与输出

Metric 是 Domain 组件，但由 Bench 构造。它只获得私有评测函数，例如最终轨迹和原生
Measure；Agent 与 VLN 的函数白名单中不包含这些能力。基础 Metric 在终止后计算，实时
更新不是核心要求。

每个组件获得独立输出目录和最小写入接口：Environment 保存视频与轨迹，Agent 保存模型
调用，VLN 保存推理记录，Metric 保存评分细节。DomainRuntime 只合并各组件提交的 manifest，
Runner 只处理 `DomainResult`，从而避免输出逻辑把 Runner 与所有组件耦合。

## 生命周期

```text
construct -> register functions -> validate dependencies
          -> start environment -> start services -> run agent once
          -> wait for environment terminal / agent failure / timeout
          -> close writes -> drain -> freeze environment
          -> evaluate metrics -> close components in reverse order
          -> persist DomainResult
```

核心只保留 `DomainRuntime`、`FunctionBus`、组件协议、终止状态和结果类型。配置加载、
进程调度与输出管理建立在这些契约之上；具体模拟器、模型和 Agent Loop 留在各自三级目录。

## 实现映射

| 责任 | 文件 |
|---|---|
| 函数注册、schema、白名单、串行键、审计 | `src/harness/functions.py` |
| Component、Agent、Environment、Metric 契约 | `src/harness/components.py` |
| 单 Domain 生命周期与收敛 | `src/harness/domain.py` |
| 组件隔离输出与 artifact 校验 | `src/harness/output.py` |
| YAML 引用与工厂描述 | `src/harness/config.py` |
| 固定资源 worker 与多 Domain 调度 | `src/harness/runner.py` |
| Case 与 Bench 聚合边界 | `src/benches/base.py` |

当前 Dummy 纵向链路只用于证明以上边界真实可运行。Habitat、AI2-THOR、Isaac Sim 以及
真实 VLN 模型应作为后续独立模块接入，不改变 Domain 或 Runner 的控制流。
