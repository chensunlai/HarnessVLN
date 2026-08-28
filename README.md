<p align="center">
  <img src="docs/harnessvln-mark.png" alt="HarnessVLN" width="112">
</p>

<h1 align="center">HarnessVLN</h1>

<p align="center">
  Agent 主导、模块化、可组合的视觉语言导航实验与评测基座。
</p>

<p align="center">
  <a href="https://chensunlai.github.io/HarnessVLN/">完整文档</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#架构">架构</a> ·
  <a href="#扩展项目">扩展项目</a>
</p>

HarnessVLN 将完整导航任务的控制权交给 Agent：Runner 负责选择 Bench、并行调度完整
Task 和汇总结果，但不在任务内部执行固定的 `observe -> predict -> step` 循环。Agent 可以
自主观察、移动、调用完整 VLN 模型、读写空间记忆、切换 Goal，并主动结束任务。

项目面向视觉语言导航模型、通用导航 Agent 和模拟器适配器的组合实验。组件通过 YAML 与
`module:factory` 动态装配，无需在核心框架中维护模型或环境分支。

> [!IMPORTANT]
> 本项目仍处于研究开发阶段。“已有 adapter”不等于“已完成官方评测对齐”。请根据下方
> [验证状态](#当前实现与验证状态)和[在线兼容矩阵](https://chensunlai.github.io/HarnessVLN/reference/compatibility.html)
> 区分接口测试、真实 smoke test 与完整 split parity。

## 核心特性

- **Agent 拥有任务循环**：每个 Task 只调用一次 `Agent.run(context)`，任务内决策全部由
  Agent 完成。
- **完整 VLN 插件**：模型自己的缓存、线程、推理频率和轨迹状态保留在插件内部；真实模型
  通过可取消的 RPC worker 接入。
- **稳定的环境边界**：Habitat、AI2-THOR、Isaac 等原生对象只存在于 Environment 中，
  对外暴露类型化导航工具。
- **Bench 与真值隔离**：Bench 加载 case 并保管评分真值，Agent 只接收公开的 `NavTask`。
- **ToolBus 约束调用**：统一处理 JSON Schema 校验、工具权限、调用审计和停止后的运动写屏障。
- **完整 Task 粒度并发**：支持单进程异步并发与多 GPU 进程池，不拆分单个 Task 的内部循环。
- **可追溯输出**：Manifest v3 保存解析后配置、配置摘要、provenance、聚合指标和分层结果；
  组件还可输出事件流、视频及逐帧元数据。

## 架构

<p align="center">
  <img src="docs/architecture-overview.png" alt="HarnessVLN architecture" width="900">
</p>

```text
Runner -> BenchmarkCase -> NavigationHarness.run_task(task, stack)
                                      |
                                      `-> Agent.run(context)       # 每个 Task 一次
                                            |-- nav.observe / move
                                            |-- vln.navigate.*
                                            |     `-- VLN worker 反向调用受限工具
                                            |-- spatial.search / remember
                                            |-- nav.goal.finish
                                            `-- nav.stop

Bench: case、私有真值、评分       Environment: 模拟器/真机控制权
Runner: 完整 Task 调度与汇总      ToolBus: schema、权限、审计、写屏障
```

核心运行时与插件之间只依赖最小 Protocol：`NavigationAgent`、`VLNNavigator`、
`Environment` 和 `SpatialMemory`。启动时还会使用 `NavigationProfile` 检查观测通道、动作、
坐标系和相机参数是否兼容。

## 当前实现与验证状态

| 范围 | 已接入实现 | 当前验证边界 |
|---|---|---|
| Agent | `PassthroughVLNAgent`、基于 Responses API 工具循环的 `NormalAgent` | Dummy 闭环与单元测试；真实效果取决于所选模型和导航栈 |
| VLN | Dummy、StreamVLN、JanusVLN、DualVLN | 三个真实模型有 R2R-CE 固定小样本 trace；官方同版本完整 split parity 待完成 |
| Environment | Dummy、Habitat-Lab/Sim、AI2-THOR/RoboTHOR、Isaac Sim/InternUtopia | Habitat 与 RoboTHOR 有真实 smoke；Isaac 组合目前主要为接口和 data contract |
| Bench | Dummy、R2R-CE、GOAT、Habitat ObjectNav、RoboTHOR ObjectNav、VLN-PE、VLNVerse | 各组合验证等级不同，不能仅凭 adapter 存在推断完整兼容 |
| Memory | 可持久化的 Dummy Landmark Memory | 支持跨 Task 查询和原子 JSON 写回；不包含拓扑图、占据图或 embedding 检索 |

VLN-PE 与 VLNVerse 所需的专用 scene、episode、H1 资产和 policy 当前不随仓库提供；相关配置
用于 adapter 与数据契约开发。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/chensunlai/HarnessVLN.git
cd HarnessVLN
```

### 2. 运行最小 Dummy 闭环

Dummy 链路不需要 GPU、数据集、模型权重或模拟器。使用 Python 3.10+ 创建轻量环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  PyYAML==6.0.2 \
  jsonschema==4.26.0 \
  imageio-ffmpeg==0.6.0 \
  tqdm==4.70.0

bash scripts/run_dummy.sh
```

成功时会得到类似输出：

```text
dummy_navigation/smoke: 2 cases, 0 task failures, 0 case errors, 0 bench errors, 0 cleanup errors, 0 output errors
/absolute/path/to/runs/dummy_passthrough/<run-id>/manifest.json
```

该示例与真实任务共用 Runner、Harness、Agent、VLN Job、ToolBus、评分和 Manifest 路径。
也可以启动单环境交互：

```bash
bash scripts/run_dummy_interactive.sh
```

输入导航指令创建 Task，输入 `quit` 结束会话。

### 3. 完整研究环境

包含真实模型与模拟器依赖的固定环境使用 Python 3.10：

```bash
conda env create -f config/conda/harnessvln.yaml
conda activate harnessvln
export PYTHONPATH="$PWD/src"
```

完整环境包含 Habitat、AI2-THOR、Isaac、模型推理与视频输出相关依赖，安装体积较大；CUDA、
驱动和模拟器运行条件仍需与本机硬件匹配。

## 运行实验

批量运行只接收一份 Runner 配置和一份 Agent 配置。Runner 配置引用 Bench，Bench 引用
Environment；Agent 配置引用可选的 VLN 与 Memory：

```text
Runner -> Bench -> Environment
CLI    -> Agent -> VLN?
                -> Memory?
```

显式运行 Dummy：

```bash
PYTHONPATH=src python -m harness.cli run \
  --runner config/runners/dummy_passthrough.yaml \
  --agent config/agents/passthrough.yaml
```

交互运行一个 Environment：

```bash
PYTHONPATH=src python -m harness.cli env \
  --environment config/envs/dummy.yaml \
  --agent config/agents/passthrough.yaml
```

准备好 R2R-CE 数据、MP3D 场景、上游源码、对应 checkpoint 和 GPU 后，可运行已经固定组合的
脚本：

```bash
bash scripts/run_r2r_streamvln.sh
bash scripts/run_r2r_janusvln.sh
bash scripts/run_r2r_dualvln.sh
bash scripts/run_r2r_dualvln_2gpu.sh
```

这些脚本默认指向真实 `val_unseen` 配置。首次验证建议先使用
`config/runners/smoke_one.yaml` 的单 case StreamVLN 组合。数据、模型与上游仓库不会随本仓库
分发；配置中的路径和 `provenance` 固定了预期资源及版本。

`NormalAgent` 在未注入测试 client 时使用 OpenAI Python SDK 的 Responses API。运行对应 Agent
配置前，需要按 SDK 约定提供 API 凭据；模型名、工具面、推理强度和预算均由 Agent YAML 控制。

## 配置系统

组件配置的最小形式如下：

```yaml
agent:
  factory: agents.passthrough:PassthroughVLNAgent
  params: {}
  vln: ../vln/dummy.yaml
```

- `factory` 使用 `module:object` 动态加载插件；
- `params` 原样传入 factory；
- `scope` 为 `task` 或 VLN 可用的 `session`；
- `serial` 声明进程级或共享资源必须串行；
- `extends` 可在同类 YAML 之间做深度覆盖；列表整体替换；
- 配置通过 JSON Schema Draft 2020-12 校验后，会生成 canonical digest 并写入 Manifest。

多 GPU slot、任务并发、超时、case 上限与输出路径均由 Runner YAML 控制。详见
[配置引用与继承](https://chensunlai.github.io/HarnessVLN/usage/configuration.html)。

## 输出结构

每次运行创建独立目录：

```text
runs/<experiment>/<run-id>/
├── manifest.json
├── config/
│   ├── resolved.yaml
│   └── sources.json
└── benches/<bench-id>/
    ├── summary.json
    └── episodes/<episode-id>/
        ├── result.json
        ├── events.jsonl
        ├── environment.json
        ├── components/
        └── artifacts/
```

`manifest.json` 是运行级索引；Bench summary 和 Episode result 保存详细状态、指标、错误分类、
资源信息及组件产物。详见[结果与 Manifest](https://chensunlai.github.io/HarnessVLN/usage/results-manifest.html)。

## 仓库结构

```text
HarnessVLN/
├── src/
│   ├── harness/      # CLI、配置、Runner、生命周期、ToolBus、输出
│   ├── schemas/      # 跨插件的导航数据类型与能力描述
│   ├── agents/       # Agent Core
│   ├── vln/          # VLN navigator、worker 与 RPC
│   ├── envs/         # 模拟器/服务中间件
│   ├── benches/      # case loader 与评分
│   └── memory/       # 空间记忆插件
├── config/           # Agent、VLN、Environment、Bench、Runner 配置
├── scripts/          # 固定组合的运行入口
├── tests/            # unit、contract、integration 测试
├── docs/             # GitHub Pages 静态文档
├── data/             # 本地数据与场景，不提交
├── model/            # 本地模型权重，不提交
├── cache/            # 上游源码与缓存，不提交
└── runs/             # Manifest 与运行产物，不提交
```

## 扩展项目

新增插件通常只需实现对应 Protocol、声明 `required_tools`/requirements、提供 factory 并添加
YAML，无需修改中央注册表。建议按以下顺序接入：

1. 用最小 fake/native session 覆盖插件生命周期与错误路径；
2. 添加配置 schema、requirements 和资源路径检查；
3. 完成 `reset -> observe -> act -> finish -> stop` 真实 smoke；
4. 保存可复核的小样本结果，再进行完整 split 与官方 evaluator 对齐。

开发入口见[插件契约](https://chensunlai.github.io/HarnessVLN/extending/plugin-contract.html)。

## 测试

```bash
PYTHONPATH="$PWD/src:$PWD" conda run -n harnessvln pytest -q
```

测试按风险分为：

- `tests/unit/`：生命周期、竞态、ToolBus、Runner、RPC 与各 adapter；
- `tests/contract/`：数据 loader、模型配置和可选真实 trace；
- `tests/integration/`：从 YAML 到 Manifest 的完整闭环。

修改共享 runtime、ToolBus、RPC 或 schema 时应运行全量测试；新增真实集成还应记录对应资源版本
和验证等级。

## 贡献

欢迎通过 Issue 和 Pull Request 提交问题、插件与实验适配。提交前请：

1. 保持 Agent、Bench、Environment、VLN 和 Memory 的职责边界；
2. 为行为变更添加与风险范围相匹配的测试；
3. 在 YAML `provenance` 中记录数据、模型、上游源码和模拟器版本；
4. 不提交数据集、checkpoint、模拟器缓存或本地运行产物；
5. 不把 contract/smoke 结果描述为完整 benchmark parity。

## 许可证

仓库当前尚未包含许可证文件。首次公开发布前，维护者需要补充根目录 `LICENSE` 并同步本节。
