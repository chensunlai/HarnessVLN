# Train-free Agentic Navigation 设计

## 定位

本方案参考 [Qwen-RobotNav](https://arxiv.org/abs/2606.18112) 的两层结构：上层 Agent 理解完整任务、拆分子任务并选择能力；底层模块只完成一个边界明确的局部子任务，并返回结果与证据。

```text
上层 Agent
  -> 调用一个底层模块（局部任务）
      -> train-free 子 Agent 主动 observe / act
      <- completed / refused / failed + evidence
  -> 根据结果继续规划，最终由上层决定 stop
```

四个模块并行驻留，由上层一次选择一个运动能力。底层结束的是**子任务**，整个 episode 仍由上层收束。

## 与 Qwen-RobotNav 的异同

两者都由上层规划器选择收窄后的导航能力，并在一次完整任务中动态切换能力。Qwen-RobotNav 用一个训练后的统一模型，以 `VLN`、`PointNav`、`ObjNav`、`Tracking` 模式预测 waypoint；本方案用多个独立的 train-free 子 Agent，每个模块内部运行有界 Agent Loop，主动观察和执行，并验证自己的执行前提。

因此，本方案底层返回的是一次局部执行的状态与证据，而不是单次 waypoint 预测。`refused` 表示任务超出模块边界或前提失效，供上层重新规划；`failed` 只表示执行或系统错误。

## 模型设计

四个模块不应长期复制四套 Agent Loop，但基础版本也不提前引入公共内核。当前只有 `agent_vln` 实现真实循环，它在模块内部直接使用 Responses 原生 function calling；等第二个运动模块出现相同代码后，再把稳定的请求、重试和轨迹记录部分抽成极小公共内核。每个能力模块始终保留自己的 prompt、输入整理、工具、上下文策略和结果校验。

```text
UpperAgent（完整任务上下文）
  -> 四个原生 function tools
      -> 能力模块（独立 prompt / context / tools / guards）
          -> 模块内 Agent Loop -> Model
          -> Env 或局部导航工具
      <- 统一的局部结果与证据
```

每次底层调用都创建隔离的短上下文，只接收收窄后的任务、当前观察和明确引用的证据，不继承上层完整对话；结束后仅把压缩结果返回上层。上层只看到四个能力函数，底层模型只看到本模块获准的工具，从模型上下文上落实职责边界。

不直接照搬 Qwen-RobotNav 的 `B`、`gamma` 等调用参数。Qwen 的模型针对这些控制量训练过，本方案使用通用模型，观察策略应由各模块的 context builder 固定并通过配置调整：VLN 保留稀疏路线历史，目标接近偏重最近帧，局部导航使用占据图，环境描述使用当前多视图。

## 底层模块

| 模块 | 模型输入与上下文 | 模型职责与硬边界 |
|---|---|---|
| `agent_vln` | 局部路线指令、当前 RGB-D、调用内稀疏历史和路线进度 | 逐段视觉验证并选择短动作序列。下一段与场景不对应时返回 `refused`；不改写指令或开放式搜索。 |
| `agent_objnav` | 固定的目标证据、当前 RGB-D 和少量最近帧 | 持续重识别并接近当前可见目标。失去视觉依据后先原地复核一次，仍未锁定则返回 `refused`；不搜索初始不可见目标。它本质是“可见目标接近”，不是传统搜索型 ObjNav。 |
| `agent_local` | 带尺度与坐标系的局部占据图、当前位姿、目标位姿和容差 | 模型负责调用、检查和必要时重规划；路径搜索与跟踪交给确定性 local planner，避免让 VLM 直接计算精确避障动作。“沙发正对面”需先解析成局部目标位姿。 |
| `agent_desc` | 当前单帧或多视图观察，以及上层给出的关注目标 | 生成带证据引用的结构化局部描述。通常一次模型调用即可；不移动、不制定完整任务，也不把未观察区域写成事实。 |

四个模块采用各自收窄的输入，但返回统一结果：`status`、`reason`、`evidence`，以及 Env 可提供时的 `final_pose`。`completed` 和 `refused` 必须经过模块自己的 guard 校验，不能只相信模型文本判断。

## 当前 `agent_vln` 基线

`agent_vln.run(instruction)` 自己调用 `env.observe` 获取 RGB-D 和位姿，模型每轮必须通过 Responses 原生工具调用选择 `agent_vln_act` 或 `agent_vln_finish`。前者一次接受 1-4 个 `forward / turn_left / turn_right` 原子动作，并逐个调用 `env.step`；后者只结束局部任务，不调用 `env.stop`。`master_agent` 收到结果后才决定 episode 状态。

图像上下文由均匀采样的长期轨迹帧、连续近期帧和当前帧组成；模型在每次动作调用中同步维护已完成指令片段、当前位置和下一步，作为长期文本总结。模块记录 `trace.jsonl`、`history.json` 和 `result.json`，但不把图片的 base64 写入轨迹。到达需要两次无移动确认，动作预算耗尽时只能完成或拒绝；同一位置转满一圈仍没有可执行方向则直接拒绝，避免继续空转。

最小真实调试入口是：

```bash
python -m cli run --runner config/runners/agent_vln_debug.yaml
```

该 Runner 使用 Habitat 与 `data/generated/agent_vln_r2r_local_v2`，模型、推理强度、图像记忆和调用上限均在 `config/modules/agent_vln.yaml` 中配置。API 地址与密钥仅由启动进程环境提供。
