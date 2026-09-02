# Fabric Architecture

Fabric separates the control plane from one navigation episode's execution plane.

```text
Runner
  -> BenchmarkController[]       enumerate episodes and aggregate metrics
       -> worker process[]       stable CPU/GPU execution slots
            -> Domain            one independent lifetime per episode
                 -> env          fixed environment slot
                 -> metric       fixed evaluator slot
                 -> module[]     arbitrary config-selected modules
                 -> register     shared functions and variable references
                 -> workspace/   episode-local files and command execution
```

## Configuration graph

`runner.yaml` references one `domain.yaml` and multiple `bench.yaml` files. The
Domain file selects arbitrary modules and workspace limits. Each Bench file owns
its dataset adapter and references exactly one Env and one Metric. Therefore a
new agent, VLN model, memory, planner, or fixed workflow changes only the Domain
composition; a new simulator or benchmark does not change Runner or Domain.

## Runtime boundary

Runner enumerates complete episode jobs, assigns each job to a stable worker
process, and collects `DomainResult`. A worker's GPU is selected before any
configured module is imported. Runner never calls observe or step.

Within a Domain, Env starts first because every other module depends on a live
environment. Env, Metric, and all configured modules then run concurrently in
dedicated threads. They share one thread-safe `DomainRegister`. Modules register
dotted function names and live references during `mount()`, then call them
directly with `call()`/`acall()` or `read()`/`write()`. The register also emits
OpenAI Responses-compatible function schemas and records every access in
`calls.jsonl`. `openai_toolset()` maps internal dotted names such as `env.step`
to provider-safe `env__step` names and returns the reverse map for native
`function_call` objects; no text JSON action protocol is introduced.
`Module.expose()` keeps the public call synchronous while dispatching the handler
onto the owning module thread, so simulator and GPU runtime thread affinity is
preserved without exposing queues or an event protocol to callers.

Env must expose `env.step` and owns the canonical, idempotent `env.stop`.
Completion, timeout, native termination, and module failure all converge on the
same terminal state. Metric evaluates the frozen Env result. No Agent, VLN, or
memory role is hard-coded in the runtime.

Multi-goal environments may additionally expose `env.goal.finish`. It advances a
GOAT subgoal inside the same Domain; only `env.stop` or native episode termination
ends that Domain.

## Episode output

```text
runs/<run>/
  config/resolved.json
  result.json
  benches/<bench>/
    config.json
    result.json
    episodes/<episode-domain>/
      episode.json
      domain_config.json
      register.json
      calls.jsonl
      result.json
      workspace/modules/<module>/...
```

The workspace rejects path traversal, runs argv without a shell, caps command
time/output, and exposes a configured Python executable. It is a lightweight
task boundary, not an OS security container. Each module writes its own files;
Runner only handles manifests and results.

## Extension rule

Implement a concrete module under a third-level directory such as
`src/modules/my_agent/`, subclass `Module`, register its public surface in
`mount()` with `expose()`, and put its factory plus parameters in YAML. Environments subclass
`EnvironmentModule`; metrics subclass `MetricModule`; benchmarks subclass
`Benchmark`. None require a core edit.

## Implemented adapters

| Layer | Implementations |
|---|---|
| Bench | Dummy, R2R-CE, Habitat ObjectNav, GOAT, RoboTHOR ObjectNav, Isaac VLN/VLNVerse dataset formats |
| Env | Replay, Habitat session, AI2-THOR/RoboTHOR, Isaac vector session |
| Metric | Generic navigation field mapping, GOAT multi-goal aggregation |
| Module | Expert trajectory driver used only for runtime validation |

Habitat owns its native Hydra files under `config/envs/habitat/`. Isaac keeps its
physics-tick loop inside `envs/isaac`; a project-specific runtime factory builds
the VLN-PE or VLNVerse native session without changing the shared adapter.
