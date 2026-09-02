# Design Evidence

Fabric keeps the useful part of the `Nexus` branch: one complete environment
lifetime per Domain, an agent-independent control plane, stable worker GPU
assignment, and component-owned output. It removes Nexus's fixed Agent/VLN/Memory
roles and capability whitelist because Fabric's Domain is meant to evaluate
arbitrary harness compositions.

The `main` branch supplied concrete evidence for dataset fields, native action
maps, Habitat configuration isolation, Isaac multi-tick actions, and output
layout. Fabric reuses those backend facts behind new generic Module and Register
contracts rather than copying the old Runner-driven stack construction.

External designs informed four decisions:

1. DeepSeek Harness composes even the loop and tool registry from configuration.
   Fabric applies that principle to Domain modules, but deliberately omits its
   event waterfalls, hot unload, UI, and persistence services.
2. AllenAct separates task sampling from environments. Fabric therefore lets a
   BenchmarkController enumerate episodes while its Bench config selects Env and
   Metric; Runner does not parse datasets.
3. Habitat defines datasets as episode streams plus scene assets. A navigation
   episode is consequently the smallest default Domain boundary.
4. GOAT places 5-10 sequential goals in one episode, while VLNVerse actions span
   multiple simulator ticks. These require `env.goal.finish` and an Isaac-local
   tick loop rather than changes to Runner or Domain.

Sources:

- <https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md>
- <https://allenact.org/getting_started/abstractions/>
- <https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md>
- <https://github.com/Ram81/goat-bench>
- <https://github.com/sihaoevery/vlnverse_emr/tree/vlnverse>
- <https://github.com/jacobkrantz/VLN-CE>
