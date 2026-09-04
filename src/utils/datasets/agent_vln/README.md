# AgentVLN R2R Local Dataset

## Purpose

This dataset is the first debugging set for `agent_vln`: execute a bounded,
language-guided route whose progress can be checked against the nearby scene.
It is not a replacement for full R2R evaluation and it does not train the
agent.

## Construction decision

R2R-CE provides a route and three human instructions for almost every
trajectory, but it does not align instruction clauses to intermediate
waypoints. Therefore v1 **does not cut a long route into synthetic sub-routes**.
Doing so would require guessing both the internal start heading and which words
describe each segment. Instead, the builder selects complete short routes from
the R2R-CE `train` split, preserving the annotated start pose, goal, reference
path, and human language.

A route is a candidate when it satisfies all of these bounds:

- geodesic distance is at most 5.5 m;
- reference path contains 4 to 6 viewpoints;
- vertical range is at most 0.35 m, excluding floor changes;
- there are at most two turns of 30 degrees or more;
- no individual turn exceeds 135 degrees;
- at least two distinct human instruction variants exist;
- the primary instruction is at most 40 words;
- with `--validate-navmesh`, all route points are navigable and a finite path
  exists from start to goal on the matching MP3D navmesh.

These are reproducible geometric proxies for a local task. They do not prove
that every mentioned object is visible in every frame. The v2 construction
below samples route views and records a visual grounding result for every
candidate.

## Split and diversity

The requested 100 routes are split `debug/dev/test = 60/20/20`. Scene IDs, not
episodes, are assigned to splits, so neither a route nor visual appearance from
the same Matterport scan leaks across them. Only R2R-CE `train` is consumed;
official `val_seen` and `val_unseen` remain untouched for downstream benchmark
checks.

Selection balances three geometric patterns (`low_turn`, `one_turn`, and
`two_turn`) and three language lengths (`concise`, `standard`, and `detailed`).
All source human variants are retained in `manifest.json`; one fixed primary
variant is written to the Habitat-compatible split file. Variants from the same
route are never counted as separate examples or placed in different splits.
Semantic tags such as doorway transitions, room references, relative landmark
phrases, and left/right turns are descriptive diagnostics, not filtering labels.
The v1 files retain only source annotations; generated text is isolated in v2.

The split files use a 1 m goal radius rather than R2R's original 3 m radius.
This prevents an agent from succeeding after traversing only a small part of a
local route. Evaluator configuration must use the same 1 m threshold.

Refusal examples are deliberately excluded from these 100 executable routes.
Instruction/scene swaps should later be generated as a separate paired
robustness set; mixing them into navigation success metrics would make SR and
SPL ambiguous.

## Build

After activating the project Python environment:

```bash
PYTHONPATH=src python -m utils.datasets.agent_vln.build \
  --source data/datasets/r2r/train/train.json \
  --scenes-root data/scene_datasets \
  --output data/generated/agent_vln_r2r_local_v1 \
  --count 100 \
  --validate-navmesh
```

Outputs:

```text
agent_vln_r2r_local_v1/
├── debug/debug.json.gz   # 60 Habitat-compatible episodes
├── dev/dev.json.gz       # 20 Habitat-compatible episodes
├── test/test.json.gz     # 20 Habitat-compatible episodes
├── manifest.json         # provenance, all language variants, geometry
└── summary.json          # counts, distributions, and leakage checks
```

Re-running with the same input, seed, and arguments is deterministic.

## Route images and three instruction styles

The v2 pipeline samples up to six positions uniformly along each complete
reference path. The start frame preserves the episode's native heading,
intermediate frames face the outgoing segment, and the goal frame preserves the
arrival direction. Each route stores individual JPEGs, their poses and hashes,
plus a contact sheet for inspection.

```bash
PYTHONPATH=src python -m utils.datasets.agent_vln.render_paths \
  --input data/generated/agent_vln_r2r_local_v1 \
  --output data/generated/agent_vln_r2r_local_v2 \
  --scenes-root data/scene_datasets \
  --gpu-device-id 0
```

The rewrite stage sends the ordered images and all source human annotations to
the Responses API. It requests `gpt-5.6-terra` with high reasoning and returns
exactly one `concise`, one `natural`, and one `landmark_rich` instruction per
route. A strict JSON schema and local validation enforce style count, word
bounds, uniqueness, and the absence of explicit image or dataset references.
Set the standard `OPENAI_API_KEY` and, when needed, `OPENAI_BASE_URL` environment
variables before running it; credentials are neither read from local config
files nor written to the dataset.

```bash
PYTHONPATH=src python -m utils.datasets.agent_vln.rewrite_instructions \
  --input data/generated/agent_vln_r2r_local_v1 \
  --output data/generated/agent_vln_r2r_local_v2 \
  --model gpt-5.6-terra \
  --reasoning-effort high
```

One route therefore becomes three independently evaluable Habitat episodes
with the same pose, path, and goal. Raw per-route generation records make API
runs resumable, while `manifest.json` contains the final instructions and their
grounding status.

## Conflict replacement

Generator review can mark a source route as `conflict` when the three original
annotations, sampled views, and reference path materially disagree. To keep a
final set of 100 executable routes without silently retaining those cases,
build a balanced reserve pool, render its missing routes, and materialize only
non-conflicting candidates:

```bash
PYTHONPATH=src python -m utils.datasets.agent_vln.extend_candidates \
  --base data/generated/agent_vln_r2r_local_v1 \
  --source data/datasets/r2r/train/train.json \
  --scenes-root data/scene_datasets \
  --output data/generated/agent_vln_r2r_local_candidates_v1

PYTHONPATH=src python -m utils.datasets.agent_vln.render_paths \
  --input data/generated/agent_vln_r2r_local_candidates_v1 \
  --output data/generated/agent_vln_r2r_local_v2 \
  --scenes-root data/scene_datasets \
  --gpu-device-id 0

PYTHONPATH=src python -m utils.datasets.agent_vln.rewrite_instructions \
  --input data/generated/agent_vln_r2r_local_candidates_v1 \
  --output data/generated/agent_vln_r2r_local_v2 \
  --model gpt-5.6-terra \
  --reasoning-effort high \
  --final-routes 100
```

The final split remains `60/20/20`, preserves the original scene-disjoint
assignment and route-pattern balance, and contains 300 episodes. `curation.json`
records every excluded conflict and unused reserve. Re-running generation with
unchanged images, prompt version, model, and effort reuses validated cached
responses.
