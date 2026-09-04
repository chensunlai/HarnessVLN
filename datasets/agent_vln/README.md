# AgentVLN R2R Local v1

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
that every mentioned object is visible in every frame. The manifest records
`visual_validation: pending`; route panoramas or VLM/human review can promote
that field in a later curation pass without changing how examples are split.

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
No model-generated paraphrases are added in v1, avoiding semantic drift.

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
python -m datasets.agent_vln.build \
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
