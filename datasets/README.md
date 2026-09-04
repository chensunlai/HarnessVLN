# Dataset construction

This directory contains offline dataset inspection, synthesis, and validation
code. It is intentionally independent of the Harness runtime: builders read
source datasets and write ordinary files, without importing `src/`, Domain,
Bench, Runner, or any Agent implementation.

Each dataset family owns a third-level directory. Run a builder from the
repository root after activating the required Python environment, for example:

```bash
python -m datasets.agent_vln.build --help
```

Generated datasets belong under `data/` and are local artifacts; source code
and construction specifications remain in this directory.
