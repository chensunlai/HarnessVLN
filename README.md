# HarnessVLN Fabric

A small navigation Harness runtime built around independent episode Domains,
parallel configurable modules, and direct function calls through a shared
register.

Run the complete dummy benchmark:

```bash
scripts/run_dummy.sh
```

Or invoke the CLI directly:

```bash
PYTHONPATH=src python -m cli run --runner config/runners/dummy.yaml
```

See `docs/architecture.md` for boundaries and extension points.
