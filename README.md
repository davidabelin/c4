## c4
Connect4 lab scaffold for AIX, intentionally shaped to mirror the `rps` project layout.

### Scope of this initial migration pass
- Establish a `rps`-like package/file organization under the sibling `c4` repo.
- Port reusable legacy heuristic agents from the old `connect4` repo into clean Python modules.
- Add a robust dataset importer for historical CSV/JSONL files from notebook workflows.
- Add a first tabular RL trainer based on the historical `connectx-with-q-learning` notebook.
- Add `rps`-style backend architecture:
  - repository-backed gameplay API
  - supervised training jobs + model registry
  - RL training jobs + model artifact registration

### What is intentionally deferred
- Full gameplay UI/animation implementation parity with `rps_web`.
- Benchmark/league tooling parity with `rps_benchmarks`.
- Deployment manifests (`app.yaml`, `.gcloudignore`) for App Engine.

### Layout (mirrors `rps` package shape)
- `c4_agents/`
- `c4_core/`
- `c4_training/`
- `c4_rl/`
- `c4_storage/`
- `c4_web/`
- `scripts/`
- `tests/`
- `docs/`
- `data/`
