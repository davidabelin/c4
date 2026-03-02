# Notebook Archaeology Notes

Source scanned: `C:\Users\David\Documents\Repositories\kaggle-files\connect4`

## Summary
- Notebook count: 58
- Dominant reusable themes:
  - Alpha-beta/minimax heuristic agents (`drop_piece`, `count_windows`, `alphabeta`)
  - Legacy move-quality datasets (`refmoves1k_kaggle.csv`, `good_move_archive.csv`, `good_move_semic.csv`)
  - Tabular Q-learning prototype (`connectx-with-q-learning.ipynb`)
- Dominant non-portable themes:
  - TensorFlow 1.15 + `stable_baselines` notebook workflows
  - Colab/Kaggle shell commands and environment assumptions

## What was ported in this pass
- Heuristic agents:
  - `alpha_beta_v9`
  - `adaptive_midrange`
  - `time_boxed_pruner`
- Dataset conversion:
  - JSONL-style and semicolon-style legacy record parsing
  - Normalized supervised CSV generation with `b00..b41`, `label`, and move-score metadata
- RL:
  - Notebook-inspired tabular Q-learning trainer with sparse Q-table and epsilon decay

## What was not ported yet
- PPO/A2C model wrappers and TF1 custom CNN policies
- Notebook visualization and ad-hoc benchmark chart cells
- Legacy model zip checkpoints referenced only by absolute Colab paths

## Immediate migration recommendations
1. Replace in-agent nested helper duplication with shared board-eval utilities.
2. Add reproducible benchmark CLI (like `rps/scripts/benchmark_agents.py`) for c4 agents.
3. Build `c4_storage.repository` schema for games/rounds/models/jobs to match rps workflow.
4. Implement `c4_web` blueprints and pages for `play`, `training`, `rl`.
