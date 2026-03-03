Legacy-derived data staged for c4 migration.

Files:
- `agent_vs_agent_legacy.csv`: baseline-vs-baseline comparison table from old repo.
- `refmoves1k_normalized.csv`: normalized supervised rows imported from `refmoves1k_kaggle.csv`.
- `good_move_semic_normalized.csv`: normalized supervised rows imported from `good_move_semic.csv` (first 1000 rows in this pass).
- `legacy_raw/`: preserved raw notebook-era CSV artifacts not yet normalized into c4 schemas:
  - `good_move_archive.csv`, `good_move_semic.csv`, `refmoves1k_kaggle.csv`
  - `training_set.csv`, `moves.csv`, `trained_vs_trained.csv`, `agent_vs_agent.csv`
  - `legacy_raw/test_data/*.csv` evaluation slices (`begin_*`, `middle_*`, `end_*`)

Source repository:
- `C:\Users\David\Documents\Repositories\kaggle-files\connect4`
