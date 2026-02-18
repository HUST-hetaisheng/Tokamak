# AGENTS.md

## Default Data Source (Codex)
- Default data root for this repository is `G:\我的云端硬盘\Fuison\data`.
- Unless user explicitly overrides, always use this path as the primary data source.
- If a script requires a relative data path, resolve from this root first.
- Do not switch to any other data source automatically.

## Project Goal And Scope
- The primary goal of this repository is to build an innovative and feasible real-time tokamak plasma disruption prediction system.
- The system should output real-time disruption probability, disruption type classification, and early warning signals.
- The disruption vs non-disruption classification accuracy target is at least 95%.
- The key priority is visualization of the disruption process and reliable real-time disruption probability output.
- The research direction should support cross-device transfer learning in later stages (for example from EAST to J-TEXT datasets).
- `D:\Fuison\analysis\paper_131.txt` is a core prior-knowledge source and industry research reference for this project.
- Project modeling, feature design, and evaluation strategy should be grounded in this paper unless the user explicitly requests a different direction.
- `1.ipynb`, `2.ipynb`, and `3.ipynb` are from a different lithium battery remaining useful life project and are out of scope for tokamak disruption modeling tasks in this repository.
- `output/` is reserved for generated output artifacts.

<!-- # Repository Guidelines

## Project Structure & Module Organization
- `analysis/train_east_realtime_sequence.py` is the main entry point for model training and streaming inference (`train` and `predict` modes).
- `shot_list/EAST/*.json` and `shot_list/TEST/*.json` define disruptive/non-disruptive shot splits and `AdvancedTime` labels.
- `data/EAST/unified_hdf5/<bucket>/<shot>.hdf5` stores source time-series signals and metadata.
- `analysis/outputs/<run_name>/` contains generated artifacts such as `best_model.pt`, `training_history.csv`, `metrics_summary.json`, and `sequence_predictions/`.
- Utility scripts live at the repo root and under `data/` (for example `find_missing_shots.py`, `data/read_hdf5_structure.py`, `debug_encoding.py`).

## Build, Test, and Development Commands
- Create environment: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
- Install dependencies: `pip install numpy pandas h5py scikit-learn torch`
- Smoke training run:
  `python analysis/train_east_realtime_sequence.py train --repo-root . --hdf5-root data/EAST/unified_hdf5 --output-dir analysis/outputs/realtime_gru_smoke --epochs 2 --max-disrupt-tv 50 --max-nondisrupt-tv 50 --max-disrupt-test 20 --max-nondisrupt-test 20`
- Streaming prediction:
  `python analysis/train_east_realtime_sequence.py predict --checkpoint analysis/outputs/realtime_gru_smoke/best_model.pt --matrix analysis/outputs/realtime_gru_run7/demo_matrix_2000x11.csv --output-csv analysis/outputs/realtime_predict.csv`
- Shot-list consistency check: `python find_missing_shots.py`

## Coding Style & Naming Conventions
- Use Python with 4-space indentation, explicit type hints for new functions, and `snake_case` naming.
- Keep module constants in `UPPER_SNAKE_CASE` (for example feature key lists).
- Prefer `pathlib.Path` for filesystem operations.
- Save scripts in UTF-8 and avoid introducing mixed-encoding comments or literals.

## Testing Guidelines
- No formal `pytest` suite exists yet; use reproducible smoke runs as baseline validation.
- A change is minimally validated when training completes and regenerates `metrics_summary.json` and `sequence_predictions/test/_summary.csv`.
- If adding unit tests, place them in `tests/` and use `test_*.py` filenames. -->

## Commit & Pull Request Guidelines
- This workspace snapshot has no accessible `.git` metadata; use Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`).
- PRs should include purpose, exact command(s), key hyperparameters, and before/after metrics.
- Avoid committing large generated artifacts or raw HDF5 data (`analysis/outputs/**`, `data/EAST/unified_hdf5/**`) unless explicitly required.

