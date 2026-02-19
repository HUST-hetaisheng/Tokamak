# J-TEXT Data Audit (Agent-2 MVP)

## Scope
- Dataset: `J-TEXT` unified HDF5.
- Goal: reproducible audit/clean/split/label build with small artifacts only.
- Builder script: `src/data/build_dataset.py`.
- Run command:
  - `python src/data/build_dataset.py --repo-root . --dataset-name jtext_v1 --split-ratio 8,1,1 --seed 42`

## Discovered Paths
- HDF5 (ordered detection):
  - `D:/Fuison/data/J-TEXT/unified_hdf5` (found, selected)
  - `./data/J-TEXT/unified_hdf5` (same effective location in this workspace)
- Shot lists:
  - `shot_list/J-TEXT/Disruption_J-TEXT.json`
  - `shot_list/J-TEXT/Non-disruption_J-TEXT.json`
  - `shot_list/J-TEXT/AdvancedTime_J-TEXT.json`

## Label/Metadata Audit and Cleaning
- Shot-list raw counts:
  - disruptive: `378`
  - non-disruptive: `1356`
  - cross-list conflicts: `0`
  - duplicates (each list): `0`
- Candidate shots after dedup/conflict handling: `1734`
- Clean shots retained: `1733`
- Excluded shots: `1`
  - reason: `missing_hdf5=1`
- Shot-list vs HDF5 label consistency:
  - compared against `meta/IsDisrupt`
  - mismatches removed by rule (observed count in this run: `0`)

## Feature Coverage Policy (20+ Signals)
- Required feature coverage (inferred from HDF5 `data/*`): `23` signals.
- Policy: **use-all-by-default for downstream training**.
  - This run enforces presence of all inferred required signals per kept shot.
  - Missing-key-feature shots are excluded by rule (observed count in this run: `0`).
- Feature list artifact: `artifacts/datasets/jtext_v1/required_features.json`

## Time Axis and FLS/Gray Labeling
- dt mapping: inferred from HDF5 time metadata (`meta/StartTime`, `meta/DownTime`) with TTD fallback available.
- Time-source usage in this run:
  - `meta_start_down=1733`
- Labeling logic applied:
  - disruptive + advanced_time:
    - `[t_end - advanced_time, t_end] -> 1`
    - preceding `30 ms` gray-zone dropped
    - earlier -> `0`
  - disruptive without advanced_time:
    - `[t_end - 25 ms, t_end] -> 1`
    - preceding `30 ms` gray-zone dropped
    - earlier -> `0`
  - non-disruptive:
    - all `0`
- FLS source counts:
  - `advanced_time=375`
  - `fallback_25ms=3`
  - `non_disruptive=1355`

## Split Build (Shot-Level Stratified, 8/1/1)
- train: `1386` shots (`302` disruptive, `1084` non-disruptive)
- val: `174` shots (`38` disruptive, `136` non-disruptive)
- test: `173` shots (`38` disruptive, `135` non-disruptive)
- Split files:
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`
- Shot-count references (authoritative):
  - `artifacts/datasets/jtext_v1/summary.json`:
    - `splits.train.shots=1386`
    - `splits.val.shots=174`
    - `splits.test.shots=173`
  - Split-id files (one shot id per line; line counts should match):
    - `splits/train.txt`
    - `splits/val.txt`
    - `splits/test.txt`

## Class Imbalance Handling
- Considered:
  - `weighted_random_sampler`
  - `focal_loss_or_hard_negative_mining`
- Implemented:
  - `class_weighting` (shot-level + point-level `bce_pos_weight`)
- Artifact:
  - `artifacts/datasets/jtext_v1/class_weights.json`

## Dataset Statistics
- raw points total: `592175`
- used points total (after gray drop): `580954`
- positive points: `7654`
- negative points: `573300`
- gray dropped points: `11221`

## 3-Shot y(t) Examples
- Note: only 3 `y(t)` plots are generated here on purpose, as representative audit samples
  (one per labeling scenario) to keep this MVP audit lightweight and readable.
- Full test-shot timelines are saved separately by downstream evaluation:
  - `reports/plots/probability_timelines_test.csv` (all test-shot timeline rows)
  - `reports/plots/probability/` (rendered timeline PNG subset)
- disruptive with advanced_time:
  - shot `1051501`
  - plot: `reports/plots/data_audit/shot_1051501_disrupt_with_advanced.png`
- disruptive without advanced_time (fallback 25ms):
  - shot `1053639`
  - plot: `reports/plots/data_audit/shot_1053639_disrupt_without_advanced.png`
- non-disruptive:
  - shot `1051500`
  - plot: `reports/plots/data_audit/shot_1051500_non_disruptive.png`

## Output Artifacts (Small)
- Core stats and summary:
  - `artifacts/datasets/jtext_v1/summary.json`
  - `artifacts/datasets/jtext_v1/stats.json`
- Small tabular outputs:
  - `artifacts/datasets/jtext_v1/clean_shots.csv`
  - `artifacts/datasets/jtext_v1/excluded_shots.csv`
  - `artifacts/datasets/jtext_v1/example_shots.csv`
  - `artifacts/datasets/jtext_v1/label_examples.csv`
- Small sample binary:
  - `artifacts/datasets/jtext_v1/small_sample.npz`
- Split ids:
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`
