# Multi-Agent Progress Board

Last Updated: 2026-02-19

## Agent-1 (Researcher / Learner)
Status: queued
Done:
- Initialized.
Next:
- Read `ref/` papers and project markdowns.
- Produce `docs/literature_review.md`, `docs/requirements.md`, `docs/feature_physics_map.md`.
Blockers:
- None.
Artifacts:
- (pending)

## Agent-2 (Data Engineer)
Status: queued
Done:
- Initialized.
Next:
- Detect J-TEXT HDF5 root, shot lists, and advanced-time metadata.
- Implement reproducible dataset build pipeline and split files.
Blockers:
- None.
Artifacts:
- (pending)

## Agent-3 (Modeler / Experimenter)
Status: waiting_dependency
Done:
- Waiting for Agent-1 and Agent-2 outputs.
Next:
- Start baseline training/eval/calibration after dependencies are ready.
Blockers:
- Waiting `docs/requirements.md` and dataset split/artifact outputs.
Artifacts:
- (pending)

## Agent-4 (Reviewer / Maintainer)
Status: queued
Done:
- Initialized.
Next:
- Start continuous reviews for Agent-2/3/5 outputs and log risks.
Blockers:
- None.
Artifacts:
- (pending)

## Agent-5 (Technical Writer)
Status: queued
Done:
- Initialized.
Next:
- Prepare architecture/changelog and keep docs synchronized with decisions.
Blockers:
- None.
Artifacts:
- (pending)
