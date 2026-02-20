#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch

if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[3]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.models.advanced import train_sequence as adv_seq
from src.models.baseline import train_xgb as train_base
from src.models.ultra_transfer import (
    TransferWorkspaceConfig,
    append_stability_if_requested,
    compute_reason_attribution,
    evaluate_transfer_predictions,
    persist_transfer_outputs,
    prepare_transfer_data,
    train_and_infer,
)


def _has_flag(args: list[str], flag: str) -> bool:
    for item in args:
        if item == flag or item.startswith(f"{flag}="):
            return True
    return False


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Ultra-transfer sequence training entrypoint. "
            "Default mode runs the isolated ultra_transfer pipeline."
        )
    )

    parser.add_argument(
        "--mode", choices=["run", "module-check", "forward"], default="run"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument(
        "--data-root", type=Path, default=Path("G:/我的云端硬盘/Fuison/data")
    )
    parser.add_argument("--hdf5-subdir", default="J-TEXT/unified_hdf5")
    parser.add_argument(
        "--dataset-artifact-dir", type=Path, default=Path("artifacts/datasets/jtext_v1")
    )
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/models/ultra_transfer")
    )
    parser.add_argument(
        "--report-root", type=Path, default=Path("reports/ultra_transfer")
    )

    parser.add_argument("--models", default="transformer_small,mamba_lite,gru")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gray-ms", type=float, default=50.0)
    parser.add_argument("--fallback-fls-ms", type=float, default=100.0)
    parser.add_argument("--fallback-dt-ms", type=float, default=1.0)
    parser.add_argument("--reconcile-len-tol", type=int, default=2)

    parser.add_argument("--max-train-shots", type=int, default=0)
    parser.add_argument("--max-val-shots", type=int, default=0)
    parser.add_argument("--max-test-shots", type=int, default=0)

    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--pad-short-shots", action="store_true", default=True)
    parser.add_argument("--short-pad-mode", choices=["edge", "zero"], default="edge")
    parser.add_argument(
        "--strict-method-checks",
        dest="strict_method_checks",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-strict-method-checks",
        dest="strict_method_checks",
        action="store_false",
    )
    parser.add_argument(
        "--augment-dynamics", dest="augment_dynamics", action="store_true", default=True
    )
    parser.add_argument(
        "--no-augment-dynamics", dest="augment_dynamics", action="store_false"
    )
    parser.add_argument("--dynamics-eps", type=float, default=1e-6)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--imbalance-sampler",
        dest="imbalance_sampler",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-imbalance-sampler", dest="imbalance_sampler", action="store_false"
    )

    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "isotonic_cv", "sigmoid"],
        default="isotonic_cv",
    )
    parser.add_argument("--threshold-max-shot-fpr", type=float, default=0.05)
    parser.add_argument("--threshold-num-steps", type=int, default=300)
    parser.add_argument("--threshold-robust-delta", type=float, default=0.03)
    parser.add_argument("--threshold-tpr-tolerance", type=float, default=0.02)
    parser.add_argument("--sustain-ms", type=float, default=3.0)
    parser.add_argument("--reason-top-k", type=int, default=3)

    parser.add_argument("--plot-all-test-shots", action="store_true", default=False)
    parser.add_argument("--plot-shot-limit", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--run-stability", action="store_true")
    parser.add_argument("--stability-n-boot", type=int, default=2000)

    args, passthrough = parser.parse_known_args()
    return args, passthrough


def build_forward_command(
    args: argparse.Namespace, passthrough: list[str]
) -> list[str]:
    repo_root = cast(Path, args.repo_root).resolve()
    target_script = (repo_root / "src/models/advanced/train_sequence.py").resolve()
    if not target_script.exists():
        raise FileNotFoundError(f"Advanced trainer not found: {target_script}")

    forward = list(passthrough)
    if not _has_flag(forward, "--output-root"):
        forward.extend(["--output-root", str(args.output_root)])
    if not _has_flag(forward, "--report-root"):
        forward.extend(["--report-root", str(args.report_root)])

    cmd = [str(args.python), str(target_script), "--repo-root", str(repo_root)]
    cmd.extend(forward)
    return cmd


def build_workspace_config(args: argparse.Namespace) -> TransferWorkspaceConfig:
    return TransferWorkspaceConfig(
        repo_root=cast(Path, args.repo_root),
        data_root=cast(Path, args.data_root),
        hdf5_subdir=str(args.hdf5_subdir),
        dataset_artifact_dir=cast(Path, args.dataset_artifact_dir),
        split_dir=cast(Path, args.split_dir),
        output_root=cast(Path, args.output_root),
        report_root=cast(Path, args.report_root),
        seed=int(args.seed),
        gray_ms=float(args.gray_ms),
        fallback_fls_ms=float(args.fallback_fls_ms),
        fallback_dt_ms=float(args.fallback_dt_ms),
        reconcile_len_tol=int(args.reconcile_len_tol),
        max_train_shots=int(args.max_train_shots),
        max_val_shots=int(args.max_val_shots),
        max_test_shots=int(args.max_test_shots),
        window_size=int(args.window_size),
        stride=int(args.stride),
        eval_stride=int(args.eval_stride),
        pad_short_shots=bool(args.pad_short_shots),
        short_pad_mode=str(args.short_pad_mode),
        strict_method_checks=bool(args.strict_method_checks),
        augment_dynamics=bool(args.augment_dynamics),
        dynamics_eps=float(args.dynamics_eps),
    )


def run_module_check() -> None:
    print(
        {
            "status": "ok",
            "modules": [
                prepare_transfer_data.__name__,
                train_and_infer.__name__,
                compute_reason_attribution.__name__,
                evaluate_transfer_predictions.__name__,
                persist_transfer_outputs.__name__,
                append_stability_if_requested.__name__,
            ],
        }
    )


def run_direct(args: argparse.Namespace) -> None:
    adv_seq.set_seed(int(args.seed))
    device = adv_seq.choose_device(str(args.device))
    cfg = build_workspace_config(args)

    prepared = prepare_transfer_data(cfg)
    repo_root = prepared.repo_root

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise RuntimeError("No models provided. Use --models.")

    summary_rows: list[dict[str, float | int | str]] = []

    for model_name in models:
        run_name = (
            f"utr_{model_name}_ws{int(cfg.window_size)}_st{int(cfg.stride)}_"
            f"e{int(args.epochs)}_s{int(cfg.seed)}"
        )
        output_dir = (repo_root / cfg.output_root / run_name).resolve()
        report_dir = (repo_root / cfg.report_root / run_name).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        train_out = train_and_infer(
            model_name=model_name,
            input_dim=prepared.input_dim,
            dropout=float(args.dropout),
            train_x=prepared.train_x,
            train_y=prepared.train_pack.y,
            val_calib_y=prepared.val_calib_pack.y,
            val_calib_x=prepared.val_calib_x,
            val_thresh_x=prepared.val_thresh_x,
            test_x=prepared.test_x,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            patience=int(args.patience),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            focal_gamma=float(args.focal_gamma),
            max_grad_norm=float(args.max_grad_norm),
            device=device,
            use_balanced_sampler=bool(args.imbalance_sampler),
        )

        contrib = compute_reason_attribution(
            model=train_out.model,
            test_x=prepared.test_x,
            batch_size=int(args.batch_size),
            device=device,
        )

        eval_out = evaluate_transfer_predictions(
            val_calib_y=prepared.val_calib_pack.y,
            val_calib_prob_raw=train_out.val_calib_prob_raw,
            val_thresh_y=prepared.val_thresh_pack.y,
            val_thresh_prob_raw=train_out.val_thresh_prob_raw,
            test_y=prepared.test_pack.y,
            test_prob_raw=train_out.test_prob_raw,
            val_timeline_base=prepared.val_thresh_pack.timeline,
            test_timeline_base=prepared.test_pack.timeline,
            sustain_ms=float(args.sustain_ms),
            max_shot_fpr=float(args.threshold_max_shot_fpr),
            calibration_method=str(args.calibration_method),
            reason_top_k=int(args.reason_top_k),
            features=prepared.features,
            contrib_by_window=contrib,
            threshold_num_steps=int(args.threshold_num_steps),
            threshold_robust_delta=float(args.threshold_robust_delta),
            threshold_tpr_tolerance=float(args.threshold_tpr_tolerance),
        )

        training_config: dict[str, object] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "model_name": model_name,
            "device": str(device),
            "data_root": str(prepared.data_root),
            "hdf5_root": str(prepared.hdf5_root),
            "features": prepared.features,
            "feature_count": int(prepared.input_dim),
            "window": {
                "window_size": int(cfg.window_size),
                "stride": int(cfg.stride),
                "eval_stride": int(cfg.eval_stride),
                "strict_method_checks": bool(cfg.strict_method_checks),
            },
            "feature_engineering": {
                "augment_dynamics": bool(cfg.augment_dynamics),
                "dynamics_eps": float(cfg.dynamics_eps),
                "feature_count_after_augmentation": int(prepared.input_dim),
            },
            "imbalance": {
                "balanced_sampler": bool(args.imbalance_sampler),
                "sampler_applied": bool(train_out.used_balanced_sampler),
                "class_balance": train_out.class_balance,
                "pos_weight_value": float(train_out.pos_weight_value),
                "focal_gamma": float(args.focal_gamma),
            },
            "optimizer": {
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
            },
            "labeling": {
                "gray_ms": float(cfg.gray_ms),
                "fallback_fls_ms": float(cfg.fallback_fls_ms),
                "fallback_dt_ms": float(cfg.fallback_dt_ms),
            },
            "method_audit": {
                "strict_method_checks": bool(cfg.strict_method_checks),
                "model_selection_validation_split": "val_calib",
                "calibration_split": "val_calib",
                "threshold_selection_split": "val_thresh",
                "val_partition_expected": ["val_calib", "val_thresh"],
            },
            "calibration": {
                "method": str(args.calibration_method),
                "calibration_shot_fraction": 0.5,
            },
            "threshold_policy": {
                "objective": "shot_fpr_constrained_stable",
                "max_shot_fpr": float(args.threshold_max_shot_fpr),
                "theta": float(eval_out.theta),
                "sustain_ms": float(args.sustain_ms),
                "selection_diag": eval_out.theta_diag,
            },
            "threshold_stability": {
                "num_steps": int(args.threshold_num_steps),
                "robust_delta": float(args.threshold_robust_delta),
                "tpr_tolerance": float(args.threshold_tpr_tolerance),
            },
            "split_counts": {
                "train_windows": int(prepared.train_x.shape[0]),
                "val_calib_windows": int(prepared.val_calib_x.shape[0]),
                "val_thresh_windows": int(prepared.val_thresh_x.shape[0]),
                "test_windows": int(prepared.test_x.shape[0]),
            },
            "data_loading_meta": {
                "train": prepared.train_meta,
                "val_calib": prepared.val_calib_meta,
                "val_thresh": prepared.val_thresh_meta,
                "test": prepared.test_meta,
                "train_short_shots_for_window": prepared.train_pack.short_shots,
                "val_calib_short_shots_for_window": prepared.val_calib_pack.short_shots,
                "val_thresh_short_shots_for_window": prepared.val_thresh_pack.short_shots,
                "test_short_shots_for_window": prepared.test_pack.short_shots,
            },
            "normalization": {
                "feature_mean": [float(x) for x in prepared.norm_mu],
                "feature_std": [float(x) for x in prepared.norm_std],
            },
            "training_history": train_out.history,
            "best_epoch": int(train_out.best_epoch),
            "best_val_roc_auc_raw": float(train_out.best_val_roc_auc_raw),
        }

        metrics_summary: dict[str, object] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "model_name": model_name,
            "window_size": int(cfg.window_size),
            "stride": int(cfg.stride),
            "eval_stride": int(cfg.eval_stride),
            "val_timepoint_calibrated": eval_out.val_metrics_calibrated,
            "test_timepoint_calibrated": eval_out.test_metrics_calibrated,
            "test_shot_policy": eval_out.shot_metrics_test,
            "threshold_policy": {
                "objective": "shot_fpr_constrained_stable",
                "max_shot_fpr": float(args.threshold_max_shot_fpr),
                "theta": float(eval_out.theta),
                "sustain_ms": float(args.sustain_ms),
                "selection_diag": eval_out.theta_diag,
            },
            "imbalance": {
                "class_balance": train_out.class_balance,
                "sampler_applied": bool(train_out.used_balanced_sampler),
                "pos_weight_value": float(train_out.pos_weight_value),
                "focal_gamma": float(args.focal_gamma),
            },
            "feature_engineering": {
                "augment_dynamics": bool(cfg.augment_dynamics),
                "feature_count_after_augmentation": int(prepared.input_dim),
            },
            "method_audit": {
                "strict_method_checks": bool(cfg.strict_method_checks),
                "model_selection_validation_split": "val_calib",
                "calibration_split": "val_calib",
                "threshold_selection_split": "val_thresh",
            },
            "calibration_delta_val_threshold": eval_out.calibration_delta,
            "reason_summary": {
                "reason_top_k": int(args.reason_top_k),
                "reason_rows": int(len(eval_out.reason_df)),
                "disruptive_shots_test": int(
                    (eval_out.shot_warn_test["shot_label"] == 1).sum()
                ),
            },
        }

        metrics_summary = append_stability_if_requested(
            run_stability=bool(args.run_stability),
            val_timeline=eval_out.val_timeline,
            theta=float(eval_out.theta),
            sustain_ms=float(args.sustain_ms),
            max_shot_fpr=float(args.threshold_max_shot_fpr),
            output_dir=output_dir,
            n_boot=int(args.stability_n_boot),
            seed=int(cfg.seed),
            metrics_summary=metrics_summary,
        )

        plot_meta = persist_transfer_outputs(
            output_dir=output_dir,
            report_dir=report_dir,
            eval_out=eval_out,
            metrics_summary=metrics_summary,
            training_config=training_config,
            sustain_ms=float(args.sustain_ms),
            plot_all_test_shots=bool(args.plot_all_test_shots),
            plot_shot_limit=int(args.plot_shot_limit),
        )

        training_config["plotting"] = {
            "plot_all_test_shots": bool(args.plot_all_test_shots),
            "plot_shot_limit": int(args.plot_shot_limit),
            "test_shot_count": int(plot_meta["test_shot_count"]),
            "generated_timeline_png": int(plot_meta["generated_timeline_png"]),
        }
        metrics_summary["plotting"] = dict(training_config["plotting"])

        (output_dir / "training_config.json").write_text(
            json.dumps(training_config, indent=2), encoding="utf-8"
        )
        (output_dir / "metrics_summary.json").write_text(
            json.dumps(metrics_summary, indent=2), encoding="utf-8"
        )
        adv_seq.write_metrics_markdown(report_dir / "metrics.md", metrics_summary)

        checkpoint_path = output_dir / f"{model_name}_best.pt"
        torch.save(
            {
                "model_state_dict": train_out.model.state_dict(),
                "model_name": model_name,
                "input_dim": int(prepared.input_dim),
                "dropout": float(args.dropout),
                "best_epoch": int(train_out.best_epoch),
                "best_val_roc_auc_raw": float(train_out.best_val_roc_auc_raw),
                "normalization": {
                    "feature_mean": prepared.norm_mu.tolist(),
                    "feature_std": prepared.norm_std.tolist(),
                },
            },
            checkpoint_path,
        )

        summary_rows.append(
            {
                "run_name": run_name,
                "model_name": model_name,
                "window_size": int(cfg.window_size),
                "stride": int(cfg.stride),
                "test_accuracy": float(eval_out.test_metrics_calibrated["accuracy"]),
                "test_roc_auc": float(eval_out.test_metrics_calibrated["roc_auc"]),
                "test_pr_auc": float(eval_out.test_metrics_calibrated["pr_auc"]),
                "test_tpr": float(eval_out.test_metrics_calibrated["tpr"]),
                "test_fpr": float(eval_out.test_metrics_calibrated["fpr"]),
                "test_ece": float(eval_out.test_metrics_calibrated["ece_15_bins"]),
                "shot_accuracy": float(eval_out.shot_metrics_test["shot_accuracy"]),
                "shot_tpr": float(eval_out.shot_metrics_test["shot_tpr"]),
                "shot_fpr": float(eval_out.shot_metrics_test["shot_fpr"]),
                "lead_time_ms_median": float(
                    eval_out.shot_metrics_test["lead_time_ms_median"]
                ),
                "theta": float(eval_out.theta),
            }
        )

    summary_csv = (repo_root / cfg.report_root / "advanced_summary.csv").resolve()
    summary_md = (repo_root / cfg.report_root / "advanced_summary.md").resolve()
    adv_seq.write_advanced_summary(
        summary_rows, summary_csv=summary_csv, summary_md=summary_md
    )

    best = max(
        summary_rows,
        key=lambda r: (
            float(r["shot_accuracy"]),
            float(r["test_roc_auc"]),
            float(r["test_accuracy"]),
        ),
    )

    print(
        json.dumps(
            {
                "best_run": best,
                "summary_csv": train_base.to_repo_rel(summary_csv, repo_root),
                "summary_md": train_base.to_repo_rel(summary_md, repo_root),
                "device": str(device),
            },
            indent=2,
        )
    )


def main() -> None:
    args, passthrough = parse_args()

    if str(args.mode) == "module-check":
        if passthrough:
            raise SystemExit(
                f"module-check mode does not accept extra args: {passthrough}"
            )
        run_module_check()
        return

    if str(args.mode) == "forward":
        cmd = build_forward_command(args, passthrough)
        result = subprocess.run(cmd, check=False)
        raise SystemExit(result.returncode)

    if passthrough:
        raise SystemExit(f"run mode does not accept unknown args: {passthrough}")
    run_direct(args)


if __name__ == "__main__":
    main()
