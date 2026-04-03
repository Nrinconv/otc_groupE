from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integration import OpenTheChestsKukaInterface


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the full OpenTheChests + KUKA integration pipeline."
    )
    parser.add_argument(
        "--env-names",
        nargs="+",
        default=["OpenTheChests-v0", "OpenTheChests-v1", "OpenTheChests-v2"],
        help="OtC environments to evaluate.",
    )
    parser.add_argument(
        "--kuka-modes",
        nargs="+",
        default=["heuristic", "ppo"],
        help="KUKA controller modes to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Episode seeds to evaluate.",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--predictor-modes",
        nargs="+",
        default=["auto"],
        help="OpenTheChests predictor modes to compare.",
    )
    parser.add_argument(
        "--kuka-reward-type",
        type=str,
        default="advanced",
        choices=["basic", "advanced"],
    )
    parser.add_argument(
        "--invoke-kuka-once-per-target",
        action="store_true",
        help="If set, a detected chest target triggers KUKA only once per episode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/integration/full_pipeline_eval.json",
    )
    return parser


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    detection_counts = [
        sum(step["predicted_target_idx"] is not None for step in run["steps"])
        for run in runs
    ]
    correct_detection_counts = [
        sum(
            step["predicted_target_idx"] is not None and step["decision_is_correct"]
            for step in run["steps"]
        )
        for run in runs
    ]
    kuka_invocation_counts = [
        sum(step["kuka_result"] is not None for step in run["steps"])
        for run in runs
    ]
    successful_kuka_counts = [
        sum(
            step["kuka_result"] is not None and step["kuka_result"]["success"]
            for step in run["steps"]
        )
        for run in runs
    ]
    correct_and_successful_kuka_counts = [
        sum(
            step["kuka_result"] is not None
            and step["decision_is_correct"]
            and step["kuka_result"]["success"]
            for step in run["steps"]
        )
        for run in runs
    ]
    first_detection_steps = [
        min(
            step["step"]
            for step in run["steps"]
            if step["predicted_target_idx"] is not None
        )
        for run in runs
        if any(step["predicted_target_idx"] is not None for step in run["steps"])
    ]
    first_successful_kuka_steps = [
        min(
            step["step"]
            for step in run["steps"]
            if step["kuka_result"] is not None and step["kuka_result"]["success"]
        )
        for run in runs
        if any(step["kuka_result"] is not None and step["kuka_result"]["success"] for step in run["steps"])
    ]
    first_correct_detection_steps = [
        min(
            step["step"]
            for step in run["steps"]
            if step["predicted_target_idx"] is not None and step["decision_is_correct"]
        )
        for run in runs
        if any(
            step["predicted_target_idx"] is not None and step["decision_is_correct"]
            for step in run["steps"]
        )
    ]
    first_correct_successful_kuka_steps = [
        min(
            step["step"]
            for step in run["steps"]
            if step["kuka_result"] is not None
            and step["decision_is_correct"]
            and step["kuka_result"]["success"]
        )
        for run in runs
        if any(
            step["kuka_result"] is not None
            and step["decision_is_correct"]
            and step["kuka_result"]["success"]
            for step in run["steps"]
        )
    ]

    all_kuka_results = [
        step["kuka_result"]
        for run in runs
        for step in run["steps"]
        if step["kuka_result"] is not None
    ]
    all_decision_steps = [
        step
        for run in runs
        for step in run["steps"]
        if step["predicted_target_idx"] is not None
    ]
    correct_decision_steps = [
        step for step in all_decision_steps if step["decision_is_correct"]
    ]
    correct_decision_kuka_results = [
        step["kuka_result"]
        for run in runs
        for step in run["steps"]
        if step["kuka_result"] is not None and step["decision_is_correct"]
    ]
    successful_kuka_results = [
        result for result in all_kuka_results if result["success"]
    ]
    correct_and_successful_kuka_results = [
        result for result in correct_decision_kuka_results if result["success"]
    ]

    return {
        "n_episodes": len(runs),
        "episode_decision_rate": float(statistics.fmean([count > 0 for count in detection_counts])) if runs else 0.0,
        "episode_correct_decision_rate": float(statistics.fmean([count > 0 for count in correct_detection_counts])) if runs else 0.0,
        "episode_kuka_invocation_rate": float(statistics.fmean([count > 0 for count in kuka_invocation_counts])) if runs else 0.0,
        "episode_success_rate": float(statistics.fmean([count > 0 for count in correct_and_successful_kuka_counts])) if runs else 0.0,
        "decision_accuracy": (
            float(statistics.fmean([step["decision_is_correct"] for step in all_decision_steps]))
            if all_decision_steps
            else None
        ),
        "mean_otc_return": mean_or_none([float(run["otc_return"]) for run in runs]),
        "mean_otc_steps": mean_or_none([float(run["total_steps"]) for run in runs]),
        "mean_target_detections": mean_or_none([float(count) for count in detection_counts]),
        "mean_correct_target_detections": mean_or_none([float(count) for count in correct_detection_counts]),
        "mean_kuka_invocations": mean_or_none([float(count) for count in kuka_invocation_counts]),
        "mean_successful_kuka_invocations": mean_or_none([float(count) for count in successful_kuka_counts]),
        "mean_first_detection_step": mean_or_none([float(step) for step in first_detection_steps]),
        "mean_first_correct_detection_step": mean_or_none([float(step) for step in first_correct_detection_steps]),
        "mean_first_successful_kuka_step": mean_or_none([float(step) for step in first_successful_kuka_steps]),
        "mean_first_correct_successful_kuka_step": mean_or_none([float(step) for step in first_correct_successful_kuka_steps]),
        "kuka_success_rate_given_invocation": (
            float(statistics.fmean([result["success"] for result in all_kuka_results]))
            if all_kuka_results
            else None
        ),
        "kuka_success_rate_given_correct_decision": (
            float(statistics.fmean([result["success"] for result in correct_decision_kuka_results]))
            if correct_decision_kuka_results
            else None
        ),
        "mean_kuka_final_distance": mean_or_none(
            [float(result["final_distance"]) for result in all_kuka_results]
        ),
        "mean_kuka_steps": mean_or_none(
            [float(result["steps"]) for result in all_kuka_results]
        ),
        "n_total_kuka_invocations": len(all_kuka_results),
        "n_successful_kuka_invocations": len(successful_kuka_results),
        "n_decisions": len(all_decision_steps),
        "n_correct_decisions": len(correct_decision_steps),
        "n_correct_and_successful_kuka_invocations": len(correct_and_successful_kuka_results),
    }


def summarize_by_env(runs: list[dict[str, Any]]) -> dict[str, Any]:
    env_names = sorted({run["env_name"] for run in runs})
    return {
        env_name: summarize_runs([run for run in runs if run["env_name"] == env_name])
        for env_name in env_names
    }


def print_console_summary(report: dict[str, Any]) -> None:
    print("Full Pipeline Evaluation")
    print(f"Seeds: {report['seeds']}")
    print(f"Max OtC steps: {report['max_steps']}")
    print(
        "KUKA trigger policy: "
        f"invoke_once_per_target={report['invoke_kuka_once_per_target']}"
    )
    print()

    for experiment in report["experiments"]:
        print(
            f"Controller mode: {experiment['kuka_mode']} | "
            f"Predictor mode: {experiment['predictor_mode']}"
        )
        overall = experiment["overall"]
        print(
            "Overall: "
            f"episode_success_rate={overall['episode_success_rate']:.3f}, "
            f"decision_rate={overall['episode_decision_rate']:.3f}, "
            f"decision_accuracy={overall['decision_accuracy'] if overall['decision_accuracy'] is not None else 'n/a'}, "
            f"kuka_success_given_correct_decision="
            f"{overall['kuka_success_rate_given_correct_decision'] if overall['kuka_success_rate_given_correct_decision'] is not None else 'n/a'}"
        )

        for env_name, metrics in experiment["by_env"].items():
            print(
                f"  {env_name}: "
                f"episode_success_rate={metrics['episode_success_rate']:.3f}, "
                f"decision_accuracy={metrics['decision_accuracy'] if metrics['decision_accuracy'] is not None else 'n/a'}, "
                f"mean_detections={metrics['mean_target_detections']}, "
                f"mean_correct_detections={metrics['mean_correct_target_detections']}, "
                f"mean_kuka_invocations={metrics['mean_kuka_invocations']}, "
                f"mean_first_correct_detection_step={metrics['mean_first_correct_detection_step']}"
            )
        print()


def run_experiment(
    *,
    env_names: list[str],
    seeds: list[int],
    max_steps: int,
    predictor_mode: str,
    kuka_mode: str,
    kuka_reward_type: str,
    invoke_kuka_once_per_target: bool,
) -> dict[str, Any]:
    interface = OpenTheChestsKukaInterface(
        predictor_mode=predictor_mode,
        kuka_mode=kuka_mode,
        kuka_reward_type=kuka_reward_type,
    )

    evaluation = interface.evaluate_dual_agent(
        env_names=env_names,
        seeds=seeds,
        max_steps=max_steps,
        invoke_kuka_once_per_target=invoke_kuka_once_per_target,
    )
    runs = evaluation["runs"]

    return {
        "predictor_mode": predictor_mode,
        "kuka_mode": kuka_mode,
        "overall": summarize_runs(runs),
        "by_env": summarize_by_env(runs),
        "raw": evaluation,
    }


def main() -> None:
    args = make_parser().parse_args()

    experiments = [
        run_experiment(
            env_names=args.env_names,
            seeds=args.seeds,
            max_steps=args.max_steps,
            predictor_mode=predictor_mode,
            kuka_mode=kuka_mode,
            kuka_reward_type=args.kuka_reward_type,
            invoke_kuka_once_per_target=args.invoke_kuka_once_per_target,
        )
        for predictor_mode in args.predictor_modes
        for kuka_mode in args.kuka_modes
    ]

    report = {
        "env_names": args.env_names,
        "seeds": args.seeds,
        "max_steps": int(args.max_steps),
        "predictor_modes": args.predictor_modes,
        "kuka_modes": args.kuka_modes,
        "kuka_reward_type": args.kuka_reward_type,
        "invoke_kuka_once_per_target": bool(args.invoke_kuka_once_per_target),
        "experiments": experiments,
    }

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_console_summary(report)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
