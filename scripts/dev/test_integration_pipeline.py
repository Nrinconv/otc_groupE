from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integration import OpenTheChestsKukaInterface


TARGET_LABELS = {
    0: "red",
    1: "green",
    2: "blue",
}


def format_target(target_idx: int) -> str:
    return f"{target_idx} ({TARGET_LABELS.get(target_idx, 'unknown')})"


def build_summary(result: dict) -> str:
    step_triggers = [step for step in result["steps"] if step["predicted_target_idx"] is not None]
    kuka_invocations = [step for step in result["steps"] if step["kuka_result"] is not None]
    successful_kuka_runs = [step for step in kuka_invocations if step["kuka_result"]["success"]]

    lines = [
        "Dual-Agent Summary",
        f"OtC environment: {result['env_name']}",
        f"Seed: {result['seed']}",
        f"Total OtC steps: {result['total_steps']}",
        f"OtC return: {result['otc_return']:.2f}",
        (
            "OtC final status: "
            f"terminated={result['otc_terminated']}, truncated={result['otc_truncated']}"
        ),
        f"Agent 1 target detections: {len(step_triggers)}",
        f"Agent 2 KUKA invocations: {len(kuka_invocations)}",
        f"Agent 2 successes: {len(successful_kuka_runs)}/{len(kuka_invocations)}",
    ]

    if result["opened_targets"]:
        opened = ", ".join(format_target(target_idx) for target_idx in result["opened_targets"])
        lines.append(f"Opened targets: {opened}")
    else:
        lines.append("Opened targets: none")

    if kuka_invocations:
        lines.append("")
        lines.append("KUKA Invocations")
        for step in kuka_invocations:
            kuka_result = step["kuka_result"]
            lines.append(
                "  "
                f"step={step['step']} target={format_target(step['predicted_target_idx'])} "
                f"success={kuka_result['success']} steps={kuka_result['steps']} "
                f"final_distance={kuka_result['final_distance']:.4f} "
                f"mode={kuka_result['controller_mode']}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dual-agent OtC + KUKA integration demo.")
    parser.add_argument("--env-name", default="OpenTheChests-v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--full-json", action="store_true")
    args = parser.parse_args()

    interface = OpenTheChestsKukaInterface(
        predictor_mode="auto",
        kuka_mode="heuristic",
    )

    result = interface.run_dual_agent_episode(
        env_name=args.env_name,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    result_dict = result.to_dict()

    print(build_summary(result_dict))

    if args.full_json:
        print()
        print("Full JSON")
        print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
