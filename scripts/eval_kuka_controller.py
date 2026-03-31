from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kuka import KukaChestController


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the KUKA controller.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "heuristic", "ppo", "sac"])
    parser.add_argument("--model-path", type=str, default="models/kuka/sac_colored_chest.zip")
    parser.add_argument("--reward-type", type=str, default="advanced", choices=["basic", "advanced"])
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/kuka/controller_eval.json")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    controller = KukaChestController(
        mode=args.mode,
        model_path=args.model_path,
        reward_type=args.reward_type,
        max_steps=args.max_steps,
    )

    summary = controller.evaluate(n_episodes=args.n_episodes)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "controller_mode": summary["controller_mode"],
        "success_rate": summary["success_rate"],
        "mean_steps": summary["mean_steps"],
        "mean_final_distance": summary["mean_final_distance"],
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
