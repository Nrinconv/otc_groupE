import json
from collections import defaultdict
from pathlib import Path

from src.openthechests.predict import OpenTheChestsPredictor


def load_histories(path):
    grouped = defaultdict(list)

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            key = (sample["env_name"], int(sample["seed"]))
            grouped[key].append(sample)

    histories = []
    for (env_name, seed), samples in grouped.items():
        samples = sorted(samples, key=lambda x: x["step"])

        history = []
        for s in samples:
            history.append(s["obs"])
            histories.append({
                "env_name": env_name,
                "seed": seed,
                "step": s["step"],
                "history": history.copy(),
                "target_action": s["target_action"],
            })

    return histories


def main():
    predictor = OpenTheChestsPredictor(
        model_dir="models/openthechests",
        mode="auto",
    )

    datasets = [
        "data/openthechests/processed/oracle_dataset_v0_large.jsonl",
        "data/openthechests/processed/oracle_dataset_v1_large.jsonl",
        "data/openthechests/processed/oracle_dataset_v2_large.jsonl",
    ]

    for dataset_path in datasets:
        print("\n" + "=" * 60)
        print(f"Testing predictor on: {dataset_path}")
        print("=" * 60)

        samples = load_histories(dataset_path)

        # test only first 5 samples to keep output readable
        for s in samples[:5]:
            pred_action = predictor.predict_action(
                history=s["history"],
                env_name=s["env_name"],
            )
            pred_target = predictor.predict_target(
                history=s["history"],
                env_name=s["env_name"],
            )

            print(f"env={s['env_name']} seed={s['seed']} step={s['step']}")
            print(f"target_action={s['target_action']}")
            print(f"pred_action={pred_action}")
            print(f"pred_target={pred_target}")
            print("-" * 40)


if __name__ == "__main__":
    main()
