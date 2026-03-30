import argparse
import json
from pathlib import Path


RULES = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1],
}


def predict_action(obs):
    e_type = obs["e_type"]
    return RULES.get(e_type, [0, 0, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/openthechests/processed/oracle_dataset.jsonl"
    )
    args = parser.parse_args()

    path = Path(args.data)

    total = 0
    correct = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            pred = predict_action(sample["obs"])
            target = sample["target_action"]

            total += 1
            if pred == target:
                correct += 1

    acc = correct / total if total > 0 else 0.0

    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()