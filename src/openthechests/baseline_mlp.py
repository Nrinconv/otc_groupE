from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Using the centralized logic we just created
from .encode_events import action_to_class, class_to_action, encode_obs


def load_jsonl_dataset(path: str):
    """Loads dataset and prepares features, labels, and seed tracking."""
    X = []
    y = []
    seeds = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            X.append(encode_obs(sample["obs"]))
            y.append(action_to_class(sample["target_action"]))
            seeds.append(int(sample["seed"]))

    return (
        np.vstack(X),
        np.array(y, dtype=np.int64),
        np.array(seeds, dtype=np.int64),
    )


def split_by_seed(X, y, seeds, test_size=0.2, random_state=42):
    """Ensures test set contains entirely unseen seeds."""
    unique_seeds = np.unique(seeds)

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_seeds)

    n_test_seeds = max(1, int(round(len(unique_seeds) * test_size)))
    n_test_seeds = min(n_test_seeds, len(unique_seeds) - 1)

    test_seeds = shuffled[:n_test_seeds]
    train_seeds = shuffled[n_test_seeds:]

    train_mask = np.isin(seeds, train_seeds)
    test_mask = np.isin(seeds, test_seeds)

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask], train_seeds, test_seeds


def print_class_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    print(title)
    for cls, count in zip(unique, counts):
        print(f"  class {cls} -> action {class_to_action(cls)}: {count}")
    print()


def save_registry_entry(registry_path: Path, env_name: str, entry: dict):
    registry = {}

    if registry_path.exists():
        text = registry_path.read_text(encoding="utf-8").strip()
        if text:
            registry = json.loads(text)

    registry[env_name] = entry

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train and save an MLP baseline with seed-level split")
    parser.add_argument("--data", type=str, required=True, help="Path to the .jsonl dataset")
    parser.add_argument("--env-name", type=str, required=True, help="Environment name (e.g., OpenTheChests-v0)")
    parser.add_argument("--save-name", type=str, required=True, help="Output filename (e.g., mlp_v0.pkl)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Data Loading
    X, y, seeds = load_jsonl_dataset(args.data)

    # Seed-based Split
    X_train, X_test, y_train, y_test, train_seeds, test_seeds = split_by_seed(
        X, y, seeds, test_size=args.test_size, random_state=args.random_state,
    )

    # Pipeline Definition
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=args.random_state,
        )),
    ])

    print(f"Training MLP for: {args.env_name}")
    print("-" * 40)
    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)} (Seeds: {sorted(train_seeds.tolist())})")
    print(f"Test samples:  {len(X_test)}  (Seeds: {sorted(test_seeds.tolist())})")
    print("-" * 40)

    # Training
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("-" * 40)
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print("-" * 40)

    # Saving Model
    model_dir = Path("models/openthechests")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / args.save_name

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    # Updating Registry
    registry_path = model_dir / "model_registry.json"
    save_registry_entry(
        registry_path=registry_path,
        env_name=args.env_name,
        entry={"type": "mlp", "path": args.save_name},
    )

    # Logging Metrics
    results_dir = Path("results/openthechests")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{Path(args.save_name).stem}_metrics.json"

    metrics = {
        "env_name": args.env_name,
        "model_type": "mlp",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "train_samples": len(X_train),
        "test_seeds": sorted(test_seeds.tolist()),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Model saved to {model_path} and registry updated.")


if __name__ == "__main__":
    main()