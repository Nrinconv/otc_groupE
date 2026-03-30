from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from .encode_events import action_to_class, class_to_action, encode_obs


def load_episodes(path: str):
    """Groups JSONL samples into sequences by seed."""
    grouped = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            key = (sample["env_name"], int(sample["seed"]))
            grouped[key].append(sample)

    episodes = []
    for (env_name, seed), samples in grouped.items():
        samples = sorted(samples, key=lambda x: x["step"])
        obs_seq = [encode_obs(s["obs"]) for s in samples]
        act_seq = [action_to_class(s["target_action"]) for s in samples]
        episodes.append({
            "env_name": env_name,
            "seed": seed,
            "obs_seq": obs_seq,
            "act_seq": act_seq,
        })
    return episodes


def split_episodes_by_seed(episodes, test_size=0.2, random_state=42):
    """Splits full episodes to prevent data leakage between train/test."""
    unique_seeds = sorted(set(ep["seed"] for ep in episodes))
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_seeds)

    n_test = max(1, int(round(len(unique_seeds) * test_size)))
    n_test = min(n_test, len(unique_seeds) - 1)

    test_seeds = set(shuffled[:n_test].tolist())
    train_seeds = set(shuffled[n_test:].tolist())

    train_eps = [ep for ep in episodes if ep["seed"] in train_seeds]
    test_eps = [ep for ep in episodes if ep["seed"] in test_seeds]
    return train_eps, test_eps, sorted(train_seeds), sorted(test_seeds)


class PrefixSequenceDataset(Dataset):
    """Dataset that generates all possible prefixes of an episode for LSTM training."""
    def __init__(self, episodes):
        self.samples = []
        for ep in episodes:
            obs_seq = ep["obs_seq"]
            act_seq = ep["act_seq"]
            for t in range(len(obs_seq)):
                prefix = np.stack(obs_seq[: t + 1], axis=0)
                target = act_seq[t]
                self.samples.append((prefix, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


def collate_fn(batch):
    """Pads variable length sequences for batch processing."""
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    targets = torch.stack(targets)
    return padded, lengths, targets


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


def evaluate(model, loader, device):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=1)
            all_targets.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    return accuracy_score(all_targets, all_preds), f1_score(all_targets, all_preds, average="macro", zero_division=0), all_targets, all_preds


def save_registry_entry(registry_path: Path, env_name: str, entry: dict):
    registry = {}
    if registry_path.exists():
        text = registry_path.read_text(encoding="utf-8").strip()
        if text: registry = json.loads(text)
    registry[env_name] = entry
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def train_model(model, train_loader, test_loader, device, epochs, lr, save_path: Path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_acc = -1.0, -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Loss {total_loss:.4f} | Test Acc {test_acc:.4f} | Test F1 {test_f1:.4f}")

        if (test_f1 > best_f1) or (test_f1 == best_f1 and test_acc > best_acc):
            best_f1, best_acc = test_f1, test_acc
            torch.save(model.state_dict(), save_path)
    return best_acc, best_f1


def main():
    parser = argparse.ArgumentParser(description="Train and save an LSTM baseline")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--save-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--update-registry", action="store_true")
    args = parser.parse_args()

    episodes = load_episodes(args.data)
    train_eps, test_eps, train_seeds, test_seeds = split_episodes_by_seed(episodes)

    train_dataset = PrefixSequenceDataset(train_eps)
    test_dataset = PrefixSequenceDataset(test_eps)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = len(train_dataset[0][0][0])
    device = torch.device("cpu")
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)

    model_dir = Path("models/openthechests")
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / args.save_name

    best_acc, best_f1 = train_model(model, train_loader, test_loader, device, args.epochs, 1e-3, save_path)

    # Reload best and evaluate
    model.load_state_dict(torch.load(save_path, map_location=device))
    final_acc, final_f1, y_true, y_pred = evaluate(model, test_loader, device)

    # Log metrics
    results_dir = Path("results/openthechests")
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"{Path(args.save_name).stem}_metrics.json").open("w") as f:
        json.dump({"env_name": args.env_name, "accuracy": final_acc, "f1": final_f1}, f, indent=2)

    if args.update_registry:
        save_registry_entry(model_dir / "model_registry.json", args.env_name, {
            "type": "lstm", "path": args.save_name, "input_dim": input_dim, "hidden_dim": args.hidden_dim, "num_layers": 1, "num_classes": 8
        })
    print(f"Done. Registry updated: {args.update_registry}")

if __name__ == "__main__":
    main()