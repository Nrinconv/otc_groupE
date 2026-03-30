from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .encode_events import (
    action_to_target_idx,
    class_to_action,
    encode_history,
    encode_obs,
)


class LSTMInferenceModel(nn.Module):
    """
    Standard LSTM architecture for inference. 
    Matches the training architecture in baseline_lstm.py.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, num_classes: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits


class OpenTheChestsPredictor:
    """
    Unified Inference API for OpenTheChests.
    Supports Rule-based, MLP, and LSTM modes via a model registry.
    """

    def __init__(
        self,
        model_dir: str = "models/openthechests",
        mode: str = "auto",
        registry_path: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mode = mode
        self.registry_path = Path(registry_path) if registry_path else self.model_dir / "model_registry.json"

        self._registry = self._load_registry()
        self._loaded_models: dict[str, Any] = {}
        self._device = torch.device("cpu")

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            # Return empty registry; user might be using mode="rule"
            return {}

        with self.registry_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _rule_action(self, last_obs: dict[str, Any]) -> list[int]:
        """Fallback hardcoded logic for basic chest opening."""
        e_type = int(last_obs["e_type"])
        if e_type == 0: return [1, 0, 0]
        if e_type == 1: return [0, 1, 0]
        if e_type == 2: return [0, 0, 1]
        return [0, 0, 0]

    def _resolve_env_name(self, env_name: str) -> str:
        valid_envs = {"OpenTheChests-v0", "OpenTheChests-v1", "OpenTheChests-v2"}
        if env_name not in valid_envs:
            raise ValueError(f"Unsupported env_name: {env_name}")
        return env_name

    def _resolve_model_spec(self, env_name: str) -> dict[str, Any]:
        env_name = self._resolve_env_name(env_name)

        if self.mode == "rule":
            return {"type": "rule"}

        spec = self._registry.get(env_name)
        if spec is None:
            if self.mode == "auto":
                raise ValueError(f"No model registered for {env_name} in registry.")
            return {"type": self.mode} # Force mode if specified but missing in registry
        
        return spec

    def _load_model_once(self, spec: dict[str, Any]) -> Any:
        model_type = spec["type"]
        if model_type == "rule": return None

        model_path = self.model_dir / spec["path"]
        cache_key = str(model_path)

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        if model_type == "mlp":
            with model_path.open("rb") as f:
                model = pickle.load(f)
            self._loaded_models[cache_key] = model
            return model

        if model_type == "lstm":
            model = LSTMInferenceModel(
                input_dim=spec.get("input_dim", 40),
                hidden_dim=spec.get("hidden_dim", 64),
                num_layers=spec.get("num_layers", 1),
                num_classes=spec.get("num_classes", 8),
            )
            state_dict = torch.load(model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.to(self._device).eval()
            self._loaded_models[cache_key] = model
            return model

        raise ValueError(f"Unsupported model type: {model_type}")

    def predict_action(self, history: list[dict[str, Any]], env_name: str) -> list[int]:
        """Returns the 3-bit binary action [R, G, B]."""
        if not history:
            raise ValueError("History cannot be empty.")

        spec = self._resolve_model_spec(env_name)
        model_type = spec["type"]

        if model_type == "rule":
            return self._rule_action(history[-1])

        model = self._load_model_once(spec)

        if model_type == "mlp":
            x = encode_obs(history[-1]).reshape(1, -1)
            # sklearn models usually return a numpy array of predictions
            pred_class = int(model.predict(x)[0])
            return class_to_action(pred_class)

        if model_type == "lstm":
            seq = encode_history(history)
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)
            lengths = torch.tensor([seq.shape[0]], dtype=torch.long).to(self._device)

            with torch.no_grad():
                logits = model(x, lengths)
                pred_class = int(torch.argmax(logits, dim=1).item())
            return class_to_action(pred_class)

        raise ValueError(f"Model type {model_type} not supported for prediction.")

    def predict_target(self, history: list[dict[str, Any]], env_name: str) -> int | None:
        """Returns the target chest index (0: Red, 1: Green, 2: Blue) or None."""
        action = self.predict_action(history, env_name)
        return action_to_target_idx(action)