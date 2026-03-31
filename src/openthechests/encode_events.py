from __future__ import annotations

from typing import Any
import numpy as np

# Constants for One-Hot encoding
NUM_EVENT_TYPES = 15
NUM_COLORS = 8

def action_to_class(action: list[int] | tuple[int, int, int]) -> int:
    """Converts a 3-bit action [0, 1, 0] to an integer class index (0-7)."""
    return int(action[0]) * 4 + int(action[1]) * 2 + int(action[2])

def class_to_action(cls: int) -> list[int]:
    """Converts an integer class (0-7) back to a 3-bit binary action list."""
    return [(cls >> 2) & 1, (cls >> 1) & 1, cls & 1]

def action_to_target_idx(action: list[int] | tuple[int, int, int]) -> int | None:
    """
    Maps specific 3-bit actions to a target index (0, 1, 2).
    Returns None if the action doesn't correspond to a standard chest interaction.
    """
    action = [int(x) for x in action]
    if action == [1, 0, 0]: return 0
    if action == [0, 1, 0]: return 1
    if action == [0, 0, 1]: return 2
    return None

def target_idx_to_action(target_idx: int) -> list[int]:
    """Maps a target index (0, 1, 2) back to its corresponding 3-bit action."""
    if target_idx == 0: return [1, 0, 0]
    if target_idx == 1: return [0, 1, 0]
    if target_idx == 2: return [0, 0, 1]
    raise ValueError(f"Invalid target_idx: {target_idx}")

def encode_obs(obs: dict[str, Any]) -> np.ndarray:
    """
    Standardizes a single observation dictionary into a numerical feature vector.
    Used by both MLP and LSTM.
    """
    active = [int(x) for x in obs["active"]]
    opened = [int(x) for x in obs["open"]]

    e_type = int(obs["e_type"])
    fg = int(obs["fg"])
    bg = int(obs["bg"])

    # Handle both single-value lists and raw floats
    start = float(obs["start"][0]) if isinstance(obs["start"], (list, np.ndarray)) else float(obs["start"])
    end = float(obs["end"][0]) if isinstance(obs["end"], (list, np.ndarray)) else float(obs["end"])
    duration = float(obs["duration"][0]) if isinstance(obs["duration"], (list, np.ndarray)) else float(obs["duration"])

    e_type_onehot = [0] * NUM_EVENT_TYPES
    e_type_onehot[e_type] = 1

    fg_onehot = [0] * NUM_COLORS
    fg_onehot[fg] = 1

    bg_onehot = [0] * NUM_COLORS
    bg_onehot[bg] = 1

    features = (
        active
        + opened
        + e_type_onehot
        + fg_onehot
        + bg_onehot
        + [start, end, duration]
    )

    return np.array(features, dtype=np.float32)

def encode_history(history: list[dict[str, Any]]) -> np.ndarray:
    """
    Stacks a list of observations into a (sequence_length, feature_dim) matrix.
    Essential for LSTM inference.
    """
    if len(history) == 0:
        raise ValueError("History must contain at least one observation.")

    encoded = [encode_obs(obs) for obs in history]
    return np.stack(encoded, axis=0)

def normalize_action(action: list[int] | tuple[int, int, int] | np.ndarray) -> list[int]:
    """Ensures action is a clean list of integers."""
    return [int(x) for x in action]