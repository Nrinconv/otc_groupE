from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.openthechests.register_envs import register_custom_envs

VALID_ENVS = {
    "OpenTheChests-v0",
    "OpenTheChests-v1",
    "OpenTheChests-v2",
}

def all_actions() -> list[np.ndarray]:
    """Generates all 8 possible 3-bit binary action combinations."""
    return [np.array(a, dtype=np.int8) for a in itertools.product([0, 1], repeat=3)]

def action_to_key(action: list[int] | np.ndarray) -> str:
    """Converts action list to string key (e.g. [1, 0, 1] -> '101')."""
    return "".join(str(int(x)) for x in action)

def obs_to_dict(obs: dict) -> dict:
    """Standardizes observation dictionary for JSON serialization."""
    return {
        "active": obs["active"].astype(int).tolist(),
        "open": obs["open"].astype(int).tolist(),
        "e_type": int(obs["e_type"]),
        "fg": int(obs["fg"]),
        "bg": int(obs["bg"]),
        "start": float(obs["start"][0]) if isinstance(obs["start"], (list, np.ndarray)) else float(obs["start"]),
        "end": float(obs["end"][0]) if isinstance(obs["end"], (list, np.ndarray)) else float(obs["end"]),
        "duration": float(obs["duration"][0]) if isinstance(obs["duration"], (list, np.ndarray)) else float(obs["duration"]),
    }

def replay_prefix(env_name: str, seed: int, prefix_actions: list[list[int]]):
    """Recreates environment state by replaying a sequence of actions."""
    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)

    for action in prefix_actions:
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.int8))
        if terminated or truncated:
            break

    return env, obs

def evaluate_all_actions(env_name: str, seed: int, prefix_actions: list[list[int]]) -> list[dict]:
    """Brute-forces every action from current state to find the optimal move."""
    results = []
    for candidate in all_actions():
        env, obs = replay_prefix(env_name, seed, prefix_actions)
        next_obs, reward, terminated, truncated, info = env.step(candidate)

        results.append({
            "obs": obs,
            "action": candidate.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "next_obs": next_obs,
        })
        env.close()
    return results

def choose_best_action(results: list[dict]) -> dict:
    """
    Selects the optimal action based on immediate reward.
    Tie-breaker: prefers actions that interact with fewer chests (efficiency).
    """
    max_reward = max(r["reward"] for r in results)
    best_candidates = [r for r in results if r["reward"] == max_reward]
    
    # Sort by number of active bits in action (fewer is better)
    best_candidates.sort(key=lambda r: sum(int(x) for x in r["action"]))
    return best_candidates[0]

def collect_dataset(env_name: str, n_seeds: int, max_steps: int) -> list[dict]:
    """Generates the Oracle dataset across multiple seeds."""
    dataset = []

    for seed in range(n_seeds):
        env = gym.make(env_name)
        obs, _ = env.reset(seed=seed)
        prefix_actions: list[list[int]] = []

        for step in range(max_steps):
            oracle_results = evaluate_all_actions(env_name, seed, prefix_actions)
            best = choose_best_action(oracle_results)

            reward_map = {
                action_to_key(r["action"]): r["reward"]
                for r in oracle_results
            }

            dataset.append({
                "env_name": env_name,
                "seed": int(seed),
                "step": int(step),
                "obs": obs_to_dict(obs),
                "target_action": best["action"],
                "target_reward": best["reward"],
                "all_action_rewards": reward_map,
            })

            # Advance the 'real' trajectory
            action = np.array(best["action"], dtype=np.int8)
            obs, reward, terminated, truncated, _ = env.step(action)
            prefix_actions.append(best["action"])

            if terminated or truncated:
                break
        env.close()

    return dataset

def main():
    parser = argparse.ArgumentParser(description="Oracle dataset generator for OpenTheChests")
    parser.add_argument("--env", type=str, required=True, help="v0, v1, or v2")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    args = parser.parse_args()

    if args.env not in VALID_ENVS:
        raise ValueError(f"Unsupported env: {args.env}")

    register_custom_envs()
    dataset = collect_dataset(args.env, args.n_seeds, args.max_steps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")

    print(f"Done. Saved {len(dataset)} samples to {out_path}")

if __name__ == "__main__":
    main()