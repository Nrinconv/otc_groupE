import itertools
import gymnasium as gym
import numpy as np

from src.openthechests.register_envs import register_custom_envs


def all_actions():
    return [np.array(a, dtype=np.int8) for a in itertools.product([0, 1], repeat=3)]


def make_env(env_name: str):
    return gym.make(env_name)


def replay_until_step(env_name: str, seed: int, prefix_actions):
    env = make_env(env_name)
    obs, info = env.reset(seed=seed)

    for action in prefix_actions:
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.int8))
        if terminated or truncated:
            break

    return env, obs, info


def evaluate_actions_at_step(env_name: str, seed: int, prefix_actions):
    results = []

    for candidate in all_actions():
        env, obs, info = replay_until_step(env_name, seed, prefix_actions)

        next_obs, reward, terminated, truncated, step_info = env.step(candidate)

        results.append({
            "obs": obs,
            "candidate_action": candidate.tolist(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_obs": next_obs,
            "info": step_info,
        })

        env.close()

    return results


def main():
    register_custom_envs()

    env_name = "OpenTheChests-v0"
    seed = 123

    # Example prefix: choose the state before step 0
    prefix_actions = []

    results = evaluate_actions_at_step(env_name, seed, prefix_actions)

    print(f"\nEvaluating all 8 actions for {env_name} with seed={seed}")
    for r in results:
        print(
            f"action={r['candidate_action']} | "
            f"reward={r['reward']} | "
            f"terminated={r['terminated']} | "
            f"truncated={r['truncated']}"
        )

    best = max(results, key=lambda x: x["reward"])
    print("\nBest action by immediate reward:")
    print(best["candidate_action"], "reward =", best["reward"])


if __name__ == "__main__":
    main()
