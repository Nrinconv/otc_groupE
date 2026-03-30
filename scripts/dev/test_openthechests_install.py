import gymnasium as gym
from register_envs import register_custom_envs

def main():
    register_custom_envs()

    env_names = ["OpenTheChests-v0", "OpenTheChests-v1", "OpenTheChests-v2"]

    for env_name in env_names:
        print(f"\n{'=' * 60}")
        print(f"Testing {env_name}")
        print(f"{'=' * 60}")

        env = gym.make(env_name)

        obs, info = env.reset()

        print("Observation type:", type(obs))
        print("Observation:", obs)
        print("Info:", info)
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        print("Sample action:", action)
        print("Next observation:", next_obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Step info:", step_info)

        env.close()

if __name__ == "__main__":
    main()
