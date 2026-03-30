import gymnasium as gym
from src.openthechests.register_envs import register_custom_envs

def run_episode(env_name="OpenTheChests-v0", max_steps=100):
    env = gym.make(env_name)
    obs, info = env.reset()
    trajectory = []

    for step in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        trajectory.append({
            "step": step,
            "obs": obs,
            "action": action.tolist(),
            "reward": reward,
            "next_obs": next_obs,
            "terminated": terminated,
            "truncated": truncated,
            "info": step_info,
        })

        obs = next_obs
        if terminated or truncated:
            break

    env.close()
    return trajectory

def main():
    register_custom_envs()

    for env_name in ["OpenTheChests-v0", "OpenTheChests-v1", "OpenTheChests-v2"]:
        traj = run_episode(env_name=env_name, max_steps=100)
        print(f"\n{env_name} finished with {len(traj)} steps")
        print("First step:")
        print(traj[0])
        print("Last step:")
        print(traj[-1])

if __name__ == "__main__":
    main()