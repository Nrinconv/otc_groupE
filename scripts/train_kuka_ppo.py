from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.kuka.colored_chest_kuka_env  # noqa: F401  # register env


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO controller for ColoredChestKuka-v0.")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--reward-type", type=str, default="advanced", choices=["basic", "advanced"])
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="models/kuka/ppo_colored_chest")
    parser.add_argument("--log-dir", type=str, default="results/kuka/ppo_logs")
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        help="Enable TensorBoard logging if tensorboard is installed.",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    try:
        import torch.nn as nn
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    except Exception as exc:
        raise ImportError(
            "Training requires stable-baselines3, torch, and gymnasium. Install project requirements first."
        ) from exc

    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    def env_kwargs() -> dict:
        return {
            "render_mode": None,
            "reward_type": args.reward_type,
            "max_steps": args.max_steps,
        }

    train_env = make_vec_env(
        env_id="ColoredChestKuka-v0",
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs(),
        monitor_dir=str(log_dir / "train_monitor"),
        vec_env_cls=SubprocVecEnv,
    )

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True,
    )

    eval_env = make_vec_env(
        env_id="ColoredChestKuka-v0",
        n_envs=1,
        seed=args.seed + 10_000,
        env_kwargs=env_kwargs(),
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    tensorboard_log = None
    if args.use_tensorboard:
        try:
            import tensorboard  # noqa: F401
            tensorboard_log = str(log_dir / "tb")
        except Exception:
            print("TensorBoard requested but not installed. Continuing without TensorBoard logging.")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=8,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=args.seed,
        device="cpu",
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    eval_freq = max(10_000 // max(1, args.n_envs), 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=50,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_kuka",
        save_vecnormalize=True,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)

    model_path = REPO_ROOT / args.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    vecnorm_path = model_path.parent / "ppo_vecnormalize.pkl"
    train_env.save(str(vecnorm_path))

    metadata = {
        "algorithm": "PPO",
        "env_id": "ColoredChestKuka-v0",
        "reward_type": args.reward_type,
        "max_steps": args.max_steps,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "saved_model": str(model_path.with_suffix(".zip").relative_to(REPO_ROOT)),
        "saved_vecnormalize": str(vecnorm_path.relative_to(REPO_ROOT)),
        "hyperparameters": {
            "learning_rate": 3e-5,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 20,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "clip_range": 0.4,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": True,
            "sde_sample_freq": 4,
            "policy_net": [256, 256],
            "value_net": [256, 256],
            "activation": "ReLU",
            "ortho_init": False,
            "log_std_init": -2,
        },
    }

    metadata_path = model_path.parent / "ppo_training_config.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    train_env.close()
    eval_env.close()

    print("Training complete.")
    print(f"Saved model to: {model_path}.zip")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()