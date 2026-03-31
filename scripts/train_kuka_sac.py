from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.kuka.colored_chest_kuka_env  # noqa: F401  # register env


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a SAC controller for ColoredChestKuka-v0.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--reward-type", type=str, default="advanced", choices=["basic", "advanced"])
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="models/kuka/sac_colored_chest")
    parser.add_argument("--log-dir", type=str, default="results/kuka/sac_logs")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--use-sde", action="store_true", help="Enable generalized state-dependent exploration.")
    parser.add_argument("--use-tensorboard", action="store_true", help="Enable TensorBoard logging if tensorboard is installed.")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    try:
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "Training requires gymnasium and stable-baselines3. Install project requirements first."
        ) from exc

    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        gym.make(
            "ColoredChestKuka-v0",
            render_mode=None,
            reward_type=args.reward_type,
            max_steps=args.max_steps,
        )
    )

    eval_env = Monitor(
        gym.make(
            "ColoredChestKuka-v0",
            render_mode=None,
            reward_type=args.reward_type,
            max_steps=args.max_steps,
        )
    )

    tensorboard_log = None
    if args.use_tensorboard:
        try:
            import tensorboard  # noqa: F401
            tensorboard_log = str(log_dir / "tb")
        except Exception:
            print("TensorBoard requested but not installed. Continuing without TensorBoard logging.")

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        use_sde=args.use_sde,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=args.seed,
        policy_kwargs={"net_arch": [256, 256]},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, log_interval=10, progress_bar=True)

    model_path = REPO_ROOT / args.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    metadata = {
        "algorithm": "SAC",
        "env_id": "ColoredChestKuka-v0",
        "reward_type": args.reward_type,
        "max_steps": args.max_steps,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "gamma": args.gamma,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "use_sde": args.use_sde,
        "saved_model": str(model_path.with_suffix(".zip").relative_to(REPO_ROOT)),
    }

    metadata_path = model_path.parent / "sac_training_config.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Saved model to: {model_path}.zip")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
