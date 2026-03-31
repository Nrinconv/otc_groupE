from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import colored_chest_kuka_env  # noqa: F401  # ensures env registration


@dataclass
class KukaRunResult:
    success: bool
    target_idx: int
    steps: int
    episode_return: float
    final_distance: float
    terminated: bool
    truncated: bool
    controller_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProportionalReachPolicy:
    """
    Deterministic baseline controller.

    The KUKA observation already contains the current end-effector position and
    the target chest position, so a proportional Cartesian controller is a clean
    baseline for integration and debugging.
    """

    def __init__(self, action_scale: float = 0.05, gain: float = 1.0, hold_tolerance: float = 0.02) -> None:
        self.action_scale = float(action_scale)
        self.gain = float(gain)
        self.hold_tolerance = float(hold_tolerance)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        ee_pos = obs[0:3]
        target_pos = obs[3:6]
        error = target_pos - ee_pos
        distance = float(np.linalg.norm(error))

        if distance <= self.hold_tolerance:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = self.gain * error
            action = np.clip(action, -self.action_scale, self.action_scale).astype(np.float32)

        return action, None


class KukaChestController:
    """
    Public inference API for Member 2.

    Modes
    -----
    - ``heuristic``: proportional Cartesian reach controller
    - ``ppo``: load a Stable-Baselines3 PPO policy from disk
    - ``auto``: use PPO if the model file exists, otherwise fall back to the heuristic controller
    """

    def __init__(
        self,
        env_id: str = "ColoredChestKuka-v0",
        mode: str = "auto",
        model_path: str = "models/kuka/ppo_colored_chest.zip",
        reward_type: str = "advanced",
        max_steps: int = 150,
        render_mode: str | None = None,
        deterministic: bool = True,
        heuristic_gain: float = 1.0,
        heuristic_hold_tolerance: float = 0.02,
    ) -> None:
        self.env_id = env_id
        self.mode = mode
        self.model_path = Path(model_path)
        self.reward_type = reward_type
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.deterministic = bool(deterministic)
        self.heuristic_gain = float(heuristic_gain)
        self.heuristic_hold_tolerance = float(heuristic_hold_tolerance)

        self._policy: Any | None = None
        self._resolved_mode: str | None = None
        self._vecnormalize_path: Path | None = None

    def _resolve_mode(self) -> str:
        if self.mode not in {"auto", "heuristic", "ppo"}:
            raise ValueError("mode must be one of 'auto', 'heuristic', or 'ppo'.")

        if self.mode == "auto":
            return "ppo" if self.model_path.exists() else "heuristic"
        return self.mode

    def _make_plain_env(self):
        import gymnasium as gym

        return gym.make(
            self.env_id,
            render_mode=self.render_mode,
            reward_type=self.reward_type,
            max_steps=self.max_steps,
        )

    def _guess_vecnormalize_path(self) -> Path:
        candidates = [
            self.model_path.parent / "ppo_vecnormalize.pkl",
            Path("models/kuka/ppo_vecnormalize.pkl"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "VecNormalize stats file not found. Expected one of:\n"
            f"  - {candidates[0]}\n"
            f"  - {candidates[1]}\n"
            "Retrain PPO with the updated script or copy ppo_vecnormalize.pkl next to the model."
        )

    def _load_policy_once(self, env_action_scale: float):
        resolved_mode = self._resolve_mode()

        if self._policy is not None and self._resolved_mode == resolved_mode:
            return self._policy

        if resolved_mode == "heuristic":
            self._policy = ProportionalReachPolicy(
                action_scale=env_action_scale,
                gain=self.heuristic_gain,
                hold_tolerance=self.heuristic_hold_tolerance,
            )
            self._resolved_mode = resolved_mode
            self._vecnormalize_path = None
            return self._policy

        try:
            from stable_baselines3 import PPO
        except Exception as exc:
            raise ImportError(
                "Stable-Baselines3 is required to use mode='ppo'. Install project requirements first."
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PPO model not found at {self.model_path}. "
                "Train it first or use mode='heuristic'."
            )

        self._policy = PPO.load(str(self.model_path), device="cpu")
        self._resolved_mode = resolved_mode
        self._vecnormalize_path = self._guess_vecnormalize_path()
        return self._policy

    def _run_heuristic(self, target_idx: int, seed: int | None = None) -> dict[str, Any]:
        env = self._make_plain_env()
        try:
            action_scale = float(getattr(env.unwrapped, "action_scale", 0.05))
            policy = self._load_policy_once(env_action_scale=action_scale)

            obs, info = env.reset(seed=seed, options={"target_idx": int(target_idx)})
            done = False
            truncated = False
            episode_return = 0.0
            steps = 0
            last_info = dict(info)

            while not done and not truncated:
                action, _ = policy.predict(obs, deterministic=self.deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                last_info = dict(info)

            result = KukaRunResult(
                success=bool(last_info.get("is_success", False)),
                target_idx=int(target_idx),
                steps=int(steps),
                episode_return=float(episode_return),
                final_distance=float(last_info.get("distance_to_target", np.nan)),
                terminated=bool(done),
                truncated=bool(truncated),
                controller_mode=str(self._resolved_mode),
            )
            return result.to_dict()
        finally:
            env.close()

    def _run_ppo(self, target_idx: int, seed: int | None = None) -> dict[str, Any]:
        try:
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except Exception as exc:
            raise ImportError(
                "Stable-Baselines3 is required to use mode='ppo'. Install project requirements first."
            ) from exc

        plain_env = self._make_plain_env()
        try:
            action_scale = float(getattr(plain_env.unwrapped, "action_scale", 0.05))
        finally:
            plain_env.close()

        policy = self._load_policy_once(env_action_scale=action_scale)
        vecnormalize_path = self._vecnormalize_path
        if vecnormalize_path is None:
            raise RuntimeError("VecNormalize path was not initialized for PPO mode.")

        vec_env = make_vec_env(
            env_id=self.env_id,
            n_envs=1,
            seed=seed,
            env_kwargs={
                "render_mode": self.render_mode,
                "reward_type": self.reward_type,
                "max_steps": self.max_steps,
            },
            vec_env_cls=DummyVecEnv,
        )

        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        try:
            base_env = vec_env.venv.envs[0]
            raw_obs, _ = base_env.reset(seed=seed, options={"target_idx": int(target_idx)})
            obs = vec_env.normalize_obs(np.asarray([raw_obs], dtype=np.float32)).copy()

            done = False
            truncated = False
            episode_return = 0.0
            steps = 0
            last_info: dict[str, Any] = {}

            while not done and not truncated:
                action, _ = policy.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = vec_env.step(action)
                reward = float(rewards[0])
                info = infos[0]
                done = bool(dones[0])
                truncated = bool(info.get("TimeLimit.truncated", False))
                episode_return += reward
                steps += 1
                last_info = dict(info)

            result = KukaRunResult(
                success=bool(last_info.get("is_success", False)),
                target_idx=int(target_idx),
                steps=int(steps),
                episode_return=float(episode_return),
                final_distance=float(last_info.get("distance_to_target", np.nan)),
                terminated=bool(done and not truncated),
                truncated=bool(truncated),
                controller_mode=str(self._resolved_mode),
            )
            return result.to_dict()
        finally:
            vec_env.close()

    def run(self, target_idx: int, seed: int | None = None) -> dict[str, Any]:
        if target_idx not in {0, 1, 2}:
            raise ValueError("target_idx must be 0, 1, or 2.")

        resolved_mode = self._resolve_mode()

        if resolved_mode == "heuristic":
            return self._run_heuristic(target_idx=target_idx, seed=seed)

        if resolved_mode == "ppo":
            return self._run_ppo(target_idx=target_idx, seed=seed)

        raise RuntimeError(f"Unsupported resolved mode: {resolved_mode}")

    def evaluate(
        self,
        n_episodes: int = 30,
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        if seeds is None:
            seeds = list(range(n_episodes))
        if not seeds:
            raise ValueError("seeds must not be empty.")

        runs = []
        for i in range(n_episodes):
            target_idx = i % 3
            seed = seeds[i % len(seeds)]
            runs.append(self.run(target_idx=target_idx, seed=seed))

        success_rate = float(np.mean([r["success"] for r in runs])) if runs else 0.0
        mean_steps = float(np.mean([r["steps"] for r in runs])) if runs else 0.0
        mean_final_distance = float(np.mean([r["final_distance"] for r in runs])) if runs else float("nan")

        by_target: dict[str, dict[str, float]] = {}
        for target_idx in range(3):
            subset = [r for r in runs if r["target_idx"] == target_idx]
            if not subset:
                continue
            by_target[str(target_idx)] = {
                "success_rate": float(np.mean([r["success"] for r in subset])),
                "mean_steps": float(np.mean([r["steps"] for r in subset])),
                "mean_final_distance": float(np.mean([r["final_distance"] for r in subset])),
            }

        return {
            "controller_mode": self._resolve_mode(),
            "env_id": self.env_id,
            "reward_type": self.reward_type,
            "max_steps": self.max_steps,
            "n_episodes": int(n_episodes),
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "mean_final_distance": mean_final_distance,
            "by_target": by_target,
            "runs": runs,
        }