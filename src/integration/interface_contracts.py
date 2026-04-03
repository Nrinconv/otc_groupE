from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import numpy as np

from src.openthechests.collect_dataset import (
    choose_best_action,
    evaluate_all_actions,
)
from src.openthechests.encode_events import action_to_target_idx
from src.openthechests.register_envs import register_custom_envs

if TYPE_CHECKING:
    from src.kuka.controller import KukaChestController
    from src.openthechests.predict import OpenTheChestsPredictor

VALID_OPENTHECHESTS_ENVS = {
    "OpenTheChests-v0",
    "OpenTheChests-v1",
    "OpenTheChests-v2",
}

RolloutPolicy = Literal["oracle", "random"]


def obs_to_history_item(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert one environment observation into the predictor input format."""
    return {
        "active": np.asarray(obs["active"]).astype(int).tolist(),
        "open": np.asarray(obs["open"]).astype(int).tolist(),
        "e_type": int(obs["e_type"]),
        "fg": int(obs["fg"]),
        "bg": int(obs["bg"]),
        "start": float(obs["start"][0]) if isinstance(obs["start"], (list, np.ndarray)) else float(obs["start"]),
        "end": float(obs["end"][0]) if isinstance(obs["end"], (list, np.ndarray)) else float(obs["end"]),
        "duration": float(obs["duration"][0]) if isinstance(obs["duration"], (list, np.ndarray)) else float(obs["duration"]),
    }


@dataclass
class OpenTheChestsEpisode:
    env_name: str
    seed: int | None
    rollout_policy: str
    history: list[dict[str, Any]]
    actions: list[list[int]]
    rewards: list[float]
    terminated: bool
    truncated: bool
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrationRunResult:
    env_name: str
    seed: int | None
    rollout_policy: str
    predicted_action: list[int]
    predicted_target_idx: int | None
    decision_steps: int
    history: list[dict[str, Any]]
    openthechests_terminated: bool
    openthechests_truncated: bool
    kuka_result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DualAgentStepRecord:
    step: int
    history_length: int
    observation: dict[str, Any]
    predicted_action: list[int]
    predicted_target_idx: int | None
    otc_reward: float
    otc_terminated: bool
    otc_truncated: bool
    kuka_result: dict[str, Any] | None
    info_otc: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DualAgentEpisodeResult:
    env_name: str
    seed: int | None
    total_steps: int
    opened_targets: list[int]
    otc_return: float
    otc_terminated: bool
    otc_truncated: bool
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenTheChestsKukaInterface:
    """
    Bridge layer between the symbolic decision module and the KUKA controller.

    The interface keeps the contract intentionally small:
    - collect an OpenTheChests history in the exact format required by the predictor
    - ask the predictor for a chest target
    - forward the target to the KUKA controller
    """

    def __init__(
        self,
        predictor: OpenTheChestsPredictor | None = None,
        controller: KukaChestController | None = None,
        predictor_model_dir: str = "models/openthechests",
        predictor_mode: str = "auto",
        kuka_mode: str = "heuristic",
        kuka_reward_type: str = "advanced",
        kuka_render_mode: str | None = None,
        kuka_max_steps: int = 150,
    ) -> None:
        from src.kuka.controller import KukaChestController
        from src.openthechests.predict import OpenTheChestsPredictor

        register_custom_envs()
        self.predictor = predictor or OpenTheChestsPredictor(
            model_dir=predictor_model_dir,
            mode=predictor_mode,
        )
        self.controller = controller or KukaChestController(
            mode=kuka_mode,
            reward_type=kuka_reward_type,
            render_mode=kuka_render_mode,
            max_steps=kuka_max_steps,
        )

    def _normalize_binary_action(self, action: list[int] | tuple[int, ...] | np.ndarray) -> list[int]:
        normalized = [int(x) for x in np.asarray(action).astype(int).tolist()]
        if len(normalized) != 3:
            raise ValueError(f"Expected a 3-bit action, got {normalized}.")
        return normalized

    def _validate_env_name(self, env_name: str) -> None:
        if env_name not in VALID_OPENTHECHESTS_ENVS:
            raise ValueError(
                f"Unsupported env_name: {env_name}. "
                f"Expected one of {sorted(VALID_OPENTHECHESTS_ENVS)}."
            )

    def _choose_action(
        self,
        env_name: str,
        seed: int | None,
        prefix_actions: list[list[int]],
        env: gym.Env,
        rollout_policy: RolloutPolicy,
    ) -> list[int]:
        if rollout_policy == "oracle":
            oracle_results = evaluate_all_actions(
                env_name=env_name,
                seed=0 if seed is None else int(seed),
                prefix_actions=prefix_actions,
            )
            return [int(x) for x in choose_best_action(oracle_results)["action"]]

        if rollout_policy == "random":
            action = env.action_space.sample()
            return [int(x) for x in np.asarray(action).astype(int).tolist()]

        raise ValueError("rollout_policy must be either 'oracle' or 'random'.")

    def collect_episode(
        self,
        env_name: str,
        seed: int | None = None,
        max_steps: int = 30,
        rollout_policy: RolloutPolicy = "oracle",
    ) -> OpenTheChestsEpisode:
        self._validate_env_name(env_name)

        env = gym.make(env_name)
        try:
            obs, _ = env.reset(seed=seed)
            history: list[dict[str, Any]] = []
            actions: list[list[int]] = []
            rewards: list[float] = []
            prefix_actions: list[list[int]] = []
            terminated = False
            truncated = False

            for _ in range(max_steps):
                history.append(obs_to_history_item(obs))

                action = self._choose_action(
                    env_name=env_name,
                    seed=seed,
                    prefix_actions=prefix_actions,
                    env=env,
                    rollout_policy=rollout_policy,
                )

                obs, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.int8))
                actions.append(action)
                rewards.append(float(reward))
                prefix_actions.append(action)

                if terminated or truncated:
                    break

            return OpenTheChestsEpisode(
                env_name=env_name,
                seed=seed,
                rollout_policy=rollout_policy,
                history=history,
                actions=actions,
                rewards=rewards,
                terminated=bool(terminated),
                truncated=bool(truncated),
                steps=len(actions),
            )
        finally:
            env.close()

    def predict_target_from_episode(self, episode: OpenTheChestsEpisode) -> tuple[list[int], int | None]:
        predicted_action = self.predictor.predict_action(
            history=episode.history,
            env_name=episode.env_name,
        )
        predicted_target_idx = self.predictor.predict_target(
            history=episode.history,
            env_name=episode.env_name,
        )
        return predicted_action, predicted_target_idx

    def run_dual_agent_episode(
        self,
        env_name: str,
        seed: int | None = None,
        max_steps: int = 30,
        invoke_kuka_once_per_target: bool = True,
    ) -> DualAgentEpisodeResult:
        """
        Run the cooperative two-agent loop.

        Agent 1 observes the OpenTheChests stream and predicts a 3-bit action at
        every step. When that action maps to one unique chest target, Agent 2 is
        called immediately to execute the physical interaction in the KUKA
        environment. The same Agent 1 action is then applied to the OtC
        environment so the symbolic environment can continue evolving.
        """
        self._validate_env_name(env_name)

        env = gym.make(env_name)
        try:
            obs, _ = env.reset(seed=seed)
            history: list[dict[str, Any]] = []
            step_records: list[dict[str, Any]] = []
            opened_targets: list[int] = []
            already_invoked_targets: set[int] = set()
            otc_return = 0.0
            terminated = False
            truncated = False
            step_count = 0

            while step_count < max_steps and not terminated and not truncated:
                history_item = obs_to_history_item(obs)
                history.append(history_item)

                predicted_action = self._normalize_binary_action(
                    self.predictor.predict_action(history=history, env_name=env_name)
                )
                predicted_target_idx = action_to_target_idx(predicted_action)

                kuka_result = None
                should_invoke_kuka = predicted_target_idx is not None
                if invoke_kuka_once_per_target and predicted_target_idx in already_invoked_targets:
                    should_invoke_kuka = False

                if should_invoke_kuka and predicted_target_idx is not None:
                    kuka_result = self.controller.run(target_idx=predicted_target_idx, seed=seed)
                    opened_targets.append(int(predicted_target_idx))
                    already_invoked_targets.add(int(predicted_target_idx))

                obs, reward, terminated, truncated, info_otc = env.step(
                    np.asarray(predicted_action, dtype=np.int8)
                )
                otc_return += float(reward)

                step_records.append(
                    DualAgentStepRecord(
                        step=step_count,
                        history_length=len(history),
                        observation=history_item,
                        predicted_action=predicted_action,
                        predicted_target_idx=predicted_target_idx,
                        otc_reward=float(reward),
                        otc_terminated=bool(terminated),
                        otc_truncated=bool(truncated),
                        kuka_result=kuka_result,
                        info_otc=dict(info_otc),
                    ).to_dict()
                )
                step_count += 1

            return DualAgentEpisodeResult(
                env_name=env_name,
                seed=seed,
                total_steps=step_count,
                opened_targets=opened_targets,
                otc_return=float(otc_return),
                otc_terminated=bool(terminated),
                otc_truncated=bool(truncated),
                steps=step_records,
            )
        finally:
            env.close()

    def run_episode(
        self,
        env_name: str,
        seed: int | None = None,
        max_steps: int = 30,
        rollout_policy: RolloutPolicy = "oracle",
    ) -> IntegrationRunResult:
        episode = self.collect_episode(
            env_name=env_name,
            seed=seed,
            max_steps=max_steps,
            rollout_policy=rollout_policy,
        )
        predicted_action, predicted_target_idx = self.predict_target_from_episode(episode)

        kuka_result = None
        if predicted_target_idx is not None:
            kuka_result = self.controller.run(target_idx=predicted_target_idx, seed=seed)

        return IntegrationRunResult(
            env_name=env_name,
            seed=seed,
            rollout_policy=rollout_policy,
            predicted_action=predicted_action,
            predicted_target_idx=predicted_target_idx,
            decision_steps=episode.steps,
            history=episode.history,
            openthechests_terminated=episode.terminated,
            openthechests_truncated=episode.truncated,
            kuka_result=kuka_result,
        )

    def evaluate_dual_agent(
        self,
        env_names: list[str] | None = None,
        seeds: list[int] | None = None,
        max_steps: int = 30,
        invoke_kuka_once_per_target: bool = True,
    ) -> dict[str, Any]:
        env_names = env_names or sorted(VALID_OPENTHECHESTS_ENVS)
        seeds = seeds or list(range(5))

        runs: list[dict[str, Any]] = []
        for env_name in env_names:
            self._validate_env_name(env_name)
            for seed in seeds:
                result = self.run_dual_agent_episode(
                    env_name=env_name,
                    seed=seed,
                    max_steps=max_steps,
                    invoke_kuka_once_per_target=invoke_kuka_once_per_target,
                )
                runs.append(result.to_dict())

        kuka_runs = [
            step["kuka_result"]
            for run in runs
            for step in run["steps"]
            if step["kuka_result"] is not None
        ]

        return {
            "env_names": env_names,
            "seeds": seeds,
            "max_steps": int(max_steps),
            "invoke_kuka_once_per_target": bool(invoke_kuka_once_per_target),
            "n_runs": len(runs),
            "n_kuka_invocations": len(kuka_runs),
            "kuka_success_rate": (
                float(np.mean([run["success"] for run in kuka_runs]))
                if kuka_runs
                else 0.0
            ),
            "runs": runs,
        }

    def evaluate(
        self,
        env_names: list[str] | None = None,
        seeds: list[int] | None = None,
        max_steps: int = 30,
        rollout_policy: RolloutPolicy = "oracle",
    ) -> dict[str, Any]:
        env_names = env_names or sorted(VALID_OPENTHECHESTS_ENVS)
        seeds = seeds or list(range(5))

        runs: list[dict[str, Any]] = []
        for env_name in env_names:
            self._validate_env_name(env_name)
            for seed in seeds:
                result = self.run_episode(
                    env_name=env_name,
                    seed=seed,
                    max_steps=max_steps,
                    rollout_policy=rollout_policy,
                )
                runs.append(result.to_dict())

        kuka_runs = [r["kuka_result"] for r in runs if r["kuka_result"] is not None]
        end_to_end_success_rate = (
            float(np.mean([run["success"] for run in kuka_runs]))
            if kuka_runs
            else 0.0
        )

        return {
            "env_names": env_names,
            "seeds": seeds,
            "rollout_policy": rollout_policy,
            "n_runs": len(runs),
            "decision_rate": float(np.mean([r["predicted_target_idx"] is not None for r in runs])) if runs else 0.0,
            "end_to_end_success_rate": end_to_end_success_rate,
            "runs": runs,
        }
