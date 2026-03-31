# KUKA Module

## Goal

This module is the **Member 2 contribution** of the project.

Its purpose is to receive a chest target index and move the KUKA robot arm to the corresponding chest in the PyBullet environment.

This is the **robot-control and physical reaction layer** of the project.

---

## Scope

This module covers:

- the custom `ColoredChestKuka-v0` environment
- a deterministic proportional reaching controller
- PPO training and evaluation
- a public controller API for Member 3

It does **not** include the OpenTheChests decision module.

---

## Main environment

The module currently works on:

- `ColoredChestKuka-v0`

The environment contains one KUKA arm and three colored chest targets.

The target convention is fixed and must remain compatible with the OpenTheChests module:

- `0 = red`
- `1 = green`
- `2 = blue`

---

## Directory overview

### `src/kuka/`
Core code for the KUKA module.

Important files:

- `colored_chest_kuka_env.py`  
  Custom Gymnasium PyBullet environment for the KUKA reaching task.

- `controller.py`  
  Public inference API exposing the heuristic and PPO controllers.

### `models/kuka/`
Saved trained PPO model and normalization statistics.

Expected files:

- `ppo_colored_chest.zip`
- `ppo_vecnormalize.pkl`
- `ppo_training_config.json`

### `results/kuka/`
Saved evaluation summaries and PPO training logs.

Typical contents:

- `controller_eval.json`
- `ppo_logs/`

### `scripts/`
Training and evaluation scripts for the KUKA module.

Important files:

- `train_kuka_ppo.py`  
  PPO training script.

- `eval_kuka_controller.py`  
  Unified evaluation script for the KUKA controllers.

- `scripts/dev/test_kuka_controller.py`  
  Small development script for API checks.

---

## Environment design

The environment is a continuous-control reaching task.

The action is a 3D Cartesian displacement applied to the end effector:

- `dx`
- `dy`
- `dz`

The action is clipped and converted to joint targets through inverse kinematics.

The observation is a 14-dimensional vector containing:

- end-effector position
- target chest top-center position
- relative target position
- current distance to the target
- one-hot target identity
- normalized episode progress

The reward used in the final experiments is `reward_type="advanced"`.

This version includes:

- dense distance reward
- progress-based shaping
- bonuses near the success radius
- an additional incentive to remain inside the success region for consecutive steps

The success condition is defined by:

- `success_distance = 0.06`
- `success_hold_steps = 5`

The simulation is advanced with multiple PyBullet steps per action in order to make the reaching dynamics more stable for PPO.

---

## Controllers explored

Two controller families were retained for the final module.

### Heuristic controller

A deterministic proportional controller using the Cartesian error between the end effector and the target chest.

This controller is simple, robust, and extremely useful for:

- debugging the environment
- validating the target convention
- giving Member 3 a reliable integration baseline

### PPO controller

A learned continuous-control policy trained directly in `ColoredChestKuka-v0`.

The final PPO training pipeline uses:

- `VecNormalize` for observation and reward normalization during training
- `VecNormalize` loading during evaluation
- periodic evaluation and checkpoint saving
- `use_sde=True` for stochastic exploration in continuous actions

---

## Public controller API

The module exposes:

```python
from src.kuka.controller import KukaChestController
```

Example usage:

```python
controller = KukaChestController(
    mode="heuristic",
    reward_type="advanced",
    max_steps=150,
)

result = controller.run(target_idx=1, seed=7)
```

The output format is a dictionary containing:

- `success`
- `target_idx`
- `steps`
- `episode_return`
- `final_distance`
- `terminated`
- `truncated`
- `controller_mode`

Supported modes for the project workflow are:

- `heuristic`
- `ppo`

---

## Recommended usage

For Member 3, two usages are recommended depending on the objective.

### Recommended for robust integration

```python
controller = KukaChestController(mode="heuristic")
```

This is the safest option when the objective is to validate the full pipeline end-to-end.

### Recommended for learned-control experiments

```python
controller = KukaChestController(
    mode="ppo",
    model_path="models/kuka/ppo_colored_chest.zip",
)
```

This is the correct option when the objective is to evaluate the learned KUKA policy.

Because the current code resolves `mode="auto"` to PPO whenever a model file exists, Member 3 should **not** rely on `auto` for final experiments. The controller mode should be selected explicitly.

---

## Training notes

The final PPO model was trained with:

- `reward_type="advanced"`
- `total_timesteps=1_000_000`
- vectorized environments
- normalized observations and rewards
- saved normalization statistics for evaluation

The training script saves:

- the final PPO model
- the `VecNormalize` statistics file
- a JSON metadata file with the training configuration

---

## Final takeaway

The KUKA module now provides both:

- a strong deterministic controller for robust integration
- a learned PPO controller that reaches a clearly useful level of performance

This is sufficient for Member 3 to continue with integration without changing the internal logic of the KUKA layer.
