# Integration Contract

## Purpose

This repository now exposes two stable layers that Member 3 can connect directly:

- the **OpenTheChests decision layer**
- the **KUKA control layer**

The purpose of this document is to define the public interfaces that should be used during integration.

---

## Layer 1 - OpenTheChests decision module

### Public API

```python
from src.openthechests.predict import OpenTheChestsPredictor
```

### Expected input

- `history`: observation history collected during an `OpenTheChests` episode
- `env_name`: one of:
  - `OpenTheChests-v0`
  - `OpenTheChests-v1`
  - `OpenTheChests-v2`

### Expected output

```python
target_idx = predictor.predict_target(history, env_name)
```

Returned values:

- `0` -> red chest
- `1` -> green chest
- `2` -> blue chest
- `None` -> no unique decision yet

### Recommended configuration

```python
predictor = OpenTheChestsPredictor(
    model_dir="models/openthechests",
    mode="auto",
)
```

This automatically uses:

- MLP for `OpenTheChests-v0`
- MLP for `OpenTheChests-v1`
- LSTM for `OpenTheChests-v2`

---

## Layer 2 - KUKA control module

### Public API

```python
from src.kuka.controller import KukaChestController
```

### Expected input

```python
result = controller.run(target_idx=target_idx, seed=seed)
```

Input constraints:

- `target_idx` must be one of `{0, 1, 2}`
- target convention must remain:
  - `0 = red`
  - `1 = green`
  - `2 = blue`

### Output format

The returned dictionary contains:

- `success`
- `target_idx`
- `steps`
- `episode_return`
- `final_distance`
- `terminated`
- `truncated`
- `controller_mode`

### Recommended controller modes

#### For robust integration

```python
controller = KukaChestController(
    mode="heuristic",
    reward_type="advanced",
)
```

#### For learned-control evaluation

```python
controller = KukaChestController(
    mode="ppo",
    model_path="models/kuka/ppo_colored_chest.zip",
    reward_type="advanced",
)
```

### Important note

For Member 3, `mode="auto"` is **not** recommended for final experiments. The mode should be selected explicitly so that the experiment objective is clear.

---

## Expected end-to-end flow

The intended full-pipeline logic is:

```python
history = []

# 1. run an OpenTheChests episode and collect the history
# 2. ask the predictor for a chest decision

target_idx = predictor.predict_target(history, env_name)

# 3. if a valid chest is predicted, send it to the KUKA layer
if target_idx is not None:
    result = controller.run(target_idx=target_idx, seed=seed)
else:
    result = None
```

---

## What Member 3 should not change

To keep the repository stable, Member 3 should avoid modifying:

- internal training code of the OpenTheChests models
- internal training code of the KUKA PPO model
- target index convention
- public API names in `predict.py` and `controller.py`

---

## What Member 3 is expected to build

Member 3 should implement the integration layer that:

- runs the OpenTheChests environment
- stores the observation history in the format expected by the predictor
- calls the predictor to obtain `target_idx`
- calls the KUKA controller with this `target_idx`
- evaluates end-to-end success across several episodes and environments

---

## Final recommendation

For the first full integration tests, Member 3 should start with:

- `OpenTheChestsPredictor(mode="auto")`
- `KukaChestController(mode="heuristic")`

This is the most reliable way to validate the whole pipeline.

Once the pipeline is stable, Member 3 can run a second set of experiments with:

- `OpenTheChestsPredictor(mode="auto")`
- `KukaChestController(mode="ppo")`

This allows the final report to compare a robust integration controller against a learned continuous-control controller.
