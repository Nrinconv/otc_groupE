# Open The Chests Project

This repository contains the project structure for **Open The Chests**, whose objective is to build a system that can **recognize activities from event observations** and then **react physically by moving a KUKA robot arm to the correct chest**. The project is naturally divided into an activity-recognition layer, a robot-control layer, and a final integration layer.

## Project overview

The full system is organized around three components:

- an **OpenTheChests decision module**, which observes event histories and predicts which chest should be opened
- a **KUKA control module**, which receives a chest target and moves the robot arm to the corresponding chest
- an **integration layer**, which connects both modules and evaluates the full pipeline end-to-end

This repository is therefore structured so that each team member can work on one part without breaking the others.

## Team structure

### Nicolas RINCON — OpenTheChests activity recognition
This module is responsible for the activity-recognition and decision-making side of the project. It works on the custom environments:

- `OpenTheChests-v0`
- `OpenTheChests-v1`
- `OpenTheChests-v2`

These correspond to increasing difficulty levels. The goal of this module is to predict the appropriate chest-opening decision from the observation history.

### Member 2 — KUKA control
This module is responsible for the physical control side. It uses the KUKA PyBullet environment and receives a chest target index:

- `0 = red`
- `1 = green`
- `2 = blue`

It must then move the robot arm to the correct chest.

### Member 3 — Integration and evaluation
This module connects the OpenTheChests predictor with the KUKA controller and evaluates the full end-to-end system. Its role is to measure whether the system:
- predicts the correct chest
- reaches the correct physical target
- succeeds as a complete pipeline

## Repository structure

```text
.
├── README.md
├── requirements.txt
│
├── docs/
│   ├── README_openthechests.md
│   ├── README_kuka.md
│   ├── results_openthechests.md
│   └── integration_contract.md
│
├── data/
│   └── openthechests/
│       ├── processed/
│       ├── raw/
│       └── samples/
│
├── models/
│   └── openthechests/
│
├── results/
│   └── openthechests/
│
├── scripts/
│   └── dev/
│
└── src/
    ├── openthechests/
    ├── kuka/
    └── integration/
```

## Current status

At the moment, the repository already contains the **Member 1 OpenTheChests module** in a structured and reusable form.

This includes:
- environment registration
- oracle-labeled dataset generation
- rule-based, MLP, and LSTM baselines
- saved models
- a public inference API for the next team members

The KUKA and integration folders are already present so that Member 2 and Member 3 can work on top of this structure without modifying the internal logic of the OpenTheChests module.

## OpenTheChests module

The OpenTheChests module is the first completed layer of the project.

Its main public entry point is:

```python
from src.openthechests.predict import OpenTheChestsPredictor
```

Example usage:

```python
predictor = OpenTheChestsPredictor(
    model_dir="models/openthechests",
    mode="auto",
)

target_idx = predictor.predict_target(history, env_name="OpenTheChests-v2")
```

This returns:
- `0` for the red chest
- `1` for the green chest
- `2` for the blue chest
- `None` if no unique chest decision should be made yet

The current recommended model selection is:
- `OpenTheChests-v0` -> MLP
- `OpenTheChests-v1` -> MLP
- `OpenTheChests-v2` -> LSTM

## Expected integration flow

The intended project flow is:

1. run an `OpenTheChests` episode
2. collect the observation history
3. use the Member 1 predictor to infer a chest target
4. pass the resulting `target_idx` to the Member 2 KUKA controller
5. evaluate the complete end-to-end behavior

In simplified form:

```python
target_idx = predictor.predict_target(history, env_name)

if target_idx is not None:
    result = controller.run(target_idx)
```

## Documentation

The most relevant documents for the current stage are:

- `docs/README_openthechests.md`  
  Technical overview of the OpenTheChests module

- `docs/results_openthechests.md`  
  Experimental results for the OpenTheChests baselines

- `docs/README_kuka.md`  
  Technical overview of the Member 2 KUKA control module

- `docs/integration_contract.md`  
  Stable interface expected by Member 2 and Member 3

## Notes

This repository is intended to evolve in stages:

- first, Member 1 stabilizes the OpenTheChests layer
- then, Member 2 builds the KUKA layer on top of the same repository
- finally, Member 3 integrates both modules and evaluates the complete system

The goal is to keep each layer stable enough so that the next team member can work on top of it without rewriting the previous one.