# Open The Chests Project

This repository contains the project structure for **Open The Chests** group E, whose objective is to build a system that can **recognize activities from event observations** and then **react physically by moving a KUKA robot arm to the correct chest**. The project is naturally divided into an activity-recognition layer, a robot-control layer, and a final integration layer.

## Project overview

The full system is organized around three components:

- an **OpenTheChests decision module**, which observes event histories and predicts which chest should be opened
- a **KUKA control module**, which receives a chest target and moves the robot arm to the corresponding chest
- an **integration layer**, which connects both modules and evaluates the full pipeline end-to-end

This repository is therefore structured so that each team member can work on one part without breaking the others.

## Team structure

### OpenTheChests activity recognition
This module is responsible for the activity-recognition and decision-making side of the project. It works on the custom environments:

- `OpenTheChests-v0`
- `OpenTheChests-v1`
- `OpenTheChests-v2`

These correspond to increasing difficulty levels. The goal of this module is to predict the appropriate chest-opening decision from the observation history.

### KUKA control
This module is responsible for the physical control side. It uses the KUKA PyBullet environment and receives a chest target index:

- `0 = red`
- `1 = green`
- `2 = blue`

It must then move the robot arm to the correct chest.

### Integration and evaluation
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
