# Results - KUKA Module

## Experimental setup

The final KUKA experiments were conducted on the custom PyBullet environment `ColoredChestKuka-v0`.

The task is a continuous-control reaching problem. At each episode, the controller receives a chest target index and must move the KUKA end effector to the top center of the corresponding chest.

The target convention used in all experiments is:

- `0 = red`
- `1 = green`
- `2 = blue`

The final environment configuration used for the reported PPO result includes:

- `reward_type="advanced"`
- relative target features in the observation
- multiple simulation steps per action
- progress-based reward shaping and a strong success bonus
- `success_distance = 0.06`
- `success_hold_steps = 5`

The final learned controller was trained with:

- PPO
- `VecNormalize`
- `use_sde=True`
- `total_timesteps = 1_000_000`

Evaluation was performed through `scripts/eval_kuka_controller.py`.

## Final comparison

| Controller | Success Rate | Mean Steps | Mean Final Distance |
|---|---:|---:|---:|
| Heuristic | 1.0000 | 69.43 | 0.0280 |
| PPO | 0.8333 | 73.77 | 0.0450 |

## Main observations

### Heuristic controller

The proportional controller solves the KUKA reaching task perfectly in the tested evaluation setup. This confirms that the target convention is correct and that the environment is physically controllable with a simple Cartesian strategy. The heuristic controller is therefore an excellent baseline for integration and debugging.

### PPO controller

The final PPO controller achieves a success rate of `0.8333`, with a mean final distance of approximately `0.0450`, which is inside the general range required for success. This is a strong improvement over the earlier PPO iterations, which were either unstable or remained close to the target without consistently satisfying the success hold condition.

The final improvements came from three main changes:

- enriching the observation with relative target information
- improving the reward shaping near the goal region
- increasing the physical stability of each action through multiple simulation steps

## Interpretation

The final experiments support the following conclusion.

A deterministic proportional controller remains the strongest practical controller for this specific reaching task because the environment directly exposes geometric target information. However, PPO can still learn a useful and reasonably robust reaching policy when the environment is shaped appropriately and when training is done with normalization and sufficient budget.

This means that the KUKA module now contains both a strong engineering baseline and a learned controller that is good enough to be reported and compared.

## Recommended final usage

Based on the final results, the recommended usage is:

- use **heuristic** when the priority is robust end-to-end integration
- use **PPO** when the priority is to demonstrate a learned continuous-control solution

## Current best controller by objective

| Objective | Recommended controller | Reason |
|---|---|---|
| Robust integration | Heuristic | Perfect success rate and simplest behavior |
| Learned control experiment | PPO | Strong enough result while remaining fully learned |

## Final takeaway

The KUKA task should not be described as a setting where reinforcement learning is the only valid solution. In this project, the geometric structure of the observation makes a proportional controller extremely effective. However, PPO still reaches a strong level of performance after the environment and training pipeline are refined.

The most useful final position is therefore not to replace the heuristic controller, but to keep both controllers for different roles in the project.
