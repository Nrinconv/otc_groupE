# Results - OpenTheChests Module

## Experimental setup

The final experiments were conducted on the oracle-labeled datasets generated from the `OpenTheChests` environments.

Large datasets used for the final comparison:
- `oracle_dataset_v0_large.jsonl` with 350 samples
- `oracle_dataset_v1_large.jsonl` with 1314 samples
- `oracle_dataset_v2_large.jsonl` with 5126 samples

Train/test splits were performed **by seed**, not by individual sample, in order to avoid leakage between highly similar states from the same episode.

The evaluated models were:
- a simple rule-based baseline
- a feedforward MLP using the current observation only
- an LSTM using prefixes of observation sequences

## Final comparison

| Model | v0 Accuracy | v0 Macro F1 | v1 Accuracy | v1 Macro F1 | v2 Accuracy | v2 Macro F1 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| Rule | 1.0000 | - | 0.4030 | - | 0.6420 | - |
| MLP  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9698 | 0.7668 |
| LSTM | 0.9600 | 0.9539 | 0.9884 | 0.9625 | 0.9844 | 0.8991 |

## Main observations

### Easy environment `v0`

The rule-based baseline already solves `v0` perfectly. This indicates that the easy environment can be handled directly from the current observation, without requiring a learned temporal model. The MLP also reaches perfect performance, while the LSTM performs slightly worse. This suggests that adding recurrence in this setting is unnecessary and even slightly less efficient than a simple feedforward approach.

### Medium environment `v1`

The rule-based baseline drops sharply on `v1`, which shows that a hand-crafted mapping from event type to action is no longer sufficient. However, the MLP reaches perfect performance on the large dataset, which means that the current observation already contains enough useful information to solve this environment. The LSTM also performs very strongly, but it does not improve over the MLP. This suggests that explicit temporal memory is not essential for `v1`.

### Hard environment `v2`

The hard environment is the only setting where the LSTM brings a clear benefit. The MLP already achieves strong overall accuracy, but its Macro F1 remains limited because the dataset is highly imbalanced and minority actions are harder to predict. The LSTM improves both overall performance and class balance, raising Macro F1 from `0.7668` to `0.8991`. This indicates that temporal context becomes genuinely useful in the hardest environment.

## Interpretation

The experimental results support the following conclusion.

A simple rule-based approach is only sufficient for the easiest setting. A learned feedforward model using the current observation alone is already enough to solve `v0` and `v1` and performs strongly on `v2`. However, sequence modeling becomes clearly beneficial in `v2`, where the LSTM improves over the MLP, especially in Macro F1 and minority-class behavior.

In other words, temporal memory is not uniformly necessary across all environments, but it does become useful when the task structure is more complex.

## Recommended final models

Based on the final results, the recommended model selection is:

- `OpenTheChests-v0` -> **MLP**
- `OpenTheChests-v1` -> **MLP**
- `OpenTheChests-v2` -> **LSTM**

This is the policy currently implemented in the inference API through `mode="auto"`.

## Current best model by environment

| Environment | Recommended model | Reason |
|---|---|---|
| OpenTheChests-v0 | MLP | Perfect performance, simpler than LSTM |
| OpenTheChests-v1 | MLP | Perfect performance with current observation only |
| OpenTheChests-v2 | LSTM | Best overall result, especially in Macro F1 |

## Final takeaway

The `OpenTheChests` task should not be described as a problem where recurrent models are always necessary. Instead, the results show a more nuanced picture.

For the easy and medium environments, the current observation is already sufficient and a simple feedforward model is the best practical choice. For the hardest environment, however, the LSTM provides a real improvement, which supports the idea that temporal dependencies matter when the event structure becomes more complex.