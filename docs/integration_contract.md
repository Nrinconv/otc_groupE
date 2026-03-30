# Integration Contract

## Purpose

This module implements the **OpenTheChests decision layer** of the project.

Its role is to process the observation history from an `OpenTheChests` environment and predict either:

- a chest-opening action as a 3-bit vector
- or a unique chest target index compatible with the KUKA module

This corresponds to the activity-recognition side of the project.

---

## Public API

The main public entry point is:

```python
from src.openthechests.predict import OpenTheChestsPredictor