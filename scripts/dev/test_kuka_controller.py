from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kuka import KukaChestController


def main() -> None:
    controller = KukaChestController(mode="heuristic")

    results = []
    for target_idx in [0, 1, 2]:
        result = controller.run(target_idx=target_idx, seed=42 + target_idx)
        results.append(result)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
