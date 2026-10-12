#!/usr/bin/env python3
"""Apply the ``shopping-task-admission-v1`` contract to every task pool.

Writes (deterministically, no timestamps):

* ``audit/admission_verdicts.jsonl`` -- one row per task, all four datasets;
* ``audit/admission_summary.json``   -- per-dataset admission summary.

The contract is read-only: this script never rewrites ``data/``. The GRPO pool
is consumed by training as parquet, and this environment has neither the parquet
toolchain nor the in-repo pool builder, so swapping the pools is a separate,
explicit step (see ``audit/fix_plan.md``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.admission import (  # noqa: E402
    CONTRACT_VERSION,
    admit_task,
)

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=audit.DEFAULT_PRODUCTS)
    parser.add_argument("--out", type=Path, default=ROOT / "audit")
    args = parser.parse_args()

    products = audit.load_products(args.products)
    goals = audit.build_goals(products)
    product_by_asin = {str(product["asin"]): product for product in products}

    dataset_ids = {
        name: audit.task_ids(path)
        for name, path in audit.DEFAULT_DATASETS.items()
        if path.exists()
    }
    evaluation_ids = set(dataset_ids.get("evaluation", ()))
    sft_ids = set(dataset_ids.get("sft_pure_v4", ()))
    grpo_ids = set(dataset_ids.get("grpo_train", ())) | set(
        dataset_ids.get("grpo_validation", ())
    )

    held_out_by_dataset = {
        "evaluation": set(),
        "sft_pure_v4": evaluation_ids | grpo_ids,
        "grpo_train": evaluation_ids | sft_ids,
        "grpo_validation": evaluation_ids | sft_ids,
    }

    verdicts: list[dict] = []
    summary: dict[str, dict] = {}
    for name, ids in dataset_ids.items():
        held_out = held_out_by_dataset.get(name, set())
        reason_counts: Counter = Counter()
        warning_counts: Counter = Counter()
        rejected: list[int] = []
        for task_id in ids:
            goal = goals[task_id]
            product = product_by_asin[str(goal["asin"])]
            verdict = admit_task(task_id, goal, product, held_out_task_ids=held_out)
            verdict["dataset"] = name
            verdicts.append(verdict)
            if not verdict["admitted"]:
                rejected.append(task_id)
                reason_counts.update(verdict["reasons"])
            warning_counts.update(verdict["warnings"])
        summary[name] = {
            "tasks": len(ids),
            "admitted": len(ids) - len(rejected),
            "rejected": len(rejected),
            "rejected_task_ids": sorted(rejected),
            "reasons_histogram": dict(sorted(reason_counts.items())),
            "warnings_histogram": dict(sorted(warning_counts.items())),
        }

    verdicts.sort(key=lambda row: (row["dataset"], row["task_id"]))
    payload = {
        "contract": CONTRACT_VERSION,
        "datasets": summary,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "admission_verdicts.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in verdicts
        ),
        encoding="utf-8",
    )
    (out / "admission_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
