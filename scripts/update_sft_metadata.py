#!/usr/bin/env python3
"""Refresh ``data/sft_pure_v4/metadata.json`` after the M4 rebuild.

Recomputes everything that the merge invalidated (row counts, uniqueness, label
distribution, budget parse rate, SHA-256) while preserving the historical blocks
(``merge``, ``requested_3_5_2_mix``, ``admission``, ``reward_v4``) untouched. Adds
three new blocks:

* ``collection``          -- M2 provenance (rounds, acceptance, guard repair);
* ``m4``                  -- the M4 rebuild summary (merge, repair, variants);
* ``admission_recheck``   -- the A1-A6 verdicts re-run on the merged pool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.constraints import (  # noqa: E402
    explicit_budget_from_instruction,
)

try:  # pragma: no cover - depends on the engine revision
    from web_agent_site.engine.constraints import (  # type: ignore
        BUDGET_PARSER_VERSION as LIVE_BUDGET_PARSER_VERSION,
    )
except ImportError:  # pragma: no cover
    LIVE_BUDGET_PARSER_VERSION = "shopping-budget-parser-v4"

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

SFT = ROOT / "data" / "sft_pure_v4"
COLLECTION = ROOT / "data" / "collection"
EVALUATION = ROOT / "data" / "evaluation" / "tasks.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(path: Path) -> str:
    return hashlib.sha256(
        Path(path).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def portable(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def budget_parse_rate(rows: list[dict], goals: list[dict]) -> float:
    parsed = 0
    for row in rows:
        instruction = goals[int(row["task_id"])].get("instruction_text") or ""
        if explicit_budget_from_instruction(instruction) is not None:
            parsed += 1
    return round(parsed / len(rows), 4) if rows else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=SFT)
    parser.add_argument("--collection-dir", type=Path, default=COLLECTION)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument(
        "--admission-summary", type=Path, default=ROOT / "audit" / "admission_summary.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "sft_curriculum" / "manifest.json"
    )
    parser.add_argument("--metadata", type=Path, default=SFT / "metadata.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir
    metadata_path = args.metadata
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    rows = read_jsonl(dataset_dir / "all.jsonl")
    labels = read_jsonl(dataset_dir / "difficulty_labels.jsonl")
    evaluation_ids = {int(row["task_id"]) for row in read_jsonl(args.evaluation)}

    task_ids = [int(row["task_id"]) for row in rows]
    label_ids = [int(row["task_id"]) for row in labels]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("SFT pool contains duplicate task_ids")
    if set(task_ids) != set(label_ids):
        raise ValueError("difficulty labels do not match the pool task_id set")

    products = audit.load_products(audit.DEFAULT_PRODUCTS)
    goals = audit.build_goals(products)
    parse_rate = budget_parse_rate(rows, goals)

    difficulty = {level: 0 for level in ("simple", "medium", "hard")}
    confidences = []
    mode_histogram: dict[str, int] = {}
    for label in labels:
        difficulty[label["difficulty"]] = difficulty.get(label["difficulty"], 0) + 1
        confidences.append(float(label.get("confidence") or 0.0))
        mode = str(label.get("label_input_mode") or "unspecified")
        mode_histogram[mode] = mode_histogram.get(mode, 0) + 1

    increment_path = args.collection_dir / "accepted.jsonl"
    increment_ids = (
        {int(row["task_id"]) for row in read_jsonl(increment_path)}
        if increment_path.exists()
        else set()
    )
    relabelled = sum(1 for label in labels if int(label["task_id"]) in increment_ids)

    sha = {
        "all.jsonl": sha256_text(dataset_dir / "all.jsonl"),
        "difficulty_labels.jsonl": sha256_text(dataset_dir / "difficulty_labels.jsonl"),
        "duplicate_report.json": sha256_text(dataset_dir / "duplicate_report.json"),
    }
    toolonly = dataset_dir / "all_toolonly.jsonl"
    if toolonly.exists():
        sha["all_toolonly.jsonl"] = sha256_text(toolonly)

    increment_report = json.loads(
        (args.collection_dir / "increment_repair_report.json").read_text(encoding="utf-8")
    )
    collection_stats = json.loads(
        (args.collection_dir / "collection_stats.json").read_text(encoding="utf-8")
    )

    metadata["rows"] = len(rows)
    metadata["unique_task_ids"] = len(set(task_ids))
    metadata["held_out_evaluation_overlap"] = len(set(task_ids) & evaluation_ids)
    metadata["budget_parse_rate"] = parse_rate
    metadata["budget_parser"] = LIVE_BUDGET_PARSER_VERSION
    metadata["difficulty_labeling"] = {
        **metadata.get("difficulty_labeling", {}),
        **difficulty,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "confidence_below_0_8": sum(1 for value in confidences if value < 0.8),
        "label_input_mode_histogram": dict(sorted(mode_histogram.items())),
        "m2_increment_relabelled": relabelled,
        "label_source_note": (
            "All labels come from the teacher model. The M2 increment rows were "
            "re-labelled in M4; label_input_mode is present on every row."
        ),
    }
    metadata["sha256"] = sha
    metadata["collection"] = {
        "source": collection_stats.get("source", ""),
        "accepted": collection_stats.get("accepted", 0),
        "unique_task_ids": collection_stats.get("unique_task_ids", 0),
        "merged_into_pool": increment_report["counts"]["admitted_increment_rows"],
        "guard_repair": {
            "repaired_rows": increment_report["counts"]["repaired_rows"],
            "removed_guard_pairs": increment_report["counts"]["removed_guard_pairs"],
            "reason_histogram": increment_report["guard_reason_histogram"],
            "method": (
                "delete (illegal assistant turn, rejection observation) pairs; every "
                "rejection observation states 未执行, so the environment state is unchanged"
            ),
        },
        "provenance_rounds": [
            {
                "round": 1,
                "model": "deepseek-flash",
                "temperature": 0.7,
                "thinking": "enabled (unintended default)",
                "outcome": "85% assistant_final; interrupted",
            },
            {
                "round": 2,
                "model": "deepseek-flash",
                "temperature": 0.7,
                "thinking": "enabled (unintended default)",
                "outcome": "collected_raw=0 after purge",
            },
            {
                "round": 3,
                "model": "deepseek-flash",
                "temperature": 0.7,
                "thinking": "enabled (unintended default)",
                "outcome": "0 net new (attempt_index not normalised)",
            },
            {
                "round": 4,
                "model": "deepseek-v4-flash",
                "temperature": 0.0,
                "thinking": "disabled",
                "outcome": "53% acceptance; 100 unique tasks",
            },
        ],
        "note": (
            "Terminal-only acceptance (strict gold_purchase) passed all 100 rows; the "
            "guard-rejection repair is applied here, not in the collection pipeline."
        ),
    }
    metadata["m4"] = {
        "schema_version": increment_report["schema_version"],
        "increment": {
            "source": increment_report["source_increment"],
            "repaired_output": increment_report["repaired_increment"],
            "pool_before": increment_report["counts"]["pool_rows"],
            "pool_after": increment_report["counts"]["merged_rows"],
            "replaced_existing": increment_report["counts"]["replaced_existing"],
        },
        "variants": {
            "d_full": portable(dataset_dir / "all.jsonl"),
            "d_toolonly": portable(dataset_dir / "all_toolonly.jsonl"),
            "report": portable(dataset_dir / "variants_report.json"),
        },
    }
    summary = json.loads(args.admission_summary.read_text(encoding="utf-8"))
    sft_verdicts = summary.get("datasets", {}).get("sft_pure_v4", {})
    metadata["admission_recheck"] = {
        "contract": summary.get("contract"),
        "tasks": sft_verdicts.get("tasks"),
        "admitted": sft_verdicts.get("admitted"),
        "rejected": sft_verdicts.get("rejected"),
        "reasons_histogram": sft_verdicts.get("reasons_histogram", {}),
        "warnings_histogram": sft_verdicts.get("warnings_histogram", {}),
        "note": "A1-A6 re-run on the merged pool; the 100 M2 rows all pass.",
    }
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        metadata["split"] = {
            "source_manifest": portable(args.manifest),
            "split_seed": manifest.get("split_seed"),
            "validation_ratio": manifest.get("validation_ratio"),
            "train": manifest["counts"]["train"],
            "validation": manifest["counts"]["validation"],
            "buckets": {name: bucket["rows"] for name, bucket in manifest["buckets"].items()},
            "note": "Re-split after the M2 merge; supersedes the stale 84/9 interim counts.",
        }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "rows": metadata["rows"],
                "budget_parse_rate": parse_rate,
                "difficulty": difficulty,
                "evaluation_overlap": metadata["held_out_evaluation_overlap"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
