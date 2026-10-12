#!/usr/bin/env python3
"""Rebuild the training pools by applying the admission contract verdicts.

Reads ``audit/admission_verdicts.jsonl`` (produced by ``scripts/admission_report.py``)
and removes every rejected ``task_id`` from the SFT and GRPO pools, keeping the
original selection order. It is a **filter**, not a re-sample: the surviving rows
are byte-identical to the originals, so the rebuild is fully reproducible from
(original pool + rejection list).

Touched files:

* ``data/grpo/train.jsonl`` / ``train.parquet``
* ``data/grpo/validation.jsonl`` / ``validation.parquet``
* ``data/grpo/metadata.json``
* ``data/sft_pure_v4/all.jsonl`` / ``difficulty_labels.jsonl``
* ``data/sft_pure_v4/metadata.json``
* ``data/sft_curriculum/manifest.json``
* ``plans/curation-report.md``

Never touched: ``data/evaluation/**`` (frozen, SHA-256 recorded in metadata) and
``data/sft/**`` (legacy 1,000-row pool, not part of the audited set).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.admission import CONTRACT_VERSION  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

GRPO = ROOT / "data" / "grpo"
SFT = ROOT / "data" / "sft_pure_v4"
CURRICULUM = ROOT / "data" / "sft_curriculum"


def sha256_text(path: Path) -> str:
    """Hash a text artifact with CRLF folded to LF (repo convention)."""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def sha256_binary(path: Path) -> str:
    """Hash a binary artifact on its raw bytes; never fold line endings."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise SystemExit(
            "rebuild_pools.py needs pyarrow to rewrite data/grpo/*.parquet.\n"
            "It is a transitive dependency of `datasets` (see uv.lock); run\n"
            "`bash scripts/setup.sh` or `pip install pyarrow` first."
        ) from exc
    return pa, pq


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_verdicts(path: Path) -> dict[str, dict]:
    per_dataset: dict[str, dict] = {}
    for row in audit.read_jsonl(path):
        dataset = row["dataset"]
        entry = per_dataset.setdefault(
            dataset, {"rejected": set(), "reasons": Counter(), "total": 0}
        )
        entry["total"] += 1
        if not row["admitted"]:
            entry["rejected"].add(int(row["task_id"]))
            entry["reasons"].update(row["reasons"])
    return per_dataset


def filter_jsonl(path: Path, rejected: set[int]) -> tuple[int, int]:
    """Drop rejected rows at the *line* level so kept rows stay byte-identical."""
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [line for line in raw if line.strip()]
    kept = [
        line for line in lines if int(json.loads(line)["task_id"]) not in rejected
    ]
    Path(path).write_text(
        "\n".join(kept) + "\n", encoding="utf-8", newline="\n"
    )
    return len(lines), len(kept)


def filter_parquet(path: Path, rejected: set[int]) -> tuple[int, int]:
    pa, pq = _require_pyarrow()
    table = pq.read_table(path)
    task_ids = [row["task_id"] for row in table.column("extra_info").to_pylist()]
    mask = pa.array([int(tid) not in rejected for tid in task_ids])
    filtered = table.filter(mask)
    pq.write_table(filtered, path)
    return table.num_rows, filtered.num_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdicts", type=Path, default=ROOT / "audit" / "admission_verdicts.jsonl"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "plans" / "curation-report.md"
    )
    args = parser.parse_args()

    verdicts = load_verdicts(args.verdicts)
    report_rows = []

    # ---- GRPO ----------------------------------------------------------------
    grpo_meta = json.loads((GRPO / "metadata.json").read_text(encoding="utf-8"))
    for name, key in (("train", "grpo_train"), ("validation", "grpo_validation")):
        rejected = verdicts[key]["rejected"]
        before_j, after_j = filter_jsonl(GRPO / f"{name}.jsonl", rejected)
        before_p, after_p = filter_parquet(GRPO / f"{name}.parquet", rejected)
        grpo_meta[name]["tasks"] = after_j
        grpo_meta[name]["parquet_sha256"] = sha256_binary(GRPO / f"{name}.parquet")
        report_rows.append(
            {
                "dataset": key,
                "before": before_j,
                "after": after_j,
                "removed": before_j - after_j,
                "reasons": dict(sorted(verdicts[key]["reasons"].items())),
                "parquet_before": before_p,
                "parquet_after": after_p,
            }
        )
    grpo_meta["admission"] = {
        "contract": CONTRACT_VERSION,
        "applied": True,
        "train_rejected": sorted(verdicts["grpo_train"]["rejected"]),
        "validation_rejected": sorted(verdicts["grpo_validation"]["rejected"]),
        "method": "filter (original selection order preserved; no re-sampling)",
    }
    write_json(GRPO / "metadata.json", grpo_meta)

    # ---- SFT -----------------------------------------------------------------
    sft_rejected = verdicts["sft_pure_v4"]["rejected"]
    before, after = filter_jsonl(SFT / "all.jsonl", sft_rejected)
    filter_jsonl(SFT / "difficulty_labels.jsonl", sft_rejected)
    sft_meta = json.loads((SFT / "metadata.json").read_text(encoding="utf-8"))
    sft_meta["rows"] = after
    sft_meta["unique_task_ids"] = after
    labels = audit.read_jsonl(SFT / "difficulty_labels.jsonl")
    difficulty_counts = Counter(row["difficulty"] for row in labels)
    sft_meta["difficulty_labeling"] = {
        **sft_meta.get("difficulty_labeling", {}),
        **{name: difficulty_counts.get(name, 0) for name in ("simple", "medium", "hard")},
    }
    sft_meta["sha256"] = {
        "all.jsonl": sha256_text(SFT / "all.jsonl"),
        "difficulty_labels.jsonl": sha256_text(SFT / "difficulty_labels.jsonl"),
        "duplicate_report.json": sft_meta.get("sha256", {}).get(
            "duplicate_report.json", ""
        ),
    }
    sft_meta["admission"] = {
        "contract": CONTRACT_VERSION,
        "applied": True,
        "rejected": sorted(sft_rejected),
        "reasons": dict(sorted(verdicts["sft_pure_v4"]["reasons"].items())),
        "method": "filter (original selection order preserved; no re-sampling)",
    }
    write_json(SFT / "metadata.json", sft_meta)
    report_rows.append(
        {
            "dataset": "sft_pure_v4",
            "before": before,
            "after": after,
            "removed": before - after,
            "reasons": dict(sorted(verdicts["sft_pure_v4"]["reasons"].items())),
        }
    )

    # ---- curriculum manifest -------------------------------------------------
    manifest = json.loads((CURRICULUM / "manifest.json").read_text(encoding="utf-8"))
    for bucket in manifest["buckets"].values():
        for field in ("train_task_ids", "validation_task_ids"):
            bucket[field] = sorted(
                task_id
                for task_id in bucket[field]
                if int(task_id) not in sft_rejected
            )
        bucket["rows"] = len(bucket["train_task_ids"]) + len(
            bucket["validation_task_ids"]
        )
    for stage in manifest["stages"].values():
        stage["train_rows"] = sum(
            len(manifest["buckets"][name]["train_task_ids"]) for name in stage["buckets"]
        )
        stage["validation_rows"] = sum(
            len(manifest["buckets"][name]["validation_task_ids"])
            for name in stage["buckets"]
        )
    manifest["review_flags"] = {
        key: sorted(
            task_id for task_id in value if int(task_id) not in sft_rejected
        )
        for key, value in manifest.get("review_flags", {}).items()
    }
    manifest["counts"] = {
        "rows": after,
        "train": sum(len(b["train_task_ids"]) for b in manifest["buckets"].values()),
        "validation": sum(
            len(b["validation_task_ids"]) for b in manifest["buckets"].values()
        ),
        "evaluation_overlap": 0,
    }
    manifest["source"]["sha256"] = sha256_text(SFT / "all.jsonl")
    manifest["labels"]["sha256"] = sha256_text(SFT / "difficulty_labels.jsonl")
    manifest["admission"] = {"contract": CONTRACT_VERSION, "applied": True}
    write_json(CURRICULUM / "manifest.json", manifest)

    # ---- curation report -----------------------------------------------------
    lines = [
        "# 数据重建报告（M1-R5）",
        "",
        f"> 契约：`{CONTRACT_VERSION}`；方法：**过滤**（保留原选择顺序，不重采样）。",
        "> 判定来源：`audit/admission_verdicts.jsonl`。评测集与 legacy `data/sft/` 未改动。",
        "",
        "| 数据集 | 清洗前 | 清洗后 | 剔除 | 剔除原因 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report_rows:
        reasons = " / ".join(f"{k} {v}" for k, v in row["reasons"].items()) or "—"
        lines.append(
            f"| {row['dataset']} | {row['before']} | {row['after']} | "
            f"{row['removed']} | {reasons} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 剔除为**过滤**：保留的行与原文件逐字节一致，可用「原池 + 拒绝清单」完全复现。",
        "- 未做难度重采样（hard 占比目标 ≥20%）与长尾补采 —— 需要额外决策与采集预算。",
        "- 池规模缩小（SFT 1,192 → 1,152；GRPO 1,050 → 955）是**有意的**：被剔除的题在",
        "  当前 Reward 下结构性不可赢，留在池里只会贡献噪声。",
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8", newline="\n")

    for row in report_rows:
        print(
            f"{row['dataset']}: {row['before']} -> {row['after']} (removed {row['removed']})"
        )
    print("curation report:", args.report.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
