#!/usr/bin/env python3
"""Pre-flight smoke test: can the rebuilt data be consumed by the pipelines?

Validates the **data contracts** that ``train_lora_sft.py``, veRL and the
curriculum runner rely on, without needing torch / verl / a GPU. This is the
check required by rule P8 ("一次改到可直接开跑") before spending any training
budget.

Exit code 0 = all checks passed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.sft.dataset import (  # noqa: E402
    normalize_messages_for_chat_template,
)

SFT = ROOT / "data" / "sft_pure_v4"
GRPO = ROOT / "data" / "grpo"
CURRICULUM = ROOT / "data" / "sft_curriculum"
EVALUATION = ROOT / "data" / "evaluation"

PARQUET_SCHEMA = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
SFT_ROW_KEYS = {"trajectory_id", "task_id", "messages", "tools"}

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((bool(ok), name, detail))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(path: Path) -> str:
    return hashlib.sha256(
        Path(path).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def sha256_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
def check_evaluation_untouched() -> None:
    metadata = json.loads((EVALUATION / "metadata.json").read_text(encoding="utf-8"))
    actual = sha256_text(EVALUATION / "tasks.jsonl")
    check(
        actual == metadata["sha256"],
        "评测集未被改动（SHA-256 与 metadata 一致）",
        f"{actual[:16]}… vs {metadata['sha256'][:16]}…",
    )


def check_sft() -> list[dict]:
    rows = read_jsonl(SFT / "all.jsonl")
    check(bool(rows), "SFT all.jsonl 非空", f"{len(rows)} 行")

    bad_keys = [r.get("task_id") for r in rows if set(r) != SFT_ROW_KEYS]
    check(not bad_keys, "SFT 每行字段集合正确", f"异常 {len(bad_keys)} 行")

    ids = [int(r["task_id"]) for r in rows]
    check(len(ids) == len(set(ids)), "SFT task_id 唯一", f"{len(set(ids))}/{len(ids)}")

    unnormalizable = []
    no_assistant = []
    bad_calls = []
    for row in rows:
        tid = int(row["task_id"])
        if normalize_messages_for_chat_template(row["messages"]) is None:
            unnormalizable.append(tid)
            continue
        assistants = [m for m in row["messages"] if m.get("role") == "assistant"]
        if not assistants:
            no_assistant.append(tid)
            continue
        for message in assistants:
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                if (
                    call.get("type") != "function"
                    or not call.get("id")
                    or not fn.get("name")
                    or not isinstance(fn.get("arguments"), str)
                ):
                    bad_calls.append(tid)
                    break
    check(not unnormalizable, "SFT messages 可被 chat template 归一化", f"异常 {unnormalizable[:5]}")
    check(not no_assistant, "SFT 每条轨迹至少 1 个 assistant 回合", f"异常 {no_assistant[:5]}")
    check(not bad_calls, "SFT tool_calls 结构合法", f"异常 {bad_calls[:5]}")

    labels = read_jsonl(SFT / "difficulty_labels.jsonl")
    label_ids = {int(r["task_id"]) for r in labels}
    check(
        label_ids == set(ids),
        "difficulty_labels 与 all.jsonl 的 task_id 集合一致",
        f"{len(label_ids)} vs {len(set(ids))}",
    )

    metadata = json.loads((SFT / "metadata.json").read_text(encoding="utf-8"))
    check(
        metadata.get("rows") == len(rows),
        "SFT metadata.rows 与实际行数一致",
        f"{metadata.get('rows')} vs {len(rows)}",
    )
    check(
        metadata.get("sha256", {}).get("all.jsonl") == sha256_text(SFT / "all.jsonl"),
        "SFT metadata.all.jsonl 哈希一致",
    )
    check(
        metadata.get("sha256", {}).get("difficulty_labels.jsonl")
        == sha256_text(SFT / "difficulty_labels.jsonl"),
        "SFT metadata.difficulty_labels 哈希一致",
    )
    return rows


def check_curriculum(sft_rows: list[dict]) -> None:
    manifest = json.loads((CURRICULUM / "manifest.json").read_text(encoding="utf-8"))
    known = {int(r["task_id"]) for r in sft_rows}
    buckets = manifest["buckets"]

    unknown = sorted(
        {
            int(t)
            for b in buckets.values()
            for field in ("train_task_ids", "validation_task_ids")
            for t in b[field]
        }
        - known
    )
    check(not unknown, "课程 buckets 的 task_id 全部存在于 all.jsonl", f"未知 {unknown[:5]}")

    overlap = sorted(
        {
            int(t)
            for b in buckets.values()
            for t in set(b["train_task_ids"]) & set(b["validation_task_ids"])
        }
    )
    check(not overlap, "课程 buckets 内 train/validation 无交集", f"{overlap[:5]}")

    train_total = sum(len(b["train_task_ids"]) for b in buckets.values())
    valid_total = sum(len(b["validation_task_ids"]) for b in buckets.values())
    counts = manifest["counts"]
    check(
        counts["rows"] == len(sft_rows)
        and counts["train"] == train_total
        and counts["validation"] == valid_total,
        "课程 counts 与 buckets 一致",
        f"rows={counts['rows']} train={counts['train']} val={counts['validation']}",
    )

    stage_bad = []
    for name, stage in manifest["stages"].items():
        expect_train = sum(len(buckets[b]["train_task_ids"]) for b in stage["buckets"])
        expect_valid = sum(
            len(buckets[b]["validation_task_ids"]) for b in stage["buckets"]
        )
        if (stage["train_rows"], stage["validation_rows"]) != (expect_train, expect_valid):
            stage_bad.append(name)
    check(not stage_bad, "课程 stages 行数与 buckets 累加一致", f"异常 {stage_bad}")

    check(
        manifest["source"]["sha256"] == sha256_text(SFT / "all.jsonl"),
        "课程 manifest.source 哈希一致",
    )
    check(
        manifest["labels"]["sha256"] == sha256_text(SFT / "difficulty_labels.jsonl"),
        "课程 manifest.labels 哈希一致",
    )


def check_grpo() -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        check(False, "pyarrow 可用（veRL parquet 校验）", "未安装，跳过 parquet 校验")
        return

    metadata = json.loads((GRPO / "metadata.json").read_text(encoding="utf-8"))
    for name in ("train", "validation"):
        jsonl_ids = [int(r["task_id"]) for r in read_jsonl(GRPO / f"{name}.jsonl")]
        table = pq.read_table(GRPO / f"{name}.parquet")
        check(
            table.column_names == PARQUET_SCHEMA,
            f"grpo/{name}.parquet schema 正确",
            str(table.column_names),
        )
        extra = table.column("extra_info").to_pylist()
        bad = [i for i, row in enumerate(extra) if not isinstance(row, dict) or "task_id" not in row]
        check(not bad, f"grpo/{name} extra_info 均含 task_id", f"异常 {bad[:5]}")
        parquet_ids = [int(row["task_id"]) for row in extra]
        check(
            parquet_ids == jsonl_ids,
            f"grpo/{name} parquet 与 jsonl task_id 逐行一致",
            f"{len(parquet_ids)} vs {len(jsonl_ids)}",
        )
        check(
            metadata[name]["tasks"] == len(jsonl_ids),
            f"grpo metadata.{name}.tasks 一致",
            f"{metadata[name]['tasks']} vs {len(jsonl_ids)}",
        )
        check(
            metadata[name]["parquet_sha256"] == sha256_raw(GRPO / f"{name}.parquet"),
            f"grpo/{name}.parquet 原始字节哈希一致",
        )
        prompt_ok = all(
            isinstance(p, list) and p and p[0].get("role") == "system"
            for p in table.column("prompt").to_pylist()
        )
        check(prompt_ok, f"grpo/{name} prompt 均为非空 system-first 消息列表")


def check_pools_disjoint() -> None:
    ids = {}
    for name, path in (
        ("evaluation", EVALUATION / "tasks.jsonl"),
        ("sft_pure_v4", SFT / "all.jsonl"),
        ("grpo_train", GRPO / "train.jsonl"),
        ("grpo_validation", GRPO / "validation.jsonl"),
    ):
        ids[name] = {int(r["task_id"]) for r in read_jsonl(path)}
    import itertools

    bad = []
    for left, right in itertools.combinations(sorted(ids), 2):
        if ids[left] & ids[right]:
            bad.append(f"{left}&{right}")
    check(not bad, "四池两两零重叠", str(bad))


def main() -> int:
    check_evaluation_untouched()
    sft_rows = check_sft()
    check_curriculum(sft_rows)
    check_grpo()
    check_pools_disjoint()

    failed = [r for r in results if not r[0]]
    for ok, name, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFAILED:")
        for _, name, detail in failed:
            print(f"  - {name}  {detail}")
        return 1
    print("数据契约冒烟通过：可以开跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
