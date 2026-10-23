#!/usr/bin/env python3
"""Validate Pure V4 and build the fixed cumulative SFT curriculum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SCHEMA_VERSION = "shopping-sft-curriculum-v1"
STAGE_CONFIG = {
    "a": {"buckets": ["foundation"], "epochs": 1.0, "learning_rate": 1e-4},
    "b": {
        "buckets": ["foundation", "constraints"],
        "epochs": 1.0,
        "learning_rate": 7e-5,
    },
    "c": {
        "buckets": ["foundation", "constraints", "strategy"],
        "epochs": 1.0,
        "learning_rate": 5e-5,
    },
}
# 「倒序课程」（2026-10-22 起，M10-R 用）：先集中学最稀缺、最想改变的行为，
# 再扩展到全分布。和上面那套「由易到难」的区别不是顺序好看，而是曝光比：
#   由易到难：模板被练 3 遍、探索只练 1 遍（曝光比 0.19）——与目标相反；
#   倒序：探索 2 遍、模板 1 遍（曝光比 ~1.45）——把稀缺行为抬到多数。
BEHAVIOR_STAGE_CONFIG = {
    "a": {"buckets": ["exploratory"], "epochs": 1.0, "learning_rate": 1e-4},
    "b": {
        "buckets": ["exploratory", "template", "other"],
        "epochs": 1.0,
        "learning_rate": 5e-5,
    },
}
BUCKET_MODES = ("difficulty", "behavior")
# 「单阶段」（2026-10-23 起，M10-R v2）：不做课程，在全量分布上直接训 1 个 epoch。
# 为什么放弃两段：A 段单独评测证明了两件事——
#   ① 模型**忠实复现数据里的行为比例**：A 段数据翻页占 28%，输出就是 31%；
#   ② B 段从「翻页 30%」出发、用 11.9% 的全量池训 137 步，被反向修正**过冲到 1%**。
# 所以正确做法是「从 0% 出发 + 数据比例本身就是目标比例」，只有一个阶段才不会过冲。
# lr 与旧课程末段一致（5e-5），保证与 M5-B / M10-R 可比。
SINGLE_STAGE_CONFIG = {
    "difficulty": {
        "full": {
            "buckets": ["foundation", "constraints", "strategy"],
            "epochs": 1.0,
            "learning_rate": 5e-5,
        }
    },
    "behavior": {
        "full": {
            "buckets": ["exploratory", "template", "other"],
            "epochs": 1.0,
            "learning_rate": 5e-5,
        }
    },
}
STAGE_PLANS = ("reverse", "single")
# 只有 next_page 算「翻页探索」。**prev_page 不算**——它是「从商品详情退回列表」
# 的常规导航，池子里出现 2,032 次，把它算成探索会让 1,098 条模板轨迹被误判成探索
# （2026-10-22 实测：误判后模板只剩 154 条，而真实是 1,108 条）。
PAGING_TOOLS = frozenset({"next_page"})
GUARD_MARKERS = ("guard rejected", "action guard", "动作守卫拒绝", "guard_violation")
ASIN_PATTERN = re.compile(r"(?:^|\n)asin:\s*([^\n]+)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tool_names(row: dict) -> set[str]:
    return {
        str((tool.get("function") or {}).get("name") or "")
        for tool in row.get("tools") or []
    }


def _validate_row(row: dict) -> list[tuple[str, str]]:
    task_id = int(row["task_id"])
    known_tools = _tool_names(row)
    calls = []
    terminal_call_id = None
    for index, message in enumerate(row.get("messages") or []):
        if message.get("role") == "tool":
            folded = str(message.get("content") or "").casefold()
            if any(marker in folded for marker in GUARD_MARKERS):
                raise ValueError(f"task {task_id}: guard rejection at message {index}")
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            raise ValueError(f"task {task_id}: multiple tool calls at message {index}")
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"task {task_id}: invalid tool arguments") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"task {task_id}: tool arguments must be an object")
            if name not in known_tools:
                raise ValueError(f"task {task_id}: unknown tool {name!r}")
            signature = (name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
            calls.append(signature)
            if name == "buy_now":
                terminal_call_id = call.get("id")

    if sum(name == "buy_now" for name, _ in calls) != 1 or not calls or calls[-1][0] != "buy_now":
        raise ValueError(f"task {task_id}: trajectory must end with exactly one buy_now")
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "tool":
        raise ValueError(f"task {task_id}: missing terminal tool response")
    if messages[-1].get("tool_call_id") != terminal_call_id:
        raise ValueError(f"task {task_id}: terminal tool response mismatch")
    terminal_content = str(messages[-1].get("content") or "").strip()
    if terminal_content != "购买已完成。" and not re.search(
        r"(?:^|\n)page_type:\s*terminal(?:\n|$)", terminal_content
    ):
        raise ValueError(f"task {task_id}: terminal reward text was not sanitized")
    return calls


def _bucket(label: dict) -> str:
    if label["difficulty"] == "simple":
        return "foundation"
    if (
        label["difficulty"] == "medium"
        and not label["needs_query_rewrite"]
        and not label["needs_candidate_comparison"]
    ):
        return "constraints"
    return "strategy"


def behavior_profile(row: dict) -> dict:
    """一条轨迹的「行为画像」：搜了几次、换没换词、开几个、翻没翻页。

    与 ``scripts/build_m10r_pool.describe_row`` 的口径一致（探索 = 翻页 / 换词 /
    开≥2 个），但**不依赖工具返回文本**：这里直接读 tool call 的参数，
    因为课程分桶只需要「做了什么」，不需要「看到了什么」。
    """
    queries: list[str] = []
    opened: list[str] = []
    paging = False
    steps = 0
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            steps += 1
            if name == "search_products":
                queries.append(" ".join(str(arguments.get("query") or "").split()).casefold())
            elif name == "open_product":
                opened.append(str(arguments.get("asin") or ""))
            elif name in PAGING_TOOLS:
                paging = True
    distinct_queries = len({query for query in queries if query})
    distinct_opens = len({asin for asin in opened if asin})
    return {
        "steps": steps,
        "searches": len(queries),
        "distinct_queries": distinct_queries,
        "opens": len(opened),
        "distinct_opens": distinct_opens,
        "paging": paging,
        # 与 build_m10r_pool.describe_row 同一个定义：三个动作任一出现就算探索过。
        "explored": bool(paging or distinct_opens >= 2 or distinct_queries >= 2),
    }


def behavior_bucket(profile: dict) -> str:
    """行为分桶：探索 / 模板（单搜单开）/ 其他。

    「模板」= 没做过任何探索动作，且只搜一次、只开一个。这类轨迹教的是
    「看到第一个就买」，正是要**减少**曝光的那一类；旧课程把它练了三遍。
    """
    if profile["explored"]:
        return "exploratory"
    if profile["searches"] <= 1 and profile["opens"] <= 1:
        return "template"
    return "other"


def _split(ids: list[int], *, seed: int, validation_ratio: float) -> tuple[list[int], list[int]]:
    ordered = sorted(
        ids,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )
    validation_count = int(len(ordered) * validation_ratio + 0.5)
    validation = sorted(ordered[:validation_count])
    train = sorted(ordered[validation_count:])
    return train, validation


def stage_config_for(bucket_mode: str, stage_plan: str = "reverse") -> dict:
    """按「分桶口径 × 阶段方案」取阶段配置。

    ``reverse`` = 倒序两段（先探索后全量，M10-R v1，已证伪）；
    ``single``  = 单阶段全量（M10-R v2）。
    """
    if bucket_mode not in BUCKET_MODES:
        raise ValueError(f"bucket_mode must be one of {BUCKET_MODES}")
    if stage_plan not in STAGE_PLANS:
        raise ValueError(f"stage_plan must be one of {STAGE_PLANS}")
    if stage_plan == "single":
        return SINGLE_STAGE_CONFIG[bucket_mode]
    return STAGE_CONFIG if bucket_mode == "difficulty" else BEHAVIOR_STAGE_CONFIG


def build_manifest(
    rows: list[dict],
    labels: list[dict],
    *,
    evaluation_ids: set[int],
    seed: int = 20260814,
    validation_ratio: float = 0.1,
    bucket_mode: str = "difficulty",
    stage_plan: str = "reverse",
) -> dict:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    stage_config = stage_config_for(bucket_mode, stage_plan)
    bucket_names = list(
        dict.fromkeys(
            name for config in stage_config.values() for name in config["buckets"]
        )
    )
    row_ids = [int(row["task_id"]) for row in rows]
    label_ids = [int(label["task_id"]) for label in labels]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate SFT task IDs")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("duplicate difficulty-label task IDs")
    if set(row_ids) != set(label_ids):
        raise ValueError("SFT rows and difficulty labels must have identical task IDs")
    overlap = sorted(set(row_ids) & {int(task_id) for task_id in evaluation_ids})
    if overlap:
        raise ValueError(f"evaluation overlap: {overlap[:10]}")

    labels_by_id = {int(label["task_id"]): label for label in labels}
    bucket_ids = {name: [] for name in bucket_names}
    review = {
        "candidate_comparison_under_evidenced": [],
        "consecutive_exact_action": [],
        "search_heavy": [],
        "trajectory_long": [],
    }
    limits = {
        "simple": {"searches": 2, "steps": 10},
        "medium": {"searches": 4, "steps": 16},
        "hard": {"searches": 6, "steps": 25},
    }
    for row in rows:
        task_id = int(row["task_id"])
        label = labels_by_id[task_id]
        if label.get("difficulty") not in limits:
            raise ValueError(f"task {task_id}: invalid difficulty")
        calls = _validate_row(row)
        if bucket_mode == "difficulty":
            bucket_ids[_bucket(label)].append(task_id)
        else:
            bucket_ids[behavior_bucket(behavior_profile(row))].append(task_id)
        opened_asins = set()
        for message in row.get("messages") or []:
            if message.get("role") != "tool":
                continue
            match = ASIN_PATTERN.search(str(message.get("content") or ""))
            if match:
                opened_asins.add(match.group(1).strip())
        if label.get("needs_candidate_comparison") and len(opened_asins) < 2:
            review["candidate_comparison_under_evidenced"].append(task_id)
        if any(left == right for left, right in zip(calls, calls[1:])):
            review["consecutive_exact_action"].append(task_id)
        difficulty_limits = limits[label["difficulty"]]
        if sum(name == "search_products" for name, _ in calls) > difficulty_limits["searches"]:
            review["search_heavy"].append(task_id)
        if len(calls) > difficulty_limits["steps"]:
            review["trajectory_long"].append(task_id)

    buckets = {}
    for name, ids in bucket_ids.items():
        train, validation = _split(ids, seed=seed, validation_ratio=validation_ratio)
        buckets[name] = {
            "rows": len(ids),
            "train_task_ids": train,
            "validation_task_ids": validation,
        }
    stages = {}
    for name, config in stage_config.items():
        selected = config["buckets"]
        stages[name] = {
            **config,
            "train_rows": sum(len(buckets[bucket]["train_task_ids"]) for bucket in selected),
            "validation_rows": sum(
                len(buckets[bucket]["validation_task_ids"]) for bucket in selected
            ),
        }
    train_count = sum(len(bucket["train_task_ids"]) for bucket in buckets.values())
    validation_count = sum(
        len(bucket["validation_task_ids"]) for bucket in buckets.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "split_seed": int(seed),
        "validation_ratio": float(validation_ratio),
        "bucket_mode": bucket_mode,
        "stage_plan": stage_plan,
        "counts": {
            "rows": len(rows),
            "train": train_count,
            "validation": validation_count,
            "evaluation_overlap": 0,
        },
        "buckets": buckets,
        "stages": stages,
        "review_flags": {key: sorted(value) for key, value in review.items()},
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/sft_pure_v4/all.jsonl")
    parser.add_argument(
        "--labels", type=Path, default=root / "data/sft_pure_v4/difficulty_labels.jsonl"
    )
    parser.add_argument(
        "--evaluation", type=Path, default=root / "data/evaluation/tasks.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/sft_curriculum/manifest.json"
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--bucket-mode",
        choices=BUCKET_MODES,
        default="difficulty",
        help=(
            "difficulty = 旧的三段「由易到难」（模板会被练 3 遍，探索 1 遍）；"
            "behavior = 倒序两段「先探索后全量」（探索 2 遍，模板 1 遍）"
        ),
    )
    parser.add_argument(
        "--stage-plan",
        choices=STAGE_PLANS,
        default="reverse",
        help=(
            "reverse = 倒序两段（先探索后全量，M10-R v1，已证伪）；"
            "single = 单阶段全量（M10-R v2，从 0% 出发按数据比例学）"
        ),
    )
    parser.add_argument(
        "--admission-contract",
        default=None,
        help="Stamp manifest.admission with this contract version (M4 rebuild keeps it).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = _read_jsonl(args.source)
    labels = _read_jsonl(args.labels)
    evaluation_ids = {int(row["task_id"]) for row in _read_jsonl(args.evaluation)}
    manifest = build_manifest(
        rows,
        labels,
        evaluation_ids=evaluation_ids,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        bucket_mode=args.bucket_mode,
        stage_plan=args.stage_plan,
    )
    manifest["source"] = {
        "path": _portable_path(args.source, root),
        "sha256": _sha256(args.source),
    }
    manifest["labels"] = {
        "path": _portable_path(args.labels, root),
        "sha256": _sha256(args.labels),
    }
    manifest["evaluation"] = {
        "path": _portable_path(args.evaluation, root),
        "sha256": _sha256(args.evaluation),
    }
    if args.admission_contract:
        manifest["admission"] = {
            "contract": args.admission_contract,
            "applied": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
