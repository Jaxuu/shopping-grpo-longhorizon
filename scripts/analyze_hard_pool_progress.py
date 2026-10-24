#!/usr/bin/env python3
"""训练中途判断「难题有没有开始被解决」——不必等 Final-200 评测。

为什么需要这个脚本
------------------
M10-H 的训练池是「200 简单 + 201 难题」各一半。难题的 gold 在 BM25 前 20 名之外，
**不翻页（或换出更精确的查询词）就买不到匹配商品**。所以「难题上出现
`purchase_success`」这件事本身就是行为改变的强证据，不需要跑 200 题评测。

诊断文件 `training_diagnostics.jsonl` 的 `generation_batch` 事件里，每个 rollout 组都记了
`terminal_utilities` 和 `purchase_success`。本脚本按 uid 把组拆成「难题 / 简单题」两组，
逐步统计：组数、kept 率、**买到匹配商品的组占比**、utility 均值。

判据（预登记，跑完不许改）
--------------------------
- 难题组「买到匹配商品」的占比，全程恒为 0 ⇒ 翻页和换词都没发生，负结果
- 从 0 抬到 ≥10% ⇒ 难题开始被解决（**注意**：换词也能达成，不等于翻页；
  翻页率仍需 Final-200 的轨迹分析）

用法
----
    .venv/bin/python scripts/analyze_hard_pool_progress.py \
        outputs/models/grpo-m10h/training_diagnostics.jsonl \
        --pool data/grpo/train_hard50.parquet \
        --window 10
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VAL_PREFIXES = ("val-core/", "val-aux/", "val/")
VAL_PRIMARY = "val-core/shopsimulator/reward/mean@1"


def coerce_uid(uid: object) -> int | None:
    """把 uid 归一成整数。

    veRL 的 uid 可能是 `int`、纯数字字符串，或 `[task_id, rollout_index]` 这类列表。
    只接受**能无损解释为整数**的形式：不做「从字符串里抠数字」这种猜测，
    否则 uuid 里的数字会被当成 task_id，静默算错。
    """
    if isinstance(uid, bool):
        return None
    if isinstance(uid, int):
        return uid
    if isinstance(uid, float):
        return int(uid) if uid.is_integer() else None
    if isinstance(uid, str):
        text = uid.strip()
        return int(text) if text.isdigit() else None
    if isinstance(uid, (list, tuple)) and len(uid) == 1:
        return coerce_uid(uid[0])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("diagnostics", type=Path, help="training_diagnostics.jsonl 路径")
    parser.add_argument("--pool", type=Path, default=None, help="训练池 parquet（含 extra_info.task_id）")
    parser.add_argument("--hard", type=Path, default=None, help="hard_targets.jsonl（默认与 pool 同目录）")
    parser.add_argument("--window", type=int, default=10, help="趋势汇总的步长窗口（默认 10）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"诊断文件不存在：{path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return records


def load_pool_ids(pool: Path, hard: Path) -> tuple[dict[str, set], int]:
    """返回 {口径名: 难题 uid 集合} 与池子行数。

    `.jsonl` 只用于离线自测（服务器上训练池是 parquet）。
    """
    if pool.suffix == ".jsonl":
        table = [
            json.loads(line)
            for line in pool.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"需要 pyarrow：{exc}") from exc
        table = pq.read_table(pool).to_pylist()
    hard_ids = {
        int(json.loads(line)["task_id"])
        for line in hard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    hard_by_index: set[int] = set()
    hard_by_task: set[int] = set()
    for row in table:
        info = row.get("extra_info") or {}
        task_id = int(info["task_id"])
        index = info.get("index")
        if task_id in hard_ids:
            hard_by_task.add(task_id)
            if index is not None:
                hard_by_index.add(int(index))
    if not hard_by_task:
        raise SystemExit(f"{pool} 里找不到 {hard} 中的任何 task_id，口径对不上")
    return {"task_id": hard_by_task, "index": hard_by_index}, len(table)


def detect_uid_kind(uids: list, hard_sets: dict[str, set], n_rows: int) -> tuple[str, dict]:
    """判断诊断文件里的 uid 是 task_id 还是 index，并给出匹配率。

    优先认 task_id：它的取值是语料行号，范围远大于池子行数，一旦出现 >n_rows 的值即可锁定。
    匹配率低于 30% 时不硬猜，返回 unknown 让上层打印诊断信息。
    """
    coerced = [coerce_uid(uid) for uid in uids]
    values = [value for value in coerced if value is not None]
    diagnostics = {
        "uid_total": len(uids),
        "uid_parsed": len(values),
        "uid_type_counts": {
            str(kind): count
            for kind, count in Counter(type(uid).__name__ for uid in uids).most_common()
        },
        "uid_samples": [repr(uid) for uid in uids[:8]],
        "uid_value_range": (min(values), max(values)) if values else None,
        "match_task_id": 0,
        "match_index": 0,
    }
    if not values:
        return "unknown", diagnostics

    as_task = sum(1 for value in values if value in hard_sets["task_id"])
    as_index = sum(1 for value in values if value in hard_sets["index"])
    diagnostics["match_task_id"] = as_task
    diagnostics["match_index"] = as_index
    has_beyond = any(value >= n_rows for value in values)
    if has_beyond or as_task > as_index:
        kind = "task_id"
        matched = as_task
    elif as_index > 0:
        kind = "index"
        matched = as_index
    else:
        return "unknown", diagnostics
    # 难题只占池子一半，抽到的组里难题应占 30%+；低于此说明口径猜错了
    if matched / len(values) < 0.30:
        return "unknown", diagnostics
    return kind, diagnostics


def summarize_val(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        if record.get("event") != "optimizer_step":
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        picked = {
            str(key): float(value)
            for key, value in metrics.items()
            if str(key).startswith(VAL_PREFIXES)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        if picked:
            rows.append({"step": record.get("global_step"), "metrics": picked})
    return sorted(rows, key=lambda row: row["step"] or 0)


def collect_groups(records: list[dict]) -> dict[int, list[dict]]:
    """把每个 generation_batch 里的组按步收集（同一步可能多个 batch）。"""
    per_step: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("event") != "generation_batch":
            continue
        step = record.get("global_step")
        if step is None:
            continue
        for group in record.get("groups") or []:
            per_step[int(step)].append(group)
    return per_step


def group_stats(groups: list[dict]) -> dict:
    if not groups:
        # 空组也要返回完整键：window_rows 会无条件取 kept / any_success / rollouts
        return {
            "groups": 0,
            "kept": 0,
            "kept_rate": 0.0,
            "any_success": 0,
            "any_success_rate": 0.0,
            "no_success": 0,
            "utility_mean": 0.0,
            "rollouts": 0,
        }
    kept = sum(1 for group in groups if group.get("kept"))
    any_success = sum(
        1 for group in groups if any(group.get("purchase_success") or ())
    )
    no_success = sum(
        1 for group in groups if not any(group.get("purchase_success") or ())
    )
    utilities = [
        float(value)
        for group in groups
        for value in (group.get("terminal_utilities") or ())
    ]
    return {
        "groups": len(groups),
        "kept": kept,
        "kept_rate": kept / len(groups),
        "any_success": any_success,
        "any_success_rate": any_success / len(groups),
        "no_success": no_success,
        "utility_mean": statistics.mean(utilities) if utilities else 0.0,
        "rollouts": len(utilities),
    }


def split_by_hardness(
    per_step: dict[int, list[dict]], hard_set: set, uid_kind: str
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for step in sorted(per_step):
        hard_groups, easy_groups, unknown = [], [], 0
        for group in per_step[step]:
            uid = coerce_uid(group.get("uid"))
            if uid is None:
                unknown += 1
                continue
            if uid in hard_set:
                hard_groups.append(group)
            else:
                easy_groups.append(group)
        out[step] = {
            "hard": group_stats(hard_groups),
            "easy": group_stats(easy_groups),
            "unknown_uid": unknown,
        }
    return out


def window_rows(stats: dict[int, dict], window: int) -> list[dict]:
    rows = []
    steps = sorted(stats)
    for start in range(0, len(steps), window):
        chunk = steps[start : start + window]
        agg: dict[str, dict] = {}
        for side in ("hard", "easy"):
            groups = sum(stats[step][side]["groups"] for step in chunk)
            kept = sum(stats[step][side]["kept"] for step in chunk)
            success = sum(stats[step][side]["any_success"] for step in chunk)
            utilities = []
            # 均值按组加权：用 rollouts 数近似（每组 n 条）
            for step in chunk:
                side_stats = stats[step][side]
                if side_stats["rollouts"]:
                    utilities.append(side_stats["utility_mean"] * side_stats["rollouts"])
            total_rollouts = sum(stats[step][side]["rollouts"] for step in chunk)
            agg[side] = {
                "groups": groups,
                "kept_rate": kept / groups if groups else 0.0,
                "any_success_rate": success / groups if groups else 0.0,
                "utility_mean": sum(utilities) / total_rollouts if total_rollouts else 0.0,
            }
        rows.append({"steps": (chunk[0], chunk[-1]), **agg})
    return rows


def render(summary: dict, args: argparse.Namespace) -> str:
    lines = ["# M10-H 训练中途诊断：难题有没有开始被解决", ""]
    lines += [
        f"- 诊断文件：`{args.diagnostics}`",
        f"- 已完成 optimizer 更新：**{summary['steps_done']}** 步"
        f"（重试轮次 {summary['skipped_rounds']}）",
        f"- 训练池 {summary['pool_rows']} 题，其中难题 {summary['hard_count']} 道",
        f"- uid 口径判定：**{summary['uid_kind']}**",
        "",
    ]
    if summary["uid_kind"] == "unknown":
        diag = summary.get("uid_diag") or {}
        lines += [
            "> ⚠️ 认不出 uid 是 task_id 还是 index，下面的难题/简单题拆分**不可信**。",
            "",
            "| 诊断项 | 值 |",
            "|---|---|",
            f"| 有 groups 的步数 | {summary.get('steps_with_groups')} |",
            f"| 收集到的组总数 | {summary.get('group_records')} |",
            f"| uid 样本 | {diag.get('uid_samples')} |",
            f"| uid 类型分布 | {diag.get('uid_type_counts')} |",
            f"| 能解析成整数的比例 | {diag.get('uid_parsed')}/{diag.get('uid_total')} |",
            f"| 解析后的取值区间 | {diag.get('uid_value_range')} |",
            f"| 命中难题 task_id 的次数 | {diag.get('match_task_id')} |",
            f"| 命中难题 index 的次数 | {diag.get('match_index')} |",
            f"| 池子行数 | {summary['pool_rows']} |",
            "",
            "> 把这整段发给我，我按真实口径改判据。若「收集到的组总数」是 0，"
            "说明补丁没往诊断文件写 `groups`，那是另一回事。",
            "",
        ]

    val_rows = summary["val"]
    if val_rows:
        keys = sorted({key for row in val_rows for key in row["metrics"]})
        primary = VAL_PRIMARY if VAL_PRIMARY in keys else (keys[0] if keys else None)
        lines += ["## 1. 验证集分数（49 题，**全是简单题**）", ""]
        lines += [f"| step | `{primary}` |", "|---:|---:|"]
        for row in val_rows:
            value = row["metrics"].get(primary)
            lines.append(f"| {row['step']} | {value:.4f} |" if value is not None else f"| {row['step']} | — |")
        lines += [
            "",
            "> 这 49 题与训练池、评测池两两零重叠，且**都是简单题**（gold 在前 20 名）。",
            "> 它只能回答「简单题有没有退化」，**回答不了「翻页学会没有」**。",
            "> 49 题的噪声带宽是 ±1/49 ≈ ±2pp，两个点差 1 道题都算噪声。",
            "",
        ]

    lines += [
        f"## 2. 难题 vs 简单题（每 {args.window} 步汇总）",
        "",
        "> 「买到匹配」= 该组 4 条 rollout 里至少一条 `purchase_success=True`"
        "（`gold_purchase` 或 `valid_alternative_purchase`）。",
        "> 难题的 gold 在 BM25 前 20 名之外 ⇒ **不翻页或换出更精确的词就买不到**，"
        "所以这一列从 0 抬起来 = 难题开始被解决。",
        "",
        "| 步 | 难题组数 | 难题 kept | 难题买到匹配 | 难题 utility | 简单组数 | 简单 kept | 简单买到匹配 | 简单 utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["windows"]:
        hard, easy = row["hard"], row["easy"]
        lines.append(
            f"| {row['steps'][0]}–{row['steps'][1]} | {hard['groups']} | "
            f"{hard['kept_rate']:.1%} | **{hard['any_success_rate']:.1%}** | "
            f"{hard['utility_mean']:+.3f} | {easy['groups']} | {easy['kept_rate']:.1%} | "
            f"{easy['any_success_rate']:.1%} | {easy['utility_mean']:+.3f} |"
        )

    hard_all = summary["overall"]["hard"]
    easy_all = summary["overall"]["easy"]
    lines += [
        "",
        "| 全程 | 组数 | kept | 买到匹配 | utility |",
        "|---|---:|---:|---:|---:|",
        f"| 难题 | {hard_all['groups']} | {hard_all['kept_rate']:.1%} | "
        f"**{hard_all['any_success_rate']:.1%}** | {hard_all['utility_mean']:+.3f} |",
        f"| 简单题 | {easy_all['groups']} | {easy_all['kept_rate']:.1%} | "
        f"{easy_all['any_success_rate']:.1%} | {easy_all['utility_mean']:+.3f} |",
        "",
        "## 3. 判读",
        "",
    ]

    if summary["uid_kind"] == "unknown":
        lines.append("- 口径未定，先解决上面的 uid 问题。")
    elif hard_all["groups"] == 0:
        lines.append("- ⚠️ 一个难题组都没抽到 —— 训练池里难题没生效，检查 `--train-data`。")
    elif hard_all["any_success_rate"] == 0.0:
        lines += [
            f"- 🔴 **{hard_all['groups']} 个难题组、{hard_all['groups'] * 4} 条 rollout 里，"
            "一次匹配商品都没买到** ⇒ 翻页和换词都没发生。",
            "- 这基本锁定负结果：模型在难题上行为没变。**建议现在就停**，不必等剩下 44 步"
            "（省 ~1.6 小时）——但停之前先确认这不是「前 50 步还在预热」。",
        ]
    elif hard_all["any_success_rate"] < 0.10:
        lines += [
            f"- 🟡 难题买到匹配的组占 **{hard_all['any_success_rate']:.1%}**（<10%）："
            "零星有信号，但很弱。",
            "- 继续跑完再看 Final-200；若终局仍 <63%，说明奖励太稀疏，要动的是「每题采样条数 n」"
            "或「把难题再分层」，不是继续加步数。",
        ]
    else:
        lines += [
            f"- 🟢 难题买到匹配的组占 **{hard_all['any_success_rate']:.1%}**（≥10%）："
            "难题确实开始被解决了。",
            "- **注意**：换出更精确的查询词也能达成，这一列不等于翻页率。"
            "翻页率仍要等 Final-200 的轨迹分析（`analyze_paging_necessity.py`）。",
        ]

    if easy_all["groups"] and easy_all["kept_rate"] < 0.3:
        lines += [
            f"- ⚠️ 简单题组 kept 率只有 {easy_all['kept_rate']:.1%} —— 大量组内奖励全同"
            "（4 条全对或全错），梯度信息偏少。",
        ]
    if summary["skipped_rounds"]:
        lines.append(
            f"- 重试轮次 {summary['skipped_rounds']} 次（连续 10 次会熔断）——"
            "重试只浪费 rollout，不丢梯度步。"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = read_records(args.diagnostics)

    pool = args.pool or Path("data/grpo/train_hard50.parquet")
    hard_path = args.hard or pool.parent / "hard_targets.jsonl"
    if not pool.is_file():
        raise SystemExit(f"训练池不存在：{pool}（用 --pool 指定）")
    if not hard_path.is_file():
        raise SystemExit(f"难题清单不存在：{hard_path}（用 --hard 指定）")
    hard_sets, pool_rows = load_pool_ids(pool, hard_path)

    per_step = collect_groups(records)
    all_uids = [
        group.get("uid") for groups in per_step.values() for group in groups
    ]
    uid_kind, uid_diag = detect_uid_kind(all_uids, hard_sets, pool_rows)
    hard_set = hard_sets.get(uid_kind, set())

    stats = split_by_hardness(per_step, hard_set, uid_kind) if uid_kind != "unknown" else {
        step: {"hard": group_stats([]), "easy": group_stats([]), "unknown_uid": 0}
        for step in sorted(per_step)
    }
    overall = {
        side: group_stats(
            [
                group
                for step in stats
                for group in per_step[step]
                if (
                    coerce_uid(group.get("uid")) is not None
                    and (coerce_uid(group.get("uid")) in hard_set) == (side == "hard")
                )
            ]
        )
        for side in ("hard", "easy")
    }

    updated = {
        int(record["global_step"])
        for record in records
        if record.get("event") == "optimizer_step" and record.get("global_step") is not None
    }
    summary = {
        "steps_done": len(updated),
        "skipped_rounds": sum(1 for r in records if r.get("event") == "skipped_update"),
        "pool_rows": pool_rows,
        "hard_count": len(hard_sets.get("task_id") or hard_sets.get("index") or ()),
        "uid_kind": uid_kind,
        "uid_diag": uid_diag,
        "group_records": sum(len(groups) for groups in per_step.values()),
        "steps_with_groups": len(per_step),
        "val": summarize_val(records),
        "windows": window_rows(stats, args.window),
        "overall": overall,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(render(summary, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
