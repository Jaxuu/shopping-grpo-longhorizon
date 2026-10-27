"""按「模型做对几条」给训练池分桶，只留中等难度题 —— GRPO 唯一能学到东西的那批。

为什么需要
----------
M11 探针（lr 1e-5、`train_hard50.parquet` 401 题、25 步）跑完，Final-200 是 **126/200**，
基线 `m10r-v2` 是 127/200，**逐题 0 进 1 出**（m11 做对的 126 题是基线 127 题的真子集）。
训练日志暴露了原因：

    group/all_equal_ratio        0.25 – 0.50
    group/no_purchase_success    0.375 – 0.75
    group/all_purchase_success   0.00 – 0.375

GRPO 的梯度来自「组内 4 条轨迹奖励不一样」。全对或全错的组，4 条奖励相同，
优势恒为 0 —— 这批题在训练里只贡献算力，不贡献方向。401 题池里有 201 道难题，
模型稳定地做不出来，4 条全错，**反而制造了更多零信号组**。

所以：加难题的方向没错，错在难度跨度太大。只有「对 1~3 条」的题有奖励差可学。

本脚本做两件事
--------------
1. `export`：把训练池的 task_id 写成评测集格式（`{"task_id": N}` 一行），
   好让 `scripts/eval_sharded.sh` 直接跑（它只认这个格式）。
2. `select`：读多轮评测的轨迹，统计每题「N 条里对几条」，按桶筛选，
   写出新的 parquet（schema 与原池完全一致，可直接喂 GRPO）。

口径
----
「对」沿用评测层的严格口径（`reward_type == gold_purchase` 且终局/Reward 全部自洽），
直接复用 `shopping_grpo.evaluation.summary._is_strict_success`，**不另写一套判断**，
避免与 63.5%/63.0% 那些历史分数口径漂移。

用法
----
    # ① 导出 task_id 清单
    .venv/bin/python scripts/label_pool_difficulty.py export \\
        --parquet data/grpo/train_hard50.parquet \\
        --tasks-out data/grpo/hard50_tasks.jsonl

    # ② 用评测脚本跑 4 轮（temperature 0.7，见回答里的命令）

    # ③ 统计 + 筛选
    .venv/bin/python scripts/label_pool_difficulty.py select \\
        --parquet data/grpo/train_hard50.parquet \\
        --eval-dir outputs/evaluation/hard50-r1 \\
        --eval-dir outputs/evaluation/hard50-r2 \\
        --eval-dir outputs/evaluation/hard50-r3 \\
        --eval-dir outputs/evaluation/hard50-r4 \\
        --output data/grpo/train_medium.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # 允许直接跑脚本时 import 到 src
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.summary import _is_strict_success  # noqa: E402


def read_parquet(path: Path) -> tuple[list[dict], object]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境缺 pyarrow
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    table = pq.read_table(path)
    return table.to_pylist(), table.schema


def write_parquet(rows: list[dict], schema, path: Path) -> None:
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def row_task_id(row: dict) -> int:
    return int(row["extra_info"]["task_id"])


def read_trajectories(eval_dir: Path) -> list[dict]:
    """读一个评测目录下所有分片的轨迹（合并前的原始产物）。"""
    records: list[dict] = []
    shard_root = eval_dir / "shards"
    if shard_root.is_dir():
        for path in sorted(shard_root.glob("shard-*/trajectories.jsonl")):
            records.extend(_read_jsonl(path))
    # 也接受已经合并过的单文件
    merged = eval_dir / "trajectories.jsonl"
    if merged.is_file():
        records.extend(_read_jsonl(merged))
    if not records:
        raise SystemExit(
            f"{eval_dir} 里没有找到任何轨迹。\n"
            f"  找过：{shard_root}/shard-*/trajectories.jsonl 和 {merged}"
        )
    return records


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cmd_export(args: argparse.Namespace) -> int:
    rows, _ = read_parquet(args.parquet)
    ids = [row_task_id(row) for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("训练池里有重复 task_id，拒绝导出")
    args.tasks_out.parent.mkdir(parents=True, exist_ok=True)
    with args.tasks_out.open("w", encoding="utf-8") as handle:
        for task_id in ids:
            handle.write(json.dumps({"task_id": task_id}) + "\n")
    print(f"导出 {len(ids)} 个 task_id → {args.tasks_out}")
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    rows, schema = read_parquet(args.parquet)
    pool_ids = [row_task_id(row) for row in rows]
    pool_set = set(pool_ids)

    # 每题累计：看过几条、对了几条
    seen: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    # 每题每条轨迹的结局，用于报告（gold / 其它 reward_type）
    outcomes: dict[int, list[str]] = defaultdict(list)

    per_round: list[str] = []
    for eval_dir in args.eval_dir:
        eval_dir = Path(eval_dir).expanduser().resolve()
        records = read_trajectories(eval_dir)
        matched = 0
        for record in records:
            task_id = record.get("task_id")
            if task_id is None:
                continue
            task_id = int(task_id)
            if task_id not in pool_set:
                continue
            matched += 1
            seen[task_id] += 1
            detail = (record.get("terminal_result") or {}).get("reward_detail") or {}
            reward_type = str(detail.get("reward_type") or "unknown")
            outcomes[task_id].append(reward_type)
            if _is_strict_success(record):
                correct[task_id] += 1
        per_round.append(f"{eval_dir.name}: {matched} 条轨迹（池内）")

    missing = sorted(pool_set - set(seen))
    total_trajectories = sum(seen.values())
    if total_trajectories == 0:
        raise SystemExit("所有评测目录里都没有匹配到池内的 task_id，检查是否跑错了池子")

    # 分桶：以「该题实际看到的轨迹数」为分母，容忍个别轮次缺题
    buckets: dict[str, list[int]] = {"all_wrong": [], "medium": [], "all_correct": []}
    for task_id in pool_ids:
        total = seen.get(task_id, 0)
        if total == 0:
            continue
        hit = correct.get(task_id, 0)
        if hit == 0:
            buckets["all_wrong"].append(task_id)
        elif hit >= total:
            buckets["all_correct"].append(task_id)
        else:
            buckets["medium"].append(task_id)

    print("== 每轮命中 ==")
    for line in per_round:
        print(f"  {line}")
    if missing:
        print(f"  ⚠ 池内 {len(missing)} 题一轮都没跑出结果：{missing[:10]}{' …' if len(missing) > 10 else ''}")

    print("\n== 分桶 ==")
    label_map = {
        "all_wrong": "全错（梯度恒为 0）",
        "medium": "中等（有奖励差，可学）",
        "all_correct": "全对（梯度恒为 0）",
    }
    for key in ("all_wrong", "medium", "all_correct"):
        count = len(buckets[key])
        share = count / max(len(pool_ids), 1)
        print(f"  {label_map[key]:<22}: {count:>4} 题（{share:5.1%}）")
    print(f"  {'合计':<20}: {len(pool_ids):>4} 题")

    # 筛选
    min_correct = args.min_correct
    max_correct = args.max_correct
    selected: list[int] = []
    for task_id in pool_ids:
        total = seen.get(task_id, 0)
        hit = correct.get(task_id, 0)
        if total == 0:
            # 一轮都没跑出结果 = 没有难度证据。默认不收；显式要求时按「未知」收进来。
            if args.include_missing:
                selected.append(task_id)
            continue
        upper = total - 1 if max_correct is None else max_correct
        if min_correct <= hit <= upper:
            selected.append(task_id)

    selected_set = set(selected)
    kept_rows = [row for row in rows if row_task_id(row) in selected_set]
    print(
        f"\n== 筛选（对 {min_correct} ~ "
        f"{'总数-1' if max_correct is None else max_correct} 条）=="
    )
    print(f"  保留 {len(kept_rows)} 题 / 原池 {len(rows)} 题")

    if args.dry_run:
        print("  dry-run：只统计分桶，未写文件")
        return 0

    if not kept_rows:
        raise SystemExit("筛选结果为空：检查 --min-correct / --max-correct，或轮数是否太少")

    write_parquet(kept_rows, schema, args.output)
    print(f"  已写出 {args.output}")

    if args.report_json:
        report = {
            "pool": str(args.parquet),
            "pool_size": len(rows),
            "eval_dirs": [str(Path(p).resolve()) for p in args.eval_dir],
            "rounds": len(args.eval_dir),
            "buckets": {key: sorted(value) for key, value in buckets.items()},
            "missing": missing,
            "selected": sorted(selected_set),
            "selected_size": len(kept_rows),
            "per_task": {
                str(task_id): {
                    "seen": seen[task_id],
                    "correct": correct.get(task_id, 0),
                    "reward_types": outcomes.get(task_id, []),
                }
                for task_id in sorted(pool_ids)
                if task_id in seen
            },
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  明细报告 {args.report_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="训练池 parquet → 评测集格式 task_id 清单")
    export.add_argument("--parquet", type=Path, required=True)
    export.add_argument("--tasks-out", type=Path, required=True)
    export.set_defaults(func=cmd_export)

    select = sub.add_parser("select", help="按做对条数分桶并筛选出新 parquet")
    select.add_argument("--parquet", type=Path, required=True)
    select.add_argument(
        "--eval-dir",
        type=Path,
        action="append",
        required=True,
        help="评测产物目录（可重复，一次一轮）",
    )
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--min-correct", type=int, default=1)
    select.add_argument(
        "--max-correct",
        type=int,
        default=None,
        help="默认 None = 该题实际轨迹数 - 1（即「不是全对」）",
    )
    select.add_argument(
        "--include-missing",
        action="store_true",
        help="把一轮都没跑出结果的题也算进来（默认不算，因为没证据）",
    )
    select.add_argument("--report-json", type=Path, default=None)
    select.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印分桶，不写 parquet（轮数不够时用它先验证脚本能跑通）",
    )
    select.set_defaults(func=cmd_select)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
