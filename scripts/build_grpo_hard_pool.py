"""把「需要翻页」的难题并进 GRPO 训练池——M6/M6-R 两轮零结果的前置原因。

为什么必须做这一步
------------------
奖励只在训练题上生效。实测 `data/grpo/train.parquet` 906 题里 **899 题的 gold 就在
搜索前 20 名**（99.2%），需要翻页的只有 7 题；而 SFT 池被 M10-R 重配比过（11.9% 难题），
所以只有 SFT 教会了翻页动作，GRPO 无论跑多少步都拿不到「翻页 → 买到 gold」的奖励差。

本脚本把 `data/grpo/hard_targets.jsonl`（离线 BM25 名次审计选出、且不属于任何现有池子）
里的难题并进训练池，产出新的 parquet。system prompt 逐字取自原池第一行，
保证新旧行完全同分布（仓库里 `rollout.SYSTEM_PROMPT` 与原池差一个结尾换行，不能直接用）。

    .venv/bin/python scripts/build_grpo_hard_pool.py \
        --base data/grpo/train.parquet \
        --hard data/grpo/hard_targets.jsonl \
        --output data/grpo/train_hardmix.parquet \
        --n-hard 201

校验（任一不过就拒绝写文件）：task_id 全局唯一、与评测集零重叠、system prompt 逐字一致、
每行 user 消息能由语料复现。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "environments" / "ShopSimulator" / "shop_env" / "data" / "fine_items_eval_train_all.json.gz"
EVAL_TASKS = ROOT / "data" / "evaluation" / "tasks.jsonl"
USER_PREFIX = "Instruction: "


def read_parquet(path: Path) -> tuple[list[dict], object]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境缺 pyarrow
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    table = pq.read_table(path)
    return table.to_pylist(), table.schema


def load_corpus() -> list[dict]:
    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_hard(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if "task_id" not in row:
            raise SystemExit(f"{path} 里缺少 task_id：{row}")
    return rows


def eval_ids() -> set[int]:
    if not EVAL_TASKS.is_file():
        return set()
    ids = set()
    for line in EVAL_TASKS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(int(json.loads(line)["task_id"]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "grpo" / "train.parquet")
    parser.add_argument("--hard", type=Path, default=ROOT / "data" / "grpo" / "hard_targets.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "grpo" / "train_hardmix.parquet")
    parser.add_argument("--n-hard", type=int, default=0, help="最多并入多少道（0=全部）")
    parser.add_argument(
        "--n-easy",
        type=int,
        default=0,
        help="从原池随机保留多少道简单题（0=全部保留）。50 步只能采 ~200 个任务，"
        "全量 906 道会把难题稀释到 18%%，难题信号太稀",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_rows, schema = read_parquet(args.base)
    if not base_rows:
        raise SystemExit(f"{args.base} 是空的")
    system_prompt = base_rows[0]["prompt"][0]["content"]
    base_ids = {int(row["extra_info"]["task_id"]) for row in base_rows}
    if args.n_easy and args.n_easy < len(base_rows):
        import random  # noqa: PLC0415

        keep = sorted(random.Random(args.seed).sample(range(len(base_rows)), args.n_easy))
        base_rows = [base_rows[index] for index in keep]

    corpus = load_corpus()
    hard = load_hard(args.hard)
    if args.n_hard:
        hard = hard[: args.n_hard]

    blocked = eval_ids()
    new_rows: list[dict] = []
    skipped = {"已在池中": 0, "与评测集重叠": 0, "语料缺指令": 0, "task_id 越界": 0}
    for row in hard:
        task_id = int(row["task_id"])
        if task_id in base_ids:
            skipped["已在池中"] += 1
            continue
        if task_id in blocked:
            skipped["与评测集重叠"] += 1
            continue
        if not (0 <= task_id < len(corpus)):
            skipped["task_id 越界"] += 1
            continue
        item = corpus[task_id]
        instructions = item.get("instructions") or []
        if not instructions:
            skipped["语料缺指令"] += 1
            continue
        instruction = str(instructions[0].get("instruction") or "").strip()
        if not instruction:
            skipped["语料缺指令"] += 1
            continue
        new_rows.append(
            {
                "data_source": "shopsimulator",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_PREFIX + instruction},
                ],
                "ability": "shopping",
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {
                    "split": "train",
                    "index": len(base_rows) + len(new_rows),
                    "task_id": task_id,
                },
            }
        )

    merged = base_rows + new_rows
    ids = [int(row["extra_info"]["task_id"]) for row in merged]
    if len(set(ids)) != len(ids):
        raise SystemExit("合并后出现重复 task_id，拒绝写出")
    if set(ids) & blocked:
        raise SystemExit("合并后与评测集重叠，拒绝写出")
    if any(row["prompt"][0]["content"] != system_prompt for row in merged):
        raise SystemExit("system prompt 不一致，拒绝写出")

    total = len(merged)
    hard_share = len(new_rows) / max(total, 1)
    print(f"原池 {len(base_rows)} 题 + 新增 {len(new_rows)} 题 = {total} 题（难题占比 {hard_share:.1%}）")
    for reason, count in skipped.items():
        if count:
            print(f"  跳过 {reason}：{count}")
    if args.dry_run:
        print("dry-run：未写文件")
        return 0

    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(merged, schema=schema)
    pq.write_table(table, args.output)
    print(f"写出 {args.output}（{total} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
