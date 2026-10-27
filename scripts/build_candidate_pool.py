"""从语料里抽一批「没被任何池子用过」的题，供难度打标 —— M11 扩池。

为什么需要
----------
M11 探针（lr 1e-5、`train_hard50.parquet` 401 题、25 步）跑完 Final-200 是 126/200，
基线 127/200，**逐题 0 进 1 出**。训练日志里 `all_equal_ratio` 0.25–0.5、
`no_purchase_success_ratio` 0.375–0.75 ⇒ 大量组是「4 条全对」或「4 条全错」，
组内优势恒为 0，这些题只耗算力不产方向。

随后用温度 0.7 给 401 题打了两轮难度标签，实际分桶：

    全错  223 题（55.6%）
    中等   57 题（14.2%）   ← GRPO 唯一能学的一批
    全对  121 题（30.2%）

57 道不够训（batch 2 × 100 步会让每题重复 3.5 遍）。而 401 题池里 201 道是**特意选的难题**，
难度偏置严重 —— 随机抽的题里中等比例应该高得多。

本脚本做的事
------------
从语料（`fine_items_eval_train_all.json.gz`，task_id = 行号）里剔除**所有已经被用过的 id**
（评测集、GRPO 各池、SFT 池），随机抽 N 道，产出与 `train.parquet` **同 schema、同 system
prompt** 的 parquet，直接丢给 `label_pool_difficulty.py export` + `eval_sharded.sh` 打标。

为什么必须排除已用的 id
----------------------
- 评测集 200 题：绝不能进训练，否则 63.5% 那个数作废。
- `train.parquet` 906 题 + `train_hard50` 401 题：已经打过分桶，重复抽是浪费。
- SFT 池：模型已经背过这批题，拿它们当「中等题」会得到虚假的奖励差。

用法
----
    # 先看候选池有多大、抽样分布如何（不写文件）
    .venv/bin/python scripts/build_candidate_pool.py --n 1000 --dry-run

    # 正式抽 1000 道
    .venv/bin/python scripts/build_candidate_pool.py --n 1000 \\
        --output data/grpo/candidate1k.parquet

    # 再导出 task_id 去打标
    .venv/bin/python scripts/label_pool_difficulty.py export \\
        --parquet data/grpo/candidate1k.parquet \\
        --tasks-out data/grpo/candidate1k_tasks.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "environments" / "ShopSimulator" / "shop_env" / "data" / "fine_items_eval_train_all.json.gz"
EVAL_TASKS = ROOT / "data" / "evaluation" / "tasks.jsonl"
GRPO_DIR = ROOT / "data" / "grpo"
USER_PREFIX = "Instruction: "

# 默认要排除的池子。都是「模型已经见过或已经打过标」的题。
DEFAULT_EXCLUDE = (
    EVAL_TASKS,
    GRPO_DIR / "train.jsonl",
    GRPO_DIR / "validation.jsonl",
    GRPO_DIR / "hard_targets.jsonl",
    GRPO_DIR / "train.parquet",
    GRPO_DIR / "validation.parquet",
    GRPO_DIR / "train_hard50.parquet",
    GRPO_DIR / "train_hardmix.parquet",
)


def load_corpus() -> list[dict]:
    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def ids_from_jsonl(path: Path) -> set[int]:
    ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task_id" in row and row["task_id"] is not None:
                ids.add(int(row["task_id"]))
            elif isinstance(row.get("extra_info"), dict) and row["extra_info"].get("task_id") is not None:
                ids.add(int(row["extra_info"]["task_id"]))
    return ids


def ids_from_parquet(path: Path) -> set[int]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境缺 pyarrow
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    # 只读需要的那一列：SFT 池的 parquet 带完整 messages，整表读会白等几十秒。
    parquet_file = pq.ParquetFile(path)
    names = set(parquet_file.schema_arrow.names)
    if "task_id" in names:
        column = parquet_file.read(columns=["task_id"]).column("task_id").to_pylist()
        return {int(value) for value in column if value is not None}
    if "extra_info" in names:
        column = parquet_file.read(columns=["extra_info"]).column("extra_info").to_pylist()
        ids = set()
        for row in column:
            if isinstance(row, dict) and row.get("task_id") is not None:
                ids.add(int(row["task_id"]))
        return ids
    raise SystemExit(f"{path} 里既没有 task_id 也没有 extra_info.task_id")


def sft_pool_files() -> list[Path]:
    """所有 `data/sft*` 目录下的 parquet。

    曾经只写死 `data/sft_m10r`，结果服务器上还有 `data/sft`、`data/sft_curriculum`、
    `data/sft_pure_v4` 三个池子，全被漏掉 —— 那批题模型已经背过，混进候选池会给出
    虚假的「有对有错」。
    """
    data_dir = ROOT / "data"
    found: list[Path] = []
    for pattern in ("sft*/*.parquet", "sft*/**/*.parquet"):
        found.extend(sorted(data_dir.glob(pattern)))
    # 去重（两个 pattern 可能命中同一文件）并保持稳定顺序
    return sorted({path for path in found if path.is_file()})


def collect_used_ids(extra: list[Path]) -> tuple[set[int], list[tuple[str, int]]]:
    """返回（已用 id 集合，每个来源贡献了多少）。"""
    used: set[int] = set()
    detail: list[tuple[str, int]] = []
    paths = list(DEFAULT_EXCLUDE)
    paths.extend(sft_pool_files())
    paths.extend(extra)

    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            ids = ids_from_jsonl(path)
        elif path.suffix == ".parquet":
            ids = ids_from_parquet(path)
        else:
            continue
        before = len(used)
        used |= ids
        detail.append((str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), len(used) - before))
    return used, detail


def instruction_of(item: dict) -> str:
    instructions = item.get("instructions") or []
    if not instructions:
        return ""
    return str(instructions[0].get("instruction") or "").strip()


def read_base_template() -> tuple[list[dict], object]:
    """从 train.parquet 取 system prompt 与 schema，保证新行与原池完全同分布。"""
    base = GRPO_DIR / "train.parquet"
    if not base.is_file():
        raise SystemExit(f"缺少 {base}，无法取 schema / system prompt")
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    table = pq.read_table(base)
    rows = table.to_pylist()
    if not rows:
        raise SystemExit(f"{base} 是空的")
    return rows, table.schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=1000, help="抽多少道")
    parser.add_argument("--output", type=Path, default=GRPO_DIR / "candidate.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="额外要排除的池子（jsonl / parquet，可重复）",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_corpus()
    used, detail = collect_used_ids(args.exclude)

    # 候选：不在任何池子里、且有可用指令
    candidates: list[int] = []
    skipped_no_instruction = 0
    skipped_used = 0
    for task_id, item in enumerate(corpus):
        if task_id in used:
            skipped_used += 1
            continue
        if not instruction_of(item):
            skipped_no_instruction += 1
            continue
        candidates.append(task_id)

    print(f"语料 {len(corpus)} 条")
    print(f"  已用（各池子去重后）: {len(used)}")
    print(f"  跳过（已在池中）    : {skipped_used}")
    print(f"  跳过（语料缺指令）  : {skipped_no_instruction}")
    print(f"  可抽候选            : {len(candidates)}")
    print("\n排除来源：")
    for name, count in detail:
        if count:
            print(f"  {name:<52} +{count}")
    sft_files = sft_pool_files()
    print(f"  （SFT 池扫到 {len(sft_files)} 个 parquet，其 task_id 已并入上面的去重集合）")

    if args.n > len(candidates):
        raise SystemExit(f"要抽 {args.n} 道，但候选只有 {len(candidates)} 道")

    picked = sorted(random.Random(args.seed).sample(candidates, args.n))
    if set(picked) & used:
        raise SystemExit("抽样结果与已用 id 重叠，拒绝写文件")

    base_rows, schema = read_base_template()
    system_prompt = base_rows[0]["prompt"][0]["content"]

    new_rows: list[dict] = []
    for index, task_id in enumerate(picked):
        new_rows.append(
            {
                "data_source": "shopsimulator",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_PREFIX + instruction_of(corpus[task_id])},
                ],
                "ability": "shopping",
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {"split": "train", "index": index, "task_id": task_id},
            }
        )

    print(f"\n抽中 {len(new_rows)} 道，task_id 范围 {picked[0]} – {picked[-1]}")
    if args.dry_run:
        print("dry-run：未写文件")
        return 0

    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"需要 pyarrow：{exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(new_rows, schema=schema), args.output)
    print(f"写出 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
