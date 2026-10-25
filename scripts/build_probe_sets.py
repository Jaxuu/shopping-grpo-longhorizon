#!/usr/bin/env python3
"""M1：造两把内部尺子 —— dev 集与「训练集自测卷」。

为什么要 dev 集
--------------
旧线在 Final-200 上做了 5 次决策（台账 C14），考卷事实上变成了验证集，分数不再可信。
dev 集是**内部尺子**：调参、选题、选检查点只看它；Final-200 只在里程碑跑一次、跑完才看。

为什么要自测卷
-------------
`docs/project-history.md` §5.3 承诺过「让模型做自己训练过的题」但从未跑。
它是唯一能把两件事分开的测量：
* 训练集做对率 ≈ 泛化分（62%）⇒ 连见过的都不会，加量/加 epoch 才有意义；
* 训练集做对率 ≥ 90% ⇒ 学会了但泛化不过去，该加多样性，加量没用。

两个子命令
----------
``dev``
    从「从未进过任何池」的 task_id 里抽 300 道，过 ``shopping-task-admission-v1`` 契约，
    与四池 + Final-200 **零重叠**，分层轴与 Final-200 同构（硬约束轴数），便于直接对比。
``selftest``
    从训练池（默认 m10r 的 ``capped/all.jsonl``）均匀抽 300 道，生成评测能直接吃的
    benchmark 文件（每行只要 ``{"task_id": N}``，与 Final-200 的 schema 一致）。

两者都写 ``metadata.json``：sha256、seed、池清单、零重叠证明、被契约拒掉的原因分布。
**没有 metadata 的探针集不许用来做决策** —— 这是防止「又一把不准的尺子」。

用法::

    .venv/bin/python scripts/build_probe_sets.py dev --out-dir data/evaluation/dev-300
    .venv/bin/python scripts/build_probe_sets.py selftest --out-dir data/probes/selftest-300
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.admission import CONTRACT_VERSION, admit_task

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

# 已用题目的全部来源。**缺一个就说明本地不是完整环境**（训练产物在服务器上），
# 这时 dev 集可能和那个池重叠 ⇒ 默认直接报错，不许静默产出。
DEFAULT_POOLS: tuple[tuple[str, Path], ...] = (
    ("evaluation", ROOT / "data" / "evaluation" / "tasks.jsonl"),
    ("sft_train", ROOT / "data" / "sft" / "train.jsonl"),
    ("sft_validation", ROOT / "data" / "sft" / "validation.jsonl"),
    ("sft_pure_v4", ROOT / "data" / "sft_pure_v4" / "all.jsonl"),
    ("grpo_train", ROOT / "data" / "grpo" / "train.jsonl"),
    ("grpo_validation", ROOT / "data" / "grpo" / "validation.jsonl"),
    ("sft_m10r", ROOT / "data" / "sft_m10r" / "all.jsonl"),
    ("sft_m10r_capped", ROOT / "data" / "sft_m10r" / "capped" / "all.jsonl"),
    ("sft_m10r_v3", ROOT / "data" / "sft_m10r" / "capped" / "v3" / "all.jsonl"),
)

DEFAULT_SELFTEST_POOL = ROOT / "data" / "sft_m10r" / "capped" / "all.jsonl"
DEFAULT_REFERENCE = ROOT / "data" / "evaluation" / "tasks.jsonl"
DEFAULT_INDEX = ROOT / ".workbuddy-ai" / "cache" / "products_local.sqlite3"
REACH_DEPTH = 150  # 与冻结的 search.top_k 一致
REACH_QUERY_KEYS = ("Q1_full_instruction", "Q2_first12", "Q4_category_plus_attributes")

DEV_SCHEMA = "shopping-dev-dataset-v1"
PROBE_SCHEMA = "shopping-probe-benchmark-v1"

STRATA_AXIS = (
    "hard_axes = (price_upper 存在) + (expected_brand 非空) "
    "+ (expected_model 非空) + (required_options_by_key 非空)"
)


# --------------------------------------------------------------------------- io


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_ids_of(path: Path) -> list[int]:
    return [int(row["task_id"]) for row in read_jsonl(path)]


def sha256_lf(path: Path) -> str:
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(body, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------------ pools


def collect_pools(
    specs: Sequence[tuple[str, Path]], *, allow_missing: bool
) -> tuple[OrderedDict[str, list[int]], list[str]]:
    """读所有池文件，返回 (池名→task_id 列表, 缺失文件描述)。"""
    pools: OrderedDict[str, list[int]] = OrderedDict()
    missing: list[str] = []
    for name, path in specs:
        if not Path(path).exists():
            missing.append(f"{name}: {path}")
            continue
        pools[name] = task_ids_of(path)
    if missing and not allow_missing:
        raise SystemExit(
            "以下池文件不存在，dev 集无法保证零重叠（加 --allow-missing-pools 只用于本地演练）：\n  "
            + "\n  ".join(missing)
        )
    return pools, missing


def used_ids(pools: Mapping[str, Iterable[int]]) -> set[int]:
    union: set[int] = set()
    for ids in pools.values():
        union |= {int(value) for value in ids}
    return union


# ---------------------------------------------------------------------- strata


def hard_axes(goal: Mapping[str, object]) -> int:
    """这道题有几条「可以硬失败」的约束轴（0–4）。分层用。"""
    price = goal.get("price_upper")
    count = 0
    if price is not None and str(price).strip() != "":
        count += 1
    if list(goal.get("expected_brand") or []):
        count += 1
    if list(goal.get("expected_model") or []):
        count += 1
    if dict(goal.get("required_options_by_key") or {}):
        count += 1
    return count


def strata_distribution(task_ids: Iterable[int], goals: Sequence[Mapping[str, object]]) -> dict[int, int]:
    counts: Counter = Counter(hard_axes(goals[int(task_id)]) for task_id in task_ids)
    return {int(axis): int(counts[axis]) for axis in sorted(counts)}


def largest_remainder(weights: Mapping[int, int], total: int) -> dict[int, int]:
    """按比例分配整数配额，余数按小数部分从大到小补，保证总和恰为 total。"""
    weights = {int(key): int(value) for key, value in weights.items() if int(value) > 0}
    if not weights:
        return {}
    mass = sum(weights.values())
    raw = {key: total * value / mass for key, value in weights.items()}
    base = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(base.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - base[key]), key))
    for key in order[:remainder]:
        base[key] += 1
    return base


def select_dev(
    candidates: Sequence[int],
    goals: Sequence[Mapping[str, object]],
    reference: Mapping[int, int],
    size: int,
    seed: int,
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """按参照分布分层抽样；某层不够时用其余候选补齐。

    返回 (选中 task_id 升序, 目标配额, 实际选中数)。
    """
    buckets: dict[int, list[int]] = {}
    for task_id in sorted(int(value) for value in candidates):
        buckets.setdefault(hard_axes(goals[task_id]), []).append(task_id)

    targets = largest_remainder(reference, size)
    rng = random.Random(seed)
    chosen: list[int] = []
    for axis in sorted(targets):
        pool = buckets.get(axis, [])
        rng.shuffle(pool)
        chosen.extend(pool[: min(targets[axis], len(pool))])

    if len(chosen) < size:
        rest = sorted({int(value) for value in candidates} - set(chosen))
        rng.shuffle(rest)
        chosen.extend(rest[: size - len(chosen)])

    chosen = sorted(set(chosen))
    actual: Counter = Counter(hard_axes(goals[task_id]) for task_id in chosen)
    return chosen, targets, {int(k): int(v) for k, v in sorted(actual.items())}


# ----------------------------------------------------------------------- build


def admit_all(
    candidates: Sequence[int],
    goals: Sequence[Mapping[str, object]],
    product_by_asin: Mapping[str, Mapping[str, object]],
    held_out: set[int],
) -> tuple[list[int], Counter]:
    admitted: list[int] = []
    rejected: Counter = Counter()
    for task_id in candidates:
        goal = goals[int(task_id)]
        product = product_by_asin.get(str(goal.get("asin")))
        if product is None:
            rejected["no_product"] += 1
            continue
        verdict = admit_task(task_id, goal, product, held_out_task_ids=held_out)
        if verdict["admitted"]:
            admitted.append(int(task_id))
            continue
        for reason in verdict["reasons"] or ["unknown"]:
            rejected[str(reason)] += 1
    return admitted, rejected


def load_searcher(index_path: Path):
    """冻结的 BM25 检索器（懒导入：只在真正要算可达性时才付这个代价）。

    索引不存在时**就地建一次**（约 1 分钟），与 `audit_retrieval_reachability.py`
    的做法一致——服务器上索引是 gitignored 的，报错退出会让这条命令没法一键跑。
    """
    from web_agent_site.engine.search import MultiFieldBM25Searcher, build_index

    index_path = Path(index_path)
    if not index_path.is_file():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(audit.DEFAULT_PRODUCTS, "rt", encoding="utf-8") as handle:
            products = json.load(handle)
        build_index(iter(products), index_path, product_data_sha256="local-audit")
    return MultiFieldBM25Searcher(index_path)


def gold_rank(
    searcher,
    goal: Mapping[str, object],
    depth: int = REACH_DEPTH,
    keys: Sequence[str] = REACH_QUERY_KEYS,
) -> int | None:
    """gold 在这道题的自然语言查询里能到的最好名次；None = 前 depth 名内从未出现。

    只用 Q1/Q2/Q4（整句题面 / 前 12 字 / 品类+属性）。**故意不用 Q3「只有品类」**：
    品类查询几乎必然命中同品类商品，拿它做可达性判断等于永远返回「可达」，是假严谨。
    """
    gold = str(goal.get("asin"))
    queries = audit.retrieval_queries(dict(goal))
    best: int | None = None
    for key in keys:
        query = queries.get(key)
        if not query:
            continue
        for hit in searcher.search(query, k=depth):
            if hit.asin == gold:
                best = hit.rank if best is None else min(best, hit.rank)
                break
    return best


def rank_profile(ranks: Iterable[int | None]) -> dict[str, object]:
    values = list(ranks)
    found = sorted(rank for rank in values if rank)
    return {
        "n": len(values),
        "unreachable": sum(1 for rank in values if rank is None),
        "median": found[len(found) // 2] if found else None,
        "p90": found[int(len(found) * 0.9)] if found else None,
        "max": max(found) if found else None,
    }


def select_reachable_dev(
    candidates: Sequence[int],
    goals: Sequence[Mapping[str, object]],
    reference: Mapping[int, int],
    size: int,
    seed: int,
    *,
    searcher=None,
    max_rounds: int = 8,
) -> tuple[list[int], dict[int, int], dict[int, int], dict[int, int | None], int]:
    """分层抽样 + 检索可达性过滤：不可达的题**换掉**，不是留着当死题。

    Final-200 里 20 道无解题的教训：准入契约（A1–A4）管的是「买对了能不能得分」，
    管不到「能不能搜到」。可达性单独查，否则 dev 集自带死题，分数又被拉低。
    """
    blocked: set[int] = set()
    rank_cache: dict[int, int | None] = {}
    dropped = 0
    for _ in range(max_rounds):
        pool = [task_id for task_id in candidates if task_id not in blocked]
        chosen, targets, actual = select_dev(pool, goals, reference, size, seed)
        if searcher is None:
            return chosen, targets, actual, rank_cache, dropped
        unreachable: list[int] = []
        for task_id in chosen:
            if task_id not in rank_cache:
                rank_cache[task_id] = gold_rank(searcher, goals[task_id])
            if rank_cache[task_id] is None:
                unreachable.append(task_id)
        if not unreachable:
            return chosen, targets, actual, rank_cache, dropped
        blocked |= set(unreachable)
        dropped += len(unreachable)
    raise SystemExit(
        f"连换 {max_rounds} 轮仍有不可达的题：候选池只有 {len(candidates)} 道，"
        "要么扩候选、要么降 --size。"
    )


def build_dev(args: argparse.Namespace) -> int:
    pools, missing = collect_pools(args.pool, allow_missing=args.allow_missing_pools)
    used = used_ids(pools)

    products = audit.load_products(args.products)
    goals = audit.build_goals(products)
    product_by_asin = {str(product["asin"]): product for product in products}

    reference_ids = task_ids_of(args.reference)
    reference = strata_distribution(reference_ids, goals)

    unused = [task_id for task_id in range(len(goals)) if task_id not in used]
    admitted, rejected = admit_all(unused, goals, product_by_asin, used)

    searcher = None if args.no_reachability else load_searcher(args.index)
    chosen, targets, actual, rank_cache, dropped = select_reachable_dev(
        admitted, goals, reference, args.size, args.seed, searcher=searcher
    )
    dev_profile = None
    ref_profile = None
    if searcher is not None:
        dev_profile = rank_profile(rank_cache.get(task_id) for task_id in chosen)
        ref_profile = rank_profile(gold_rank(searcher, goals[task_id]) for task_id in reference_ids)

    if len(chosen) != args.size:
        raise SystemExit(
            f"候选不足：需要 {args.size} 道，只选到 {len(chosen)} 道。"
            f"可用候选 {len(admitted)} 道（未用 {len(unused)} 道）。"
        )

    overlap = {
        name: len(set(ids) & set(chosen)) for name, ids in pools.items()
    }
    dirty = {name: count for name, count in overlap.items() if count}
    if dirty:
        raise SystemExit(f"零重叠校验失败：{dirty}")

    out_dir = Path(args.out_dir)
    tasks_path = out_dir / "tasks.jsonl"
    write_jsonl(tasks_path, [{"task_id": task_id} for task_id in chosen])

    metadata = OrderedDict(
        [
            ("schema_version", DEV_SCHEMA),
            ("name", out_dir.name),
            ("purpose", "内部尺子：调参/选题/选检查点只看这里；Final-200 只在里程碑跑一次"),
            ("contract", CONTRACT_VERSION),
            ("environment", "shopsimulator-environment-v2.1"),
            ("tasks", len(chosen)),
            ("sha256", sha256_lf(tasks_path)),
            ("seed", args.seed),
            ("strata_axis", STRATA_AXIS),
            ("strata_reference", {str(k): v for k, v in reference.items()}),
            ("strata_target", {str(k): v for k, v in targets.items()}),
            ("strata_actual", {str(k): v for k, v in actual.items()}),
            ("reference", {"path": str(args.reference), "tasks": len(reference_ids)}),
            ("pools", {name: len(ids) for name, ids in pools.items()}),
            ("pools_missing", missing),
            ("overlap", overlap),
            ("candidates", {
                "unused": len(unused),
                "admitted": len(admitted),
                "rejected": {k: v for k, v in sorted(rejected.items())},
            }),
            ("reachability", {
                "checked": bool(searcher),
                "depth": REACH_DEPTH,
                "index": str(args.index) if searcher else None,
                "query_keys": list(REACH_QUERY_KEYS),
                "dropped_unreachable": dropped,
                "dev": dev_profile,
                "reference": ref_profile,
            }),
            ("provisional", bool(missing)),
            ("task_ids", chosen),
        ]
    )
    write_json(out_dir / "metadata.json", metadata)

    _print_dev_report(metadata, tasks_path)
    return 0


def build_selftest(args: argparse.Namespace) -> int:
    pool_path = Path(args.pool_path)
    if not pool_path.exists():
        raise SystemExit(f"训练池不存在：{pool_path}（加 --allow-missing-pools 只用于本地演练）")
    pool_ids = sorted({int(row["task_id"]) for row in read_jsonl(pool_path)})
    if args.size > len(pool_ids):
        raise SystemExit(f"--size {args.size} 超过池子大小 {len(pool_ids)}")

    sample = list(pool_ids)
    random.Random(args.seed).shuffle(sample)
    chosen = sorted(sample[: args.size])

    out_dir = Path(args.out_dir)
    tasks_path = out_dir / "tasks.jsonl"
    write_jsonl(tasks_path, [{"task_id": task_id} for task_id in chosen])

    metadata = OrderedDict(
        [
            ("schema_version", PROBE_SCHEMA),
            ("name", out_dir.name),
            ("purpose", "训练集自测：模型对自己训练过的题能做对多少"),
            ("environment", "shopsimulator-environment-v2.1"),
            ("tasks", len(chosen)),
            ("sha256", sha256_lf(tasks_path)),
            ("seed", args.seed),
            ("pool", {"path": str(pool_path), "tasks": len(pool_ids), "sha256": sha256_lf(pool_path)}),
            ("sampling", "均匀随机（不打散池内难度构成）"),
            ("branch_rules", {
                "ge_0.90": "泛化问题 ⇒ 加多样性，别加量",
                "le_0.70": "没学会 ⇒ 加量/加 epoch 先于新场景",
                "between": "灰区 ⇒ 扩样本再判",
            }),
            ("task_ids", chosen),
        ]
    )
    write_json(out_dir / "metadata.json", metadata)

    print(f"训练池 {pool_path}：{len(pool_ids)} 道 → 抽 {len(chosen)} 道")
    print(f"写入 {tasks_path}（sha256 {metadata['sha256'][:16]}…）")
    print(f"评测：BENCHMARK={tasks_path} bash scripts/evaluate.sh {out_dir.name}")
    return 0


def _print_dev_report(metadata: Mapping[str, object], tasks_path: Path) -> None:
    print(f"# dev 集：{metadata['tasks']} 道 → {tasks_path}")
    print(f"sha256 {metadata['sha256']}（seed {metadata['seed']}）")
    print()
    print("| 硬约束轴数 | Final-200 | 配额 | 实选 |")
    print("|---|---:|---:|---:|")
    ref = metadata["strata_reference"]  # type: ignore[index]
    tgt = metadata["strata_target"]  # type: ignore[index]
    act = metadata["strata_actual"]  # type: ignore[index]
    for axis in sorted(set(ref) | set(tgt) | set(act), key=int):
        print(f"| {axis} | {ref.get(axis, 0)} | {tgt.get(axis, 0)} | {act.get(axis, 0)} |")
    print()
    print("| 池 | 道数 | 与 dev 重叠 |")
    print("|---|---:|---:|")
    for name, count in metadata["pools"].items():  # type: ignore[union-attr]
        print(f"| `{name}` | {count} | {metadata['overlap'][name]} |")
    print()
    cand = metadata["candidates"]  # type: ignore[index]
    print(f"未用候选 {cand['unused']} 道 → 过契约 {cand['admitted']} 道；拒因 {cand['rejected'] or '无'}")
    reach = metadata["reachability"]  # type: ignore[index]
    if reach["checked"]:
        dev_p = reach["dev"] or {}
        ref_p = reach["reference"] or {}
        print(
            f"检索可达性（top-{reach['depth']}，查询用 {'/'.join(reach['query_keys'])}）："
            f"换掉 {reach['dropped_unreachable']} 道搜不到的题"
        )
        print()
        print("| 集合 | 搜不到 | gold 名次中位 | p90 | 最差 |")
        print("|---|---:|---:|---:|---:|")
        for label, profile in (("Final-200", ref_p), ("dev", dev_p)):
            print(
                f"| {label} | {profile.get('unreachable')} | {profile.get('median')} | "
                f"{profile.get('p90')} | {profile.get('max')} |"
            )
    else:
        print("⚠️ 未做检索可达性过滤（--no-reachability），可能混入「搜不到」的死题。")
    if metadata.get("provisional"):
        print()
        print("⚠️ provisional=true：有池文件缺失，这份只能当演练，不许用来做决策。")


# ------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dev = sub.add_parser("dev", help="造 dev 集（零重叠、过准入契约、分层同构 Final-200）")
    dev.add_argument("--out-dir", type=Path, required=True)
    dev.add_argument("--size", type=int, default=300)
    dev.add_argument("--seed", type=int, default=20261025)
    dev.add_argument("--products", type=Path, default=audit.DEFAULT_PRODUCTS)
    dev.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    dev.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    dev.add_argument(
        "--no-reachability",
        action="store_true",
        help="跳过 BM25 可达性过滤（快，但会混入搜不到的死题）",
    )
    dev.add_argument(
        "--pool",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="覆盖默认池清单，可重复",
    )
    dev.add_argument(
        "--allow-missing-pools",
        action="store_true",
        help="池文件缺失也继续（本地演练用，产物会标 provisional=true）",
    )
    dev.set_defaults(func=build_dev)

    selftest = sub.add_parser("selftest", help="造训练集自测卷（从训练池均匀抽样）")
    selftest.add_argument("--out-dir", type=Path, required=True)
    selftest.add_argument("--size", type=int, default=300)
    selftest.add_argument("--seed", type=int, default=20261025)
    selftest.add_argument("--pool-path", type=Path, default=DEFAULT_SELFTEST_POOL)
    selftest.add_argument("--allow-missing-pools", action="store_true")
    selftest.set_defaults(func=build_selftest)
    return parser


def parse_pool_specs(values: Sequence[str] | None) -> tuple[tuple[str, Path], ...]:
    if not values:
        return DEFAULT_POOLS
    specs: list[tuple[str, Path]] = []
    for value in values:
        name, _, raw = value.partition("=")
        if not raw:
            raise SystemExit(f"--pool 需要 NAME=PATH 形式，收到 {value!r}")
        specs.append((name, Path(raw)))
    return tuple(specs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "dev":
        args.pool = parse_pool_specs(args.pool)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
