"""纯 CPU 的规格选择器：买对商品但选错规格，能不能用规则修。

服务器实测：冠军 127/200，缺口 13 题。52 道「看到 gold 却没买对」里，
**20 道是「买对商品、规格或属性不符」**，其中 18 道 gold 就在搜索页前 3 名。
这类失败多打开几个商品完全没用——商品已经选对了，错在规格。

典型例子：gold 规格「双龙眼木板中软10mm±」，模型选了「设计定制」，
因为题面里有「DIY定制」四个字。字面上「定制」确实命中了，但它只是 2 个字，
而「双龙眼木板」是 5 个字。

所以做法：拿这个商品自己的 `available_options`，按题面重合度给每个选项打分，
每个规格轴取最高分。全是字面匹配，不调模型、不读 goal。

**只在「已经买对商品」的题上替换规格**，商品选错的那 32 道管不了。
gold 仅用于最后算命中，不参与选择。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ens = _load("eval_ensemble", ROOT / "scripts" / "eval_ensemble.py")
rs = _load("rule_selector", ROOT / "scripts" / "rule_selector.py")


def available_options(record: Mapping, asin: str) -> dict[str, list[str]]:
    """在这道题的轨迹里找该商品的详情页，取出 `available_options`。"""
    marker = f"asin: {asin}"
    for step in record.get("steps") or []:
        observation = step.get("observation") or ""
        if marker not in observation or "available_options" not in observation:
            continue
        for line in observation.splitlines():
            if not line.startswith("available_options:"):
                continue
            raw = line.split(":", 1)[1].strip()
            try:
                payload = json.loads(raw)
            except ValueError:
                continue
            if isinstance(payload, Mapping):
                return {
                    str(axis): [str(item) for item in values]
                    for axis, values in payload.items()
                    if isinstance(values, list)
                }
    return {}


def best_value(instruction: str, values: Sequence[str]) -> str | None:
    """题面重合度最高的那个选项值。同分取靠前的（列表顺序即页面顺序）。"""
    best: str | None = None
    best_score = float("-inf")
    for value in values:
        score, _ = rs.overlap_score(value, instruction)
        if score > best_score:
            best, best_score = value, score
    return best


def choose_options(
    instruction: str,
    available: Mapping[str, Sequence[str]],
    *,
    original: Mapping[str, str] | None = None,
    min_gap: float = 0.0,
    max_keep_score: float | None = None,
) -> dict[str, str]:
    """每个规格轴取题面重合度最高的值。

    给了 `original` 时进入**闸门模式**，两道闸：
    - `min_gap`：新选项要比原选项高出这么多分才换；
    - `max_keep_score`：原选项自己的分数必须低于这个值才允许换（**默认这道闸更关键**）。

    为什么需要 `max_keep_score`：实测不加闸门会把 37 道本来对的改错，
    典型是「浅蓝色【无线款】」被换成「浅蓝色」、「2米加宽加厚2条」被换成「4条」——
    模型原本选对了，规则在等价选项里乱跳。而真正该改的那几道，原选项的分数是 0
    （模型根本没在题面里找依据）。所以「原选项毫无依据才动它」比「分差够大才动它」可靠。
    """
    chosen = {}
    for axis, values in available.items():
        pick = best_value(instruction, values)
        if pick is None:
            continue
        if original is not None and axis in original:
            keep = original[axis]
            if keep not in values:
                chosen[axis] = keep
                continue
            keep_score, _ = rs.overlap_score(keep, instruction)
            if max_keep_score is not None and keep_score > max_keep_score:
                chosen[axis] = keep
                continue
            pick_score, _ = rs.overlap_score(pick, instruction)
            if pick_score - keep_score < min_gap:
                chosen[axis] = keep
                continue
        chosen[axis] = pick
    return chosen


def canon(text: str) -> str:
    """规格值的规范化形式：全半角统一 + 大小写折叠 + 空白折叠。

    **必须做这一步**：环境判 gold 时是大小写不敏感的（实测 task 2105 买的是
    「英伟达 tesla l20 48g」，gold 是「英伟达 Tesla L20 48G」，照样判成功）。
    拿精确字符串去比，会把一堆本来对的题算成「改错」，得出完全错误的结论。
    """
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def gold_satisfied(chosen: Mapping[str, str], gold_options: Sequence[str]) -> bool:
    """gold 要求的每个规格值都要被选中（按规范化形式比）。"""
    have = {canon(value) for value in chosen.values()}
    return {canon(value) for value in gold_options}.issubset(have)


def evaluate(
    per_model: Mapping[str, Mapping[int, Mapping]],
    *,
    champion: str,
    min_gap: float = 0.0,
    max_keep_score: float | None = None,
) -> dict:
    champ = per_model.get(champion, {})
    total = 0
    asin_ok = 0
    original_ok = 0
    rule_ok = 0
    fixed = 0
    broken = 0
    no_options = 0
    rows = []
    for task_id, record in sorted(champ.items()):
        total += 1
        gold = (record.get("terminal_result") or {}).get("goal") or {}
        gold_asin = gold.get("asin")
        gold_options = gold.get("goal_options") or []
        purchase = (record.get("terminal_result") or {}).get("purchase") or {}
        if not gold_asin or purchase.get("asin") != gold_asin:
            continue
        asin_ok += 1
        instruction = (record.get("initial_result") or {}).get("instruction") or ""
        was_ok = ens.is_success(record, strict=True)
        available = available_options(record, gold_asin)
        if not available:
            no_options += 1
            continue
        original = {str(k): str(v) for k, v in (purchase.get("options") or {}).items()}
        chosen = choose_options(
            instruction,
            available,
            original=original,
            min_gap=min_gap,
            max_keep_score=max_keep_score,
        )
        now_ok = gold_satisfied(chosen, gold_options)
        original_ok += 1 if was_ok else 0
        rule_ok += 1 if now_ok else 0
        if now_ok and not was_ok:
            fixed += 1
        if was_ok and not now_ok:
            broken += 1
        rows.append(
            {
                "task_id": task_id,
                "was_ok": was_ok,
                "rule_ok": now_ok,
                "gold_options": list(gold_options),
                "original": original,
                "chosen": chosen,
            }
        )
    return {
        "total": total,
        "asin_ok": asin_ok,
        "no_options": no_options,
        "original_ok": original_ok,
        "rule_ok": rule_ok,
        "fixed": fixed,
        "broken": broken,
        "tasks": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=ens.DEFAULT_REFERENCE)
    parser.add_argument("--champion", type=str, required=True)
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    parser.add_argument("--target", type=int, default=140)
    parser.add_argument(
        "--min-gap", type=float, default=0.0, help="新选项要比原选项高出多少分才替换"
    )
    parser.add_argument(
        "--max-keep-score",
        type=float,
        default=None,
        help="原选项自己的分数高于此值就不动它（只在它毫无依据时才改）",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", type=int, default=8, help="打印前几道改动的明细")
    args = parser.parse_args(argv)

    reference_set = set(ens.probe.task_ids_of(args.reference))
    runs = ens.resolve_runs(args.runs, min_tasks=args.min_tasks)
    kept = {}
    for label, rows in runs.items():
        ids = {int(row["task_id"]) for row in rows if row.get("task_id") is not None}
        if len(ids & reference_set) / max(1, len(reference_set)) >= args.min_overlap:
            kept[label] = {
                int(row["task_id"]): row for row in rows if row.get("task_id") is not None
            }
    if args.champion not in kept:
        raise SystemExit(f"冠军目录 {args.champion!r} 不在可用目录里：{sorted(kept)}")

    stats = evaluate(
        kept, champion=args.champion, min_gap=args.min_gap, max_keep_score=args.max_keep_score
    )
    score = sum(1 for row in kept[args.champion].values() if ens.is_success(row, strict=True))

    print(f"# 规格选择器（纯 CPU）：冠军 `{args.champion}`，{stats['total']} 题，现 {score}")
    print()
    print("## 闸门扫描 A：原选项的分数低于多少才允许替换")
    print()
    print("| 原选项分数上限 | 修好 | 改错 | 净 | 投影 |")
    print("|---:|---:|---:|---:|---:|")
    for cap in (0.0, 1.0, 2.0, 4.0, 9.0, 16.0):
        item = evaluate(kept, champion=args.champion, max_keep_score=cap)
        net = item["fixed"] - item["broken"]
        projected = score + net
        print(
            f"| {cap:g} | {item['fixed']} | {item['broken']} | {net:+d} | "
            f"{projected}/{item['total']} = {projected / item['total']:.1%} |"
        )
    print()
    print("## 闸门扫描 B：新选项要比原选项高出多少分")
    print()
    print("| 分差门槛 | 修好 | 改错 | 净 | 投影 |")
    print("|---:|---:|---:|---:|---:|")
    for gap in (0, 2, 4, 6, 9, 16):
        item = evaluate(kept, champion=args.champion, min_gap=float(gap))
        net = item["fixed"] - item["broken"]
        projected = score + net
        print(
            f"| {gap} | {item['fixed']} | {item['broken']} | {net:+d} | "
            f"{projected}/{item['total']} = {projected / item['total']:.1%} |"
        )
    print()
    print("| 项 | 题数 |")
    print("|---|---:|")
    print(f"| 买对了商品（asin == gold） | {stats['asin_ok']} |")
    print(f"| 其中轨迹里找不到可选规格 | {stats['no_options']} |")
    print(f"| 现在规格就对 | {stats['original_ok']} |")
    print(f"| **规则选完规格就对** | {stats['rule_ok']} |")
    print(f"| **修好** | {stats['fixed']} |")
    print(f"| **改错** | {stats['broken']} |")
    print()

    shown = 0
    for item in stats["tasks"]:
        if item["rule_ok"] == item["was_ok"] or shown >= args.show:
            continue
        mark = "✅修好" if item["rule_ok"] else "❌改错"
        print(f"{mark} task {item['task_id']}")
        print(f"     gold  : {item['gold_options']}")
        print(f"     规则选: {list(item['chosen'].values())}")
        shown += 1
    if shown:
        print()

    print("## 判读")
    print()
    print(
        f"- 净收益：修好 {stats['fixed']} − 改错 {stats['broken']} = **{stats['fixed'] - stats['broken']:+d}**"
    )
    print(
        f"- 投影：{score} + {stats['fixed'] - stats['broken']:+d} = **{projected}/{stats['total']} = {projected / stats['total']:.1%}**"
    )
    need = args.target - score
    print(f"- 到 {args.target} 需要净 +{need}")
    if stats["fixed"] - stats["broken"] >= need:
        print("- **够到 70%**：规格这一档单独就能补上缺口。")
    else:
        print(
            f"- **不够**：规格这档只能给 {stats['fixed'] - stats['broken']:+d}，还差 "
            f"{need - (stats['fixed'] - stats['broken'])} 题，得靠「买错商品」那 32 道里的 13 道。"
        )
    print()
    print("> 前提：agent 要把规则选出的规格真的填进购买请求。这一步需要改动作，离线测不了。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
