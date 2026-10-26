"""纯 CPU 的确定性候选重排：不调用任何模型，只按题面约束给候选打分。

要解决的问题：多数票 129/200，目标 140。差的分落在「正确答案已在票数前 2 名、
但不知道该信谁」的那十几道题上。让模型去比较需要显卡；这里用一条可解释的规则
在同样的候选里挑，**零显卡、零训练、可复现**。

打分只由三块组成：
1. 题面与商品页的字面重合（越长越具体，权重按 (长度-1)^2 累加）；
2. 预算硬约束（商品最低价已经超过预算上限 ⇒ 直接判死）；
3. 品牌硬约束（题面说了「亿力牌」，候选品牌里没有「亿力」⇒ 判死）。

**只用 `instruction` 与模型自己看过的商品详情页。**
绝不读 `terminal_result.goal` / `initial_result.goal_options`——那是答案。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 预登记门槛：14 道「正确答案在票数前 2 名」的题里判对 11 道，总分才到 140。
MIN_HITS = 11
TARGET_RATE = 0.70

OVER_BUDGET_PENALTY = 200.0
BRAND_MISMATCH_PENALTY = 100.0
MAX_MATCH_LEN = 8
MIN_MATCH_LEN = 2

_FIELD_KEYS = ("asin", "title", "brand", "category", "price", "key_attributes", "available_options")

_BUDGET_LIMITED = re.compile(r"(\d+(?:\.\d+)?)\s*元\s*(以内|以下|内|左右|上下|出头|封顶)")
_BUDGET_CONTEXT = re.compile(
    r"(?:预算|价格|价位|控制在|不超过)[^。；，]{0,10}?(\d+(?:\.\d+)?)\s*元"
)
_BUDGET_ANY = re.compile(r"(\d+(?:\.\d+)?)\s*元")


def _load_pw():
    spec = importlib.util.spec_from_file_location(
        "eval_pairwise_selector", ROOT / "scripts" / "eval_pairwise_selector.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pw = _load_pw()


def parse_detail(text: str) -> dict[str, str]:
    """从一条 observation 里取商品字段。取不到 title 就当作不是详情页。"""
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in _FIELD_KEYS:
            fields[key] = value.strip()
    return fields if fields.get("title") else {}


def parse_price(field: str) -> tuple[float, float] | None:
    """`5.0 to 18.0` → (5.0, 18.0)；`27.0` → (27.0, 27.0)。"""
    if not field:
        return None
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", field)]
    if not numbers:
        return None
    return min(numbers), max(numbers)


def parse_budget(instruction: str) -> tuple[float, float] | None:
    """从题面取预算区间。`20元左右` → (0, 30)；`400元以内` → (0, 400)。"""
    if not instruction:
        return None
    match = _BUDGET_LIMITED.search(instruction)
    if match:
        amount = float(match.group(1))
        soft = match.group(2) in {"左右", "上下", "出头"}
        return 0.0, amount * 1.5 if soft else amount
    match = _BUDGET_CONTEXT.search(instruction)
    if match:
        return 0.0, float(match.group(1))
    match = _BUDGET_ANY.search(instruction)
    if match:
        return 0.0, float(match.group(1))
    return None


_BRAND_NOISE = ("旗舰店", "专营店", "专卖店", "官方", "商城", "店铺", "店")


def brand_core(brand: str) -> str:
    """`漫步喜旗舰店` → `漫步喜`。店名后缀对品牌判断没意义，先剥掉。"""
    core = brand or ""
    for suffix in _BRAND_NOISE:
        core = core.replace(suffix, "")
    return core.strip()


def brand_mentioned(brand: str, instruction: str) -> bool:
    """候选的品牌名有没有出现在题面里。

    **不做「从题面里抽品牌」**——中文没有分词边界，正则会把「想买亿力品牌的」
    抽成「想买亿力」。反过来拿候选自己的品牌去题面里找，就没有这个问题。
    按前缀从长到短试，`荷兰品牌店` 能被题面里的「荷兰」认出来。
    """
    core = brand_core(brand)
    for length in range(len(core), 1, -1):
        if core[:length] in (instruction or ""):
            return True
    return False


def option_values(field: str) -> list[str]:
    """`available_options` 是 JSON：`{"颜色分类": ["a", "b"]}` → ["a", "b"]。"""
    if not field:
        return []
    try:
        payload = json.loads(field)
    except (ValueError, TypeError):
        return []
    values: list[str] = []
    for item in payload.values() if isinstance(payload, Mapping) else []:
        if isinstance(item, list):
            values.extend(str(entry) for entry in item)
        elif item is not None:
            values.append(str(item))
    return values


def searchable_text(detail: Mapping[str, str], options: Mapping[str, str]) -> str:
    """候选的可比文本：标题、品牌、品类、属性、可选规格、本次已选规格。"""
    parts = [
        detail.get("title", ""),
        detail.get("brand", ""),
        detail.get("category", ""),
        detail.get("key_attributes", ""),
    ]
    parts.extend(option_values(detail.get("available_options", "")))
    parts.extend(str(value) for value in options.values())
    return " ".join(part for part in parts if part)


def overlap_score(text: str, instruction: str) -> tuple[float, list[str]]:
    """贪心最长匹配：从左到右，每处取能匹配上的最长片段，按 (长度-1)^2 累加。

    用平方是为了让「卡通-永结同心」这种 6 字精确命中远大于「礼盒」这种通用词，
    而通用词两个候选都会有，差值里自然抵消。
    """
    total = 0.0
    pieces: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        matched = 0
        for length in range(min(MAX_MATCH_LEN, size - index), MIN_MATCH_LEN - 1, -1):
            if text[index : index + length] in instruction:
                matched = length
                break
        if matched:
            total += (matched - 1) ** 2
            pieces.append(text[index : index + matched])
            index += matched
        else:
            index += 1
    return total, pieces


def score_candidates(
    instruction: str,
    candidates: Sequence[Mapping],
) -> list[dict]:
    """给每个候选打分。candidates 每项含 `detail`（字段字典）与 `options`（已选规格）。"""
    budget = parse_budget(instruction)
    mentioned = [
        brand_mentioned(candidate["detail"].get("brand", ""), instruction)
        for candidate in candidates
    ]
    # 只有当「有候选的品牌被题面点名、又不是全都被点名」时，品牌才是有区分度的信息。
    brand_discriminates = any(mentioned) and not all(mentioned)
    scored = []
    for index, candidate in enumerate(candidates):
        detail = candidate["detail"]
        options = candidate.get("options") or {}
        text = searchable_text(detail, options)
        overlap, pieces = overlap_score(text, instruction)
        score = overlap
        reasons = []
        price = parse_price(detail.get("price", ""))
        if budget and price and price[0] > budget[1]:
            score -= OVER_BUDGET_PENALTY
            reasons.append(f"超预算：最低价 {price[0]:g} > 预算上限 {budget[1]:g}")
        if brand_discriminates and not mentioned[index]:
            score -= BRAND_MISMATCH_PENALTY
            reasons.append(
                f"品牌不符：题面点名了别的品牌，这家是「{brand_core(detail.get('brand', ''))}」"
            )
        scored.append(
            {
                "score": round(score, 2),
                "overlap": round(overlap, 2),
                "pieces": pieces[:12],
                "reasons": reasons,
                "price": price,
            }
        )
    return scored


def select(candidates: Sequence[Mapping], scored: Sequence[Mapping]) -> int:
    """分数最高者胜。**同分取票数高的**（candidates 已按票数降序），保持确定性。"""
    best = 0
    for index in range(1, len(scored)):
        if scored[index]["score"] > scored[best]["score"]:
            best = index
    return best


def run(
    tasks: Sequence[Mapping], *, details: Mapping, instructions: Mapping, top_k: int
) -> list[dict]:
    results = []
    for task in tasks:
        task_id = task["task_id"]
        instruction = instructions.get(task_id)
        if not instruction:
            continue
        candidates = []
        missing = False
        for key in task["candidates"][:top_k]:
            asin, options = pw.split_identity(key)
            detail_text = details.get((task_id, asin))
            if detail_text is None:
                missing = True
                break
            candidates.append(
                {"key": key, "asin": asin, "options": options, "detail": parse_detail(detail_text)}
            )
        if missing or len(candidates) < 2:
            continue
        scored = score_candidates(instruction, candidates)
        index = select(candidates, scored)
        picked = candidates[index]["key"]
        results.append(
            {
                "task_id": task_id,
                "picked": picked,
                "hit": picked in task["good_keys"],
                "plurality_wrong": task.get("plurality_wrong", True),
                "candidates": [
                    {"key": item["key"], **score}
                    for item, score in zip(candidates, scored, strict=False)
                ],
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--headroom", type=Path, default=ROOT / "outputs" / "analysis" / "headroom.json"
    )
    parser.add_argument("--reference", type=Path, default=pw.ens.DEFAULT_REFERENCE)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--scope", choices=("disputed", "all"), default="disputed")
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    headroom = json.loads(args.headroom.read_text(encoding="utf-8"))
    reference_set = set(pw.ens.probe.task_ids_of(args.reference))
    runs = pw.ens.resolve_runs(args.runs, min_tasks=args.min_tasks)
    for label in list(runs):
        ids = {int(row["task_id"]) for row in runs[label] if row.get("task_id") is not None}
        if len(ids & reference_set) / max(1, len(reference_set)) < args.min_overlap:
            del runs[label]
    if not runs:
        raise SystemExit("没有可用的评测目录。")

    tasks = pw.disputed_tasks(headroom, top_k=args.top_k)
    if args.scope == "all":
        tasks = pw.all_tasks(headroom, top_k=args.top_k)
    if not tasks:
        raise SystemExit("headroom.json 里没有符合条件的题（先跑 analyze_ensemble_headroom.py）。")

    print(f"# 规则重排（纯 CPU，不调模型）：{len(tasks)} 题，取票数前 {args.top_k} 名")
    print()

    results = run(
        tasks,
        details=pw.product_details(runs),
        instructions=pw.instructions(runs),
        top_k=args.top_k,
    )
    if not results:
        raise SystemExit("一题都没跑成（候选商品详情页在轨迹里找不到？）")

    if not args.quiet:
        for item in results:
            mark = "✅" if item["hit"] else "❌"
            cells = "  ".join(
                f"{chr(ord('A') + i)}={c['key'].split('|')[0]}({c['score']:g})"
                for i, c in enumerate(item["candidates"])
            )
            print(f"{mark} task {item['task_id']}: {cells}")
            for candidate in item["candidates"]:
                if candidate["reasons"]:
                    print(
                        f"     {candidate['key'].split('|')[0]}: {'；'.join(candidate['reasons'])}"
                    )
        print()

    hit = sum(1 for item in results if item["hit"])
    n = len(results)
    print(f"## 结果：判对 {hit}/{n}（{hit / n:.1%}）")

    if args.scope == "disputed":
        rules = headroom.get("rules", {})
        plurality = next((v for k, v in rules.items() if "多数票（平票算失败）" in k), None)
        if plurality is not None:
            total = headroom["n"]
            projected = plurality + hit
            target = round(total * TARGET_RATE)
            print()
            print(
                f"多数票 {plurality}/{total} + 规则重排救回 {hit} ⇒ 投影 **{projected}/{total} "
                f"= {projected / total:.1%}**（目标 {target}）"
            )
            if hit >= MIN_HITS:
                print(f"**达标**：判对 {hit} ≥ 预登记门槛 {MIN_HITS}。")
            else:
                print(f"**未达标**：判对 {hit} < 门槛 {MIN_HITS}，这条不够。")
            print()
            print("> 注意这是**上界**：只在「多数票判错」的题上跑，没有量会不会把本来对的题改错。")
            print("> 要端到端数字，加 `--scope all`。")
    else:
        wrong_fixed = sum(1 for item in results if item["hit"] and item["plurality_wrong"])
        broken = sum(1 for item in results if not item["hit"] and not item["plurality_wrong"])
        print()
        print(f"救回 {wrong_fixed} 题，改错 {broken} 题 ⇒ 净 **{wrong_fixed - broken:+d}**")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": args.scope,
            "top_k": args.top_k,
            "n": n,
            "hit": hit,
            "tasks": results,
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
