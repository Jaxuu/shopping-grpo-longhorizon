#!/usr/bin/env python3
"""拿不同写法的 query 直接打商店环境，看 gold 能不能进模型可见的前 20 名。

为什么需要它：A 组（gold 从没出现在我们的搜索结果里）看着像"query 写得不好"，
但"写得不好"至少有三种，修法完全不同：

- **词太多**：多加了一个修饰词把结果挤掉 ⇒ 简化 query 就能救，甚至不用训练；
- **漏了品类词**：用"大干皮/遮瑕"代替"粉底液" ⇒ 要教它先说品类；
- **需要世界知识**：老师用"子擎""NARS""ZIPPO"这种品牌词 ⇒ 2B 学不会，只能放弃。

本脚本对每道题跑几种简化写法 + 老师的原始 query（作为上界对照），
用环境真实返回的名次说话。**只读环境，不改任何评测产物。**

`buy_now` 只能买当前页，所以这个探测只看"搜得到搜不到"，不涉及选择策略。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 常见修饰词：颜色、材质、风格、数量。去掉它们通常能让 BM25 少几个必须命中的词。
FILLER_WORDS = (
    "黑色", "白色", "绿色", "红色", "蓝色", "紫色", "灰色", "粉色", "银色", "金色",
    "高端", "简约", "金属", "铝制", "铝合金", "不锈钢", "无油", "散热", "大干皮",
    "新款", "2025新款", "专用", "便携",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vc = _load("analyze_visible_ceiling", ROOT / "scripts" / "analyze_visible_ceiling.py")


def variants(query: str) -> dict[str, str]:
    """给一个 query 生成几种简化写法。空结果会被跳过。"""
    words = [word for word in query.split() if word]
    out: dict[str, str] = {}
    if len(words) >= 2:
        out["去尾词"] = " ".join(words[:-1])
    if len(words) >= 3:
        out["前2词"] = " ".join(words[:2])
    stripped = [word for word in words if word not in FILLER_WORDS]
    if stripped and stripped != words:
        out["去修饰词"] = " ".join(stripped)
    if len(stripped) >= 2:
        out["去修饰+前2词"] = " ".join(stripped[:2])
    return {name: value for name, value in out.items() if value}


def rank_of(observation: str, gold: str, *, visible_top_k: int = 20) -> int | None:
    """gold 在这一次搜索结果里最好的名次；只认模型真的能看到的行。"""
    best = None
    for rank, asin in vc.search_rows(observation):
        if asin != gold or rank > visible_top_k:
            continue
        if best is None or rank < best:
            best = rank
    return best


def probe(env_factory, task_id: int, queries: dict[str, str], gold: str, *, base_url: str, visible_top_k: int) -> dict:
    """对每个写法单独 reset 一次再搜，避免上一次搜索的页面状态干扰。"""
    results: dict[str, int | None] = {}
    for label, query in queries.items():
        env = env_factory(base_url=base_url)
        try:
            env.reset(task_id)
            action = _search_action(query)
            response = env.step(action)
            state = response.get("observation_state")
            observation = _render(state) if state is not None else str(response.get("observation", ""))
            results[label] = rank_of(observation, gold, visible_top_k=visible_top_k)
        finally:
            try:
                env.release()
            except Exception:
                pass
    return results


def _search_action(query: str):
    from shopping_grpo.environment.actions import tool_call_to_action

    return tool_call_to_action("search_products", {"query": query})


def _render(state) -> str:
    from shopping_grpo.environment.observation import render_structured_observation

    return render_structured_observation(state)


def build_queries(ours: Sequence[str], teacher: Sequence[str]) -> dict[str, str]:
    """要探测的写法：我们的原句 + 它的简化版 + 老师第一次的 query。"""
    base = ours[0] if ours else ""
    out: dict[str, str] = {}
    if base:
        out["我们原句"] = base
        out.update(variants(base))
    if teacher:
        out["老师第1次"] = teacher[0]
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="探测 query 写法对 gold 可见性的影响")
    parser.add_argument("--gap-json", type=Path, required=True)
    parser.add_argument("--group", default="A")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--visible-top-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gap = json.loads(args.gap_json.read_text(encoding="utf-8"))
    rows = [row for row in gap["rows"] if row["group"] == args.group and row.get("gold_asin")]
    if args.limit:
        rows = rows[: args.limit]

    from shopping_grpo.environment.client import ShopAgentEnv

    report = []
    for row in rows:
        queries = build_queries(row.get("ours_queries") or [], row.get("teacher_queries") or [])
        results = probe(
            ShopAgentEnv,
            int(row["task_id"]),
            queries,
            str(row["gold_asin"]),
            base_url=args.base_url,
            visible_top_k=args.visible_top_k,
        )
        report.append({"task_id": row["task_id"], "gold": row["gold_asin"], "queries": queries, "ranks": results})
        hit = {label: rank for label, rank in results.items() if rank is not None}
        summary = "、".join(f"{label}={rank}" for label, rank in hit.items()) or "全部没搜到"
        print(f"  task {row['task_id']}: {summary}")

    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写入 {args.output}")

    # 统计：几种简化里有没有能把 gold 拉进前 20 的。
    rescued = [item for item in report if any(rank is not None for label, rank in item["ranks"].items() if label != "老师第1次")]
    teacher_hit = [item for item in report if item["ranks"].get("老师第1次") is not None]
    print()
    print(f"共 {len(report)} 题；老师的 query 能搜到 {len(teacher_hit)} 题；"
          f"我们的简化写法能搜到 {len(rescued)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
