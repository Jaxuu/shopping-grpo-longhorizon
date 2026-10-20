#!/usr/bin/env python3
"""M10 A 桶：用脚本生成「翻页才找到」的轨迹（不碰老师、不碰采集脚本）。

为什么不能用采集
----------------
Step 2 实测：老师**自己从不翻页**（6 题 gold 在 21–99 名、翻页 0 次）。
池子里翻页示范只有 `next_page` 的 11 行 = 0.9%，而 A 桶有 20 道题需要它。
老师不做的动作，自然采样采不到（`docs/lessons-learned.md` §2.7a）。

为什么脚本生成是干净的
----------------------
训练口径是 `tool-call-only`：助手说的自然语言**本来就被清空、不参与梯度**，
只有工具调用参与。所以「谁来写查询词、谁决定翻页」不影响训练目标 ——
观测是环境真实返回的，动作是合法动作，轨迹就是一条合法轨迹。

**不修改任何现有文件**：只读地复用 `select_m10_collection_targets` 的选题逻辑、
`shopping_grpo` 的动作转换与行构造。原有采集/评测链路一行不动。

两步走
------
1. ``--mode plan``（纯 CPU，不连环境）：在全语料里找 gold 名次 **21–120** 的题，
   记下把 gold 排到该名次的那个查询词，写 ``data/collection/m10_paging_targets.jsonl``。
2. ``--mode generate``（需要环境服务）：对每道题执行
   ``search → next_page×k → open → select_option×m → buy_now``，
   逐步校验观测里真的出现了 gold，最后校验终局是 `gold_purchase`。

用法::

    .venv/Scripts/python.exe scripts/generate_m10_paging.py --mode plan
    .venv/Scripts/python.exe scripts/generate_m10_paging.py --mode generate --limit 3
    .venv/Scripts/python.exe scripts/generate_m10_paging.py --mode generate
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from diagnose_m10_d_bucket import norm  # noqa: E402

CORPUS = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
DEFAULT_INDEX = ROOT / ".workbuddy-ai/cache/products_local.sqlite3"
DEFAULT_TARGETS = ROOT / "data/collection/m10_paging_targets.jsonl"
DEFAULT_OUT = ROOT / "outputs/m10-paging"

#: 与 `environment.actions.product_ids` 逐字同源（`^\d+\|(\d{8,12})\|`）。
#: 这里若放宽，就会出现「脚本以为看见了 gold、点击闸门却判它不可点」的分歧。
SEARCH_ROW = re.compile(r"^(?P<rank>\d+)\|(?P<asin>\d{8,12})\|", re.MULTILINE)
AVAILABLE_OPTIONS = re.compile(r"^available_options: (?P<payload>.+)$", re.MULTILINE)
PAGE_MARKER = "page_type: search_results"

#: 第一页只渲染前 20 名。评测 A 桶那 18 道可达题里 gold 最远在第 114 名，
#: 所以上限要盖到 120（= 6 页），否则示范只教到「翻两页」，深题照样放弃。
MIN_RANK = 21
MAX_RANK = 120
MAX_PAGES = 6


def pages_needed(rank: int, *, page_size: int = 20) -> int:
    """gold 在第 rank 名时，从第一页开始要翻几页才能看见它。"""
    return max(1, -(-int(rank) // int(page_size)))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def in_paging_band(
    rank: int | None, *, min_rank: int = MIN_RANK, max_rank: int = MAX_RANK
) -> bool:
    """gold 在 min_rank–max_rank 名：第一页看不见，翻页可达。"""
    return rank is not None and min_rank <= rank <= max_rank


def best_candidate(candidates: list[tuple[str, int | None]]) -> tuple[str, int] | None:
    """在多个查询里挑把 gold 排得最靠前的那个（名次相同取先出现的）。"""
    ranked = [(query, rank) for query, rank in candidates if rank is not None]
    if not ranked:
        return None
    query, rank = min(ranked, key=lambda item: item[1])
    return query, int(rank)


def page_asins(observation: str) -> list[str]:
    """搜索结果页里渲染出来的 ASIN，按出现顺序。非搜索页返回空表。"""
    if PAGE_MARKER not in observation:
        return []
    return [match.group("asin").strip() for match in SEARCH_ROW.finditer(observation)]


def page_contains(observation: str, asin: str) -> bool:
    return asin in page_asins(observation)


def available_options(observation: str) -> dict[str, list[str]]:
    match = AVAILABLE_OPTIONS.search(observation)
    if not match:
        return {}
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(axis): [str(v) for v in values]
        for axis, values in payload.items()
        if isinstance(values, list)
    }


def match_option(required: str, options: dict[str, list[str]]) -> str | None:
    """把「要求的规格值」对到页面上真实存在的那个字符串（大小写/斜杠/空格归一后比）。"""
    target = norm(required)
    for values in options.values():
        for value in values:
            if norm(value) == target:
                return value
    return None


def tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def assemble_messages(system_prompt: str, instruction: str, executed: list[dict]) -> list[dict]:
    """把逐步执行记录拼成对话消息（与采集链路同构）。"""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    for item in executed:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [item["tool_call"]],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": item["tool_call"]["id"],
                "name": item["tool_name"],
                "content": item["observation"],
            }
        )
    return messages


def build_trajectory(
    *,
    trajectory_id: str,
    task_id: int,
    instruction: str,
    system_prompt: str,
    executed: list[dict],
    terminal: dict,
) -> dict:
    """拼出与 `raw.jsonl` 同构的轨迹字典，供 `acceptance_reasons` / `build_sft_row` 消费。"""
    steps = [
        {
            "step_index": index,
            "tool_call": item["tool_call"],
            "tool_name": item["tool_name"],
            "parameters": item["parameters"],
            "env_action": item["env_action"],
            "observation": item["observation"],
            "reward": item["reward"],
            "done": item["done"],
        }
        for index, item in enumerate(executed)
    ]
    return {
        "trajectory_id": trajectory_id,
        "task_id": int(task_id),
        "messages": assemble_messages(system_prompt, instruction, executed),
        "steps": steps,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": terminal.get("reward_detail") or {},
            "purchase": terminal.get("purchase") or {},
            "goal": terminal.get("goal") or {},
            "termination_reason": terminal.get("termination_reason"),
            "reward_valid": terminal.get("reward_valid"),
        },
        "status": "done",
        "done": True,
        "error": None,
    }


def plan_targets(
    *, limit: int, index: Path, min_rank: int = MIN_RANK, max_rank: int = MAX_RANK
) -> list[dict]:
    """离线选题：gold 名次 21–60 的未使用商品。"""
    import select_m10_collection_targets as selection  # noqa: PLC0415

    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    used = selection.load_used_ids()
    searcher = selection.load_searcher(index)
    print(f"语料 {len(products)} 条；已用 {len(used)} 条；剩余 {len(products) - len(used)} 条")

    chosen: list[dict] = []
    histogram: collections.Counter = collections.Counter()
    for task_id, item in enumerate(products):
        if task_id in used:
            continue
        instructions = item.get("instructions") or []
        if not instructions:
            continue
        instruction = instructions[0]
        if not (instruction.get("attributes") or []):
            continue
        asin = str(item.get("asin"))
        candidates = [
            (query, _rank_of(searcher, query, asin))
            for query in selection.build_queries(item, instruction)
        ]
        picked = best_candidate(candidates)
        if picked is None:
            histogram["miss"] += 1
            continue
        query, rank = picked
        band = in_paging_band(rank, min_rank=min_rank, max_rank=max_rank)
        histogram[f"{min_rank}-{max_rank}" if band else "out"] += 1
        if not band:
            continue
        chosen.append(
            {
                "task_id": int(task_id),
                "asin": asin,
                "query": query,
                "rank": int(rank),
                "pages": pages_needed(rank),
                "instruction": str(instruction.get("instruction") or ""),
            }
        )
        if limit and len(chosen) >= limit:
            break
    print(f"扫描统计 {dict(histogram)}；选出 {len(chosen)} 条")
    return chosen


def _rank_of(searcher, query: str, asin: str) -> int | None:
    try:
        hits = searcher.search(query, k=150)
    except Exception:  # noqa: BLE001 - 空查询会抛，跳过
        return None
    for hit in hits:
        if str(hit.asin) == asin:
            return int(hit.rank)
    return None


def generate(targets: list[dict], *, base_url: str, out_dir: Path, max_pages: int) -> dict:
    """连环境跑脚本化轨迹。逐题校验：翻页后真的看见 gold、终局真的是 gold_purchase。"""
    from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row  # noqa: PLC0415
    from shopping_grpo.environment.client import ShopAgentEnv  # noqa: PLC0415
    from shopping_grpo.environment.observation import render_structured_observation  # noqa: PLC0415
    from shopping_grpo.environment.tools import tool_call_to_action  # noqa: PLC0415
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    kept_path = out_dir / "kept.jsonl"
    rejections: list[dict] = []
    raw_rows: list[dict] = []
    kept_rows: list[dict] = []

    for target in targets:
        task_id = int(target["task_id"])
        gold = str(target["asin"])
        executed: list[dict] = []
        terminal: dict = {}
        failure: str | None = None

        def run(env, name: str, arguments: dict) -> dict:
            call = tool_call(name, arguments, f"c{len(executed)}")
            action = tool_call_to_action(name, arguments)
            result = env.step(action)
            if result.get("observation_state") is not None:
                observation = render_structured_observation(result["observation_state"])
            else:
                observation = result.get("instruction", result.get("observation", ""))
            executed.append(
                {
                    "tool_call": call,
                    "tool_name": name,
                    "parameters": arguments,
                    "env_action": action,
                    "observation": observation,
                    "reward": float(result.get("reward", 0.0)),
                    "done": bool(result.get("done", False)),
                }
            )
            return result

        with ShopAgentEnv(base_url=base_url) as env:
            env.include_trace_target = True
            initial = env.reset(task_id)
            instruction = str(initial.get("instruction") or "")
            goal_options = [str(v) for v in initial.get("goal_options") or []]

            result = run(env, "search_products", {"query": target["query"]})
            observation = executed[-1]["observation"]
            if page_contains(observation, gold):
                failure = "gold_already_on_page_1"
            pages = 1
            while failure is None and not page_contains(observation, gold):
                if pages >= max_pages:
                    failure = "gold_not_found_within_max_pages"
                    break
                result = run(env, "next_page", {})
                observation = executed[-1]["observation"]
                pages += 1
                if not page_asins(observation):
                    failure = "no_more_pages"
                    break

            if failure is None:
                result = run(env, "open_product", {"asin": gold})
                observation = executed[-1]["observation"]
                options = available_options(observation)
                if not options:
                    failure = "detail_page_without_options"
                for required in goal_options:
                    if failure is not None:
                        break
                    actual = match_option(required, options)
                    if actual is None:
                        failure = f"required_option_missing:{required}"
                        break
                    result = run(env, "select_option", {"value": actual})

            if failure is None:
                terminal = run(env, "buy_now", {})
                detail = terminal.get("reward_detail") or {}
                if detail.get("reward_type") != "gold_purchase":
                    failure = f"not_gold:{detail.get('reward_type')}"

        if failure is not None:
            rejections.append({"task_id": task_id, "reason": failure})
            continue

        trajectory = build_trajectory(
            trajectory_id=f"m10paging-{task_id}-0",
            task_id=task_id,
            instruction=instruction,
            system_prompt=SYSTEM_PROMPT,
            executed=executed,
            terminal=terminal,
        )
        ok, reasons = acceptance_reasons(trajectory)
        raw_rows.append(trajectory)
        if not ok:
            rejections.append({"task_id": task_id, "reason": "acceptance:" + ",".join(reasons)})
            continue
        kept_rows.append(build_sft_row(trajectory))
        print(
            f"  [ok] task {task_id} 名次 {target['rank']} → 翻 {len([e for e in executed if e['tool_name']=='next_page'])} 页"
            f"，{len(executed)} 步"
        )

    write_jsonl(raw_path, raw_rows)
    write_jsonl(kept_path, kept_rows)
    summary = {
        "targets": len(targets),
        "raw": len(raw_rows),
        "kept": len(kept_rows),
        "rejected": len(rejections),
        "reject_reasons": dict(collections.Counter(r["reason"].split(":")[0] for r in rejections)),
        "raw_path": str(raw_path),
        "kept_path": str(kept_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps({**summary, "rejections": rejections}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("plan", "generate"), default="plan")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--limit", type=int, default=0, help="plan: 最多选多少题；generate: 最多跑多少题")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--min-rank", type=int, default=MIN_RANK)
    parser.add_argument("--max-rank", type=int, default=MAX_RANK)
    args = parser.parse_args(argv)

    if args.mode == "plan":
        if args.max_rank < args.min_rank:
            raise SystemExit("--max-rank 不能小于 --min-rank")
        chosen = plan_targets(
            limit=args.limit,
            index=args.index,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
        )
        write_jsonl(args.targets, chosen)
        ranks = collections.Counter(row["rank"] // 10 * 10 for row in chosen)
        pages = collections.Counter(row["pages"] for row in chosen)
        print(f"写入 {args.targets}：{len(chosen)} 题")
        print(f"名次分布（按 10 分段）：{dict(sorted(ranks.items()))}")
        print(f"需翻页数分布：{dict(sorted(pages.items()))}")
        return 0

    if not args.targets.is_file():
        raise SystemExit(f"没有 {args.targets}；先跑 --mode plan")
    targets = read_jsonl(args.targets)
    if args.limit:
        targets = targets[: args.limit]
    deepest = max((pages_needed(int(t["rank"])) for t in targets), default=1)
    if deepest > args.max_pages:
        raise SystemExit(
            f"目标里有 gold 在第 {deepest * 20} 名之后的题，需要翻 {deepest} 页，"
            f"但 --max-pages={args.max_pages}；要么加 --max-pages，要么重跑 plan 收窄 --max-rank"
        )
    summary = generate(
        targets, base_url=args.base_url, out_dir=args.out_dir, max_pages=args.max_pages
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
