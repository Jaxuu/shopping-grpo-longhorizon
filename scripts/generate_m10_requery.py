#!/usr/bin/env python3
"""M10-R 换查询重搜：合成「第一次搜偏了，换个词再搜」的轨迹。

为什么补这个
------------
评测里 A 桶（20 题）的病根不是翻不动页，是**查询词选得差**：
用「好查询」算，200 道评测题里 195 道的 gold 就在第 1 页；而按模型自己
写的查询算，才有 20 道跑到 20 名外。池子里「搜过 ≥2 次」的轨迹只有
11.2%（140/1252），「换过查询词」也是 140 条 —— **规则齐备、示范稀少**。

而且环境本身在提示这条路：`exact_repeat_limit=2` 让连点同一个按钮
（翻页）最多两次，但换一个查询词是不同动作、还能拿探索进度。
换句话说，环境更希望你换词，而不是死磕同一个列表。

素材供给（抽样 3,000 道外推）
----------------------------
「差查询把 gold 甩到 60 名外、好查询拉回第 1 页」占 **49.8%**，
全语料约 **10,457 道**。要多少有多少。

合法性
------
与 `generate_m10_paging.py` 同一套理由：训练口径是 `tool-call-only`，
助手自然语言本来就被清空、不参与梯度，只有工具调用参与；观测是环境
真实返回的，动作是合法动作。**不改任何现有文件**。

两步走
------
1. ``--mode plan``（纯 CPU）：找「差查询 >60 名 且 好查询 ≤20 名」的题，
   记下两个查询词。
2. ``--mode generate``（需要环境服务）：执行
   ``search(差) → back_to_search → search(好) → open → select_option×m → buy_now``，
   逐步校验差查询里确实没有 gold、好查询里确实有，终局必须 gold_purchase。

用法::

    .venv/Scripts/python.exe scripts/generate_m10_requery.py --mode plan
    .venv/Scripts/python.exe scripts/generate_m10_requery.py --mode generate --limit 3
    .venv/Scripts/python.exe scripts/generate_m10_requery.py --mode generate
    .venv/Scripts/python.exe scripts/generate_m10_requery.py --mode generate --resume
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from generate_m10_paging import (  # noqa: E402
    CORPUS,
    DEFAULT_INDEX,
    DEFAULT_OUT,
    _rank_of,
    available_options,
    build_trajectory,
    count_lines,
    match_option,
    page_asins,
    page_contains,
    read_jsonl,
    tool_call,
    write_jsonl,
)

DEFAULT_TARGETS = ROOT / "data/collection/m10_requery_targets.jsonl"

#: 差查询必须把 gold 甩到翻页也够不到的地方（>60 名），否则「该翻页还是
#: 该换词」两种解释混在一起，教不干净。
BAD_MIN_RANK = 61
#: 好查询要把 gold 放回第 1 页，模型才有机会直接看到。
GOOD_MAX_RANK = 20


def pick_query_pair(
    ranked: list[tuple[str, int | None]],
    *,
    bad_min_rank: int = BAD_MIN_RANK,
    good_max_rank: int = GOOD_MAX_RANK,
) -> tuple[str, int, str, int | None] | None:
    """从「查询词 → gold 名次」里挑一对（好查询, 差查询）。

    返回 ``(good_query, good_rank, bad_query, bad_rank)``；凑不齐返回 None。
    ``bad_rank=None`` 表示那个查询的召回里根本没有 gold。
    差查询优先挑「确实返回了结果、只是 gold 排在后面」的——模型看到的是一屏
    像模像样的错结果，而不是「搜不到」，后者是另一种能力。
    """
    good = [(query, rank) for query, rank in ranked if rank is not None and rank <= good_max_rank]
    bad_known = [
        (query, rank) for query, rank in ranked if rank is not None and rank >= bad_min_rank
    ]
    bad_missing = [(query, None) for query, rank in ranked if rank is None]
    if not good or not (bad_known or bad_missing):
        return None
    good_query, good_rank = min(good, key=lambda item: item[1])
    if bad_known:
        bad_query, bad_rank = max(bad_known, key=lambda item: item[1])
        return good_query, int(good_rank), bad_query, int(bad_rank)
    return good_query, int(good_rank), bad_missing[0][0], None


def plan_targets(
    *,
    limit: int,
    index: Path,
    bad_min_rank: int = BAD_MIN_RANK,
    good_max_rank: int = GOOD_MAX_RANK,
    exclude_ids: set[int] | None = None,
) -> list[dict]:
    """离线选题：一个好查询 + 一个差查询，两个查询词都记下来。

    ``exclude_ids`` 用来排除别的批次已经选走的题（翻页 / 对比），
    否则同一个 task_id 会在池子里出现两次，建池时硬失败。
    """
    import select_m10_collection_targets as selection  # noqa: PLC0415

    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    used = selection.load_used_ids() | set(exclude_ids or ())
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
        queries = selection.build_queries(item, instruction)
        if len(queries) < 2:
            continue
        ranked = [(query, _rank_of(searcher, query, asin)) for query in queries]
        pair = pick_query_pair(
            ranked, bad_min_rank=bad_min_rank, good_max_rank=good_max_rank
        )
        if pair is None:
            histogram["no_pair"] += 1
            continue
        good_query, good_rank, bad_query, bad_rank = pair
        chosen.append(
            {
                "task_id": int(task_id),
                "asin": asin,
                "bad_query": bad_query,
                "bad_rank": bad_rank,
                "good_query": good_query,
                "good_rank": int(good_rank),
                "instruction": str(instruction.get("instruction") or ""),
            }
        )
        histogram["paired"] += 1
        if limit and len(chosen) >= limit:
            break
    print(f"扫描统计 {dict(histogram)}；选出 {len(chosen)} 条")
    return chosen


def generate(
    targets: list[dict],
    *,
    base_url: str,
    out_dir: Path,
    resume: bool = False,
) -> dict:
    """连环境跑「换查询重搜」的脚本化轨迹，逐题校验两次搜索结果。"""
    from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row  # noqa: PLC0415
    from shopping_grpo.environment.actions import clickable_buttons  # noqa: PLC0415
    from shopping_grpo.environment.client import ShopAgentEnv  # noqa: PLC0415
    from shopping_grpo.environment.observation import render_structured_observation  # noqa: PLC0415
    from shopping_grpo.environment.tools import tool_call_to_action  # noqa: PLC0415
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    kept_path = out_dir / "kept.jsonl"
    reject_path = out_dir / "rejections.jsonl"

    done_ids: set[int] = set()
    if resume:
        for path in (raw_path, reject_path):
            if path.is_file():
                done_ids |= {int(row["task_id"]) for row in read_jsonl(path)}
    else:
        for path in (raw_path, kept_path, reject_path):
            path.write_text("", encoding="utf-8", newline="\n")

    rejections: list[dict] = []
    counts = collections.Counter()

    def emit(handle, row: dict) -> None:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()

    raw_handle = raw_path.open("a", encoding="utf-8", newline="\n")
    kept_handle = kept_path.open("a", encoding="utf-8", newline="\n")
    reject_handle = reject_path.open("a", encoding="utf-8", newline="\n")
    try:
        for position, target in enumerate(targets, start=1):
            task_id = int(target["task_id"])
            if task_id in done_ids:
                continue
            gold = str(target["asin"])
            executed: list[dict] = []
            terminal: dict = {}
            instruction = ""
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

            def stopped(result: dict, phase: str) -> str | None:
                if not result.get("done"):
                    return None
                reason = result.get("termination_reason") or (
                    (result.get("reward_detail") or {}).get("termination_reason")
                )
                return f"terminated_{phase}:{reason}"

            try:
                with ShopAgentEnv(base_url=base_url) as env:
                    env.include_trace_target = True
                    initial = env.reset(task_id)
                    instruction = str(initial.get("instruction") or "")
                    goal_options = [str(v) for v in initial.get("goal_options") or []]

                    # 1) 差查询：必须真的返回一屏结果，且 gold 不在这一页
                    result = run(env, "search_products", {"query": target["bad_query"]})
                    observation = executed[-1]["observation"]
                    failure = stopped(result, "bad_search")
                    if failure is None and page_contains(observation, gold):
                        failure = "gold_on_first_page_with_bad_query"
                    if failure is None and not page_asins(observation):
                        failure = "bad_query_returned_nothing"
                    # 2) 回搜索页再换词
                    if failure is None:
                        if "back to search" not in {
                            button.casefold() for button in clickable_buttons(observation)
                        }:
                            failure = "no_back_to_search_button"
                        else:
                            result = run(env, "back_to_search", {})
                            failure = stopped(result, "back")
                    # 3) 好查询：gold 必须出现在第 1 页
                    if failure is None:
                        result = run(env, "search_products", {"query": target["good_query"]})
                        observation = executed[-1]["observation"]
                        failure = stopped(result, "good_search")
                        if failure is None and not page_contains(observation, gold):
                            failure = "gold_not_on_first_page_with_good_query"
                    # 4) 打开、选规格、下单
                    if failure is None:
                        result = run(env, "open_product", {"asin": gold})
                        observation = executed[-1]["observation"]
                        failure = stopped(result, "open")
                        options = available_options(observation) if failure is None else {}
                        if failure is None and not options:
                            failure = "detail_page_without_options"
                        for required in goal_options:
                            if failure is not None:
                                break
                            actual = match_option(
                                required, options, clickable=set(clickable_buttons(observation))
                            )
                            if actual is None:
                                failure = f"required_option_missing:{required}"
                                break
                            result = run(env, "select_option", {"value": actual})
                            failure = stopped(result, "select")
                    if failure is None:
                        terminal = run(env, "buy_now", {})
                        detail = terminal.get("reward_detail") or {}
                        if detail.get("reward_type") != "gold_purchase":
                            failure = f"not_gold:{detail.get('reward_type')}"
            except Exception as exc:  # noqa: BLE001 - 单题异常不该带崩整批
                failure = f"exception:{type(exc).__name__}:{exc}"

            if failure is not None:
                code = failure.split(":")[0]
                rejections.append(
                    {"task_id": task_id, "code": code, "reason": failure, "steps": len(executed)}
                )
                counts[code] += 1
                emit(reject_handle, rejections[-1])
                print(f"  [skip {position}/{len(targets)}] task {task_id} {failure}")
                continue

            trajectory = build_trajectory(
                trajectory_id=f"m10requery-{task_id}-0",
                task_id=task_id,
                instruction=instruction,
                system_prompt=SYSTEM_PROMPT,
                executed=executed,
                terminal=terminal,
            )
            emit(raw_handle, trajectory)
            ok, reasons = acceptance_reasons(trajectory)
            if not ok:
                code = "acceptance"
                rejections.append(
                    {
                        "task_id": task_id,
                        "code": code,
                        "reason": "acceptance:" + ",".join(reasons),
                        "steps": len(executed),
                    }
                )
                counts[code] += 1
                emit(reject_handle, rejections[-1])
                print(f"  [skip {position}/{len(targets)}] task {task_id} acceptance:{reasons}")
                continue
            emit(kept_handle, build_sft_row(trajectory))
            print(
                f"  [ok {position}/{len(targets)}] task {task_id} "
                f"差查询 {target['bad_rank']} 名 → 好查询 {target['good_rank']} 名，"
                f"{len(executed)} 步"
            )
    finally:
        raw_handle.close()
        kept_handle.close()
        reject_handle.close()

    summary = {
        "targets": len(targets),
        "skipped_already_done": len(done_ids),
        "raw": count_lines(raw_path),
        "kept": count_lines(kept_path),
        "rejected": len(rejections),
        "reject_reasons": dict(counts.most_common()),
        "raw_path": str(raw_path),
        "kept_path": str(kept_path),
        "rejections_path": str(reject_path),
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT.parent / "m10-requery")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--limit", type=int, default=0, help="plan: 最多选多少题；generate: 最多跑多少题")
    parser.add_argument("--bad-min-rank", type=int, default=BAD_MIN_RANK)
    parser.add_argument("--good-max-rank", type=int, default=GOOD_MAX_RANK)
    parser.add_argument(
        "--resume", action="store_true", help="generate: 跳过输出目录里已完成的 task_id"
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="plan: 排除这些 jsonl 里已选的 task_id（可重复）",
    )
    args = parser.parse_args(argv)

    if args.mode == "plan":
        if args.good_max_rank >= args.bad_min_rank:
            raise SystemExit("--good-max-rank 必须小于 --bad-min-rank，否则两种查询分不开")
        exclude: set[int] = set()
        for path in args.exclude:
            exclude |= {int(row["task_id"]) for row in read_jsonl(path)}
        if exclude:
            print(f"排除 {len(exclude)} 道已被其他批次选走的题")
        chosen = plan_targets(
            limit=args.limit,
            index=args.index,
            bad_min_rank=args.bad_min_rank,
            good_max_rank=args.good_max_rank,
            exclude_ids=exclude,
        )
        write_jsonl(args.targets, chosen)
        bad = collections.Counter(
            "无名次" if row["bad_rank"] is None else f"{row['bad_rank'] // 40 * 40 + 1}+"
            for row in chosen
        )
        print(f"写入 {args.targets}：{len(chosen)} 题")
        print(f"差查询名次分布：{dict(sorted(bad.items()))}")
        return 0

    if not args.targets.is_file():
        raise SystemExit(f"没有 {args.targets}；先跑 --mode plan")
    targets = read_jsonl(args.targets)
    if args.limit:
        targets = targets[: args.limit]
    summary = generate(
        targets,
        base_url=args.base_url,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
