#!/usr/bin/env python3
"""M10-R 对比候选：合成「先开第 1 名发现不对，回列表换一个」的轨迹。

为什么补这个
------------
评测 B 桶 35 题是「看见了 gold，却开了别的并买下」。池子里
**91.1%（1,140/1,252）的轨迹只开过 1 个商品**，「开过 ≥2 个」只有 8.9%，
真正「回头再开前面那个」只有 2.88%。模型没有「第一个不对就换」的示范。

动作序列不是编的：真实轨迹里 `open_product → back_to_search` 出现 **95 次**，
只是稀少。本脚本把这个真实模式变密，并明确框成「第 1 名是错的」。

不教什么
--------
不教「回头买」（重新打开更早开过的商品）—— M9 已证明那条路教不动，
这里最后买的仍然是**刚打开的那个**。

合法性
------
与 `generate_m10_paging.py` 同一套：训练口径是 `tool-call-only`，助手自然语言
本来就被清空、不参与梯度，只有工具调用参与；观测是环境真实返回的，
动作是合法动作。**不改任何现有文件**。

两步走
------
1. ``--mode plan``（纯 CPU）：找 gold 排 2–20 名（在第 1 页但不在最上面）的题，
   记下 gold、它用的查询词、以及第 1 名的 ASIN（当诱饵）。
2. ``--mode generate``（需要环境服务）：执行
   ``search → open(诱饵) → back_to_search → open(gold) → select_option×m → buy_now``，
   校验两次都在第 1 页看见、终局必须 gold_purchase。

用法::

    .venv/Scripts/python.exe scripts/generate_m10_compare.py --mode plan
    .venv/Scripts/python.exe scripts/generate_m10_compare.py --mode generate --limit 3
    .venv/Scripts/python.exe scripts/generate_m10_compare.py --mode generate
    .venv/Scripts/python.exe scripts/generate_m10_compare.py --mode generate --resume
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
    available_options,
    best_candidate,
    build_trajectory,
    count_lines,
    match_option,
    page_contains,
    read_jsonl,
    tool_call,
    write_jsonl,
)

DEFAULT_TARGETS = ROOT / "data/collection/m10_compare_targets.jsonl"

#: gold 必须落在第 1 页（≤20 名）但不在最上面（≥2 名）——这样诱饵才存在。
MIN_GOLD_RANK = 2
MAX_GOLD_RANK = 20


def plan_targets(
    *,
    limit: int,
    index: Path,
    min_rank: int = MIN_GOLD_RANK,
    max_rank: int = MAX_GOLD_RANK,
    exclude_ids: set[int] | None = None,
) -> list[dict]:
    """离线选题：gold 在第 1 页但不在第 1 名，且能拿到一个诱饵 ASIN。

    ``exclude_ids`` 排除别的批次已选走的题，避免同一 task_id 进池两次。
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
        candidates = [
            (query, _rank_and_top(searcher, query, asin))
            for query in selection.build_queries(item, instruction)
        ]
        pairs = [(query, payload) for query, payload in candidates if payload is not None]
        if not pairs:
            histogram["miss"] += 1
            continue
        # 取 gold 名次最小的那个查询；同分取先出现的
        query, (rank, top_asin) = min(pairs, key=lambda pair: pair[1][0])
        if not (min_rank <= rank <= max_rank):
            histogram["out"] += 1
            continue
        if top_asin is None or top_asin == asin:
            histogram["no_decoy"] += 1
            continue
        chosen.append(
            {
                "task_id": int(task_id),
                "asin": asin,
                "decoy_asin": top_asin,
                "query": query,
                "rank": int(rank),
                "instruction": str(instruction.get("instruction") or ""),
            }
        )
        histogram["paired"] += 1
        if limit and len(chosen) >= limit:
            break
    print(f"扫描统计 {dict(histogram)}；选出 {len(chosen)} 条")
    return chosen


def _rank_and_top(searcher, query: str, asin: str) -> tuple[int, str | None] | None:
    """返回 (gold 名次, 第 1 名 ASIN)；gold 不在召回里返回 None。"""
    try:
        hits = searcher.search(query, k=150)
    except Exception:  # noqa: BLE001 - 空查询会抛，跳过
        return None
    top = str(hits[0].asin) if hits else None
    for hit in hits:
        if str(hit.asin) == asin:
            return int(hit.rank), top
    return None


def generate(
    targets: list[dict],
    *,
    base_url: str,
    out_dir: Path,
    resume: bool = False,
) -> dict:
    """连环境跑「先开诱饵、回列表、再开 gold」的脚本化轨迹。"""
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
            decoy = str(target["decoy_asin"])
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

                    result = run(env, "search_products", {"query": target["query"]})
                    observation = executed[-1]["observation"]
                    failure = stopped(result, "search")
                    if failure is None and not page_contains(observation, gold):
                        failure = "gold_not_on_first_page"
                    if failure is None and not page_contains(observation, decoy):
                        failure = "decoy_not_on_first_page"
                    # 先开第 1 名（错的）
                    if failure is None:
                        result = run(env, "open_product", {"asin": decoy})
                        observation = executed[-1]["observation"]
                        failure = stopped(result, "open_decoy")
                    # 回列表
                    if failure is None:
                        if "back to search" not in {
                            button.casefold() for button in clickable_buttons(observation)
                        }:
                            failure = "no_back_to_search_button"
                        else:
                            result = run(env, "back_to_search", {})
                            observation = executed[-1]["observation"]
                            failure = stopped(result, "back")
                    # 再开 gold
                    if failure is None and not page_contains(observation, gold):
                        failure = "gold_lost_after_back_to_search"
                    if failure is None:
                        result = run(env, "open_product", {"asin": gold})
                        observation = executed[-1]["observation"]
                        failure = stopped(result, "open_gold")
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
                trajectory_id=f"m10compare-{task_id}-0",
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
                f"诱饵(第1名) → gold(第{target['rank']}名)，{len(executed)} 步"
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT.parent / "m10-compare")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--limit", type=int, default=0, help="plan: 最多选多少题；generate: 最多跑多少题")
    parser.add_argument("--min-rank", type=int, default=MIN_GOLD_RANK)
    parser.add_argument("--max-rank", type=int, default=MAX_GOLD_RANK)
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
        if args.max_rank < args.min_rank:
            raise SystemExit("--max-rank 不能小于 --min-rank")
        exclude: set[int] = set()
        for path in args.exclude:
            exclude |= {int(row["task_id"]) for row in read_jsonl(path)}
        if exclude:
            print(f"排除 {len(exclude)} 道已被其他批次选走的题")
        chosen = plan_targets(
            limit=args.limit,
            index=args.index,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            exclude_ids=exclude,
        )
        write_jsonl(args.targets, chosen)
        ranks = collections.Counter(row["rank"] // 5 * 5 for row in chosen)
        print(f"写入 {args.targets}：{len(chosen)} 题")
        print(f"gold 名次分布（按 5 分段）：{dict(sorted(ranks.items()))}")
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
