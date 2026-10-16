#!/usr/bin/env python3
"""离线行为多样性探针：在**不训练**的前提下测量 GRPO 组内信号是否充足。

对应 plans/task.md 的 M6-P0 诊断。回答四个问题：
  1. 同题 n 条 rollout 的终局奖励是否真有差异（零方差组 = 零梯度，会被动态采样丢弃）
  2. 组内"行为"是否多样（打开的商品、搜索词、工具序列、最终购买）
  3. n 从 4 提到 8 能救回多少组（同一批轨迹前 n 条配对比较）
  4. 实测每轨迹耗时，用于校准训练预算

M6-R 追加能力（2026-10-16）：`--filter-training-pool` 在**全池**上按每题的组内 reward 方差
打标签，产出 `difficulty_labels.jsonl` / `filtered_tasks.jsonl` / 可选过滤后 parquet。
依据：M6 第一轮零结果的根因是训练池里约 91% 的组终局奖励完全相同（零梯度），
详见 `plans/m6-grpo-runbook.md` §−1 与 §8。

关键设计：走与 GRPO 训练**完全一致**的 rollout 路径
  - system prompt : `rollout.SYSTEM_PROMPT`（与 `data/grpo/train.parquet` 的 system 消息逐字相同）
  - tool schema   : `--tools-config`（默认 `configs/tools.json`，即训练用 schema；
                    评测默认的 `SHOP_TOOL_SCHEMAS` 有 11/13 处描述不同，必须覆盖）
  - 环境与上限   : ShopSimulator v2.1，`--max-steps 35`，与 `configs/agent_loop.yaml` 对齐
  - 采样参数     : `--temperature` / `--top-p` 与 `configs/grpo.yaml` 对齐

只做推理，不加载 torch / veRL，**单卡即可，不需要双卡**。

用法
----
    # 1) 冒烟：先确认服务通（2 题 × 2 次；注意要显式给 --compare-n）
    python scripts/probe_group_diversity.py --make-task-list 2 --out-dir outputs/probe/smoke
    python scripts/probe_group_diversity.py \\
        --tasks outputs/probe/smoke/tasks.jsonl --out-dir outputs/probe/smoke \\
        --attempts-per-task 2 --compare-n 2 --temperature 0.7 --workers 2

    # 2) 从 GRPO 训练池抽题（固定 seed，可复现）
    python scripts/probe_group_diversity.py --make-task-list 50 \\
        --out-dir outputs/probe/t07

    # 3) 跑探针（可断点续跑：已完成的 (task_id, attempt) 会跳过）
    python scripts/probe_group_diversity.py \\
        --tasks outputs/probe/t07/tasks.jsonl --out-dir outputs/probe/t07 \\
        --attempts-per-task 8 --temperature 0.7 --top-p 0.9 --workers 8

    # 4) 只重算报告（不跑 rollout）
    python scripts/probe_group_diversity.py --report-only --out-dir outputs/probe/t07

    # 5) 换温度再跑一次，用同一个 --tasks 清单
    python scripts/probe_group_diversity.py \\
        --tasks outputs/probe/t07/tasks.jsonl --out-dir outputs/probe/t10 \\
        --attempts-per-task 8 --temperature 1.0 --top-p 0.9 --workers 8

    # 6) M6-R 重训练前的**全池**探针 + 难度过滤（见 plans/m6-grpo-runbook.md §8）
    #    训练用 n=4 ⇒ 每题采 4 条，"4 条里 reward 有无差异"就是"该组有无梯度"的定义。
    python scripts/probe_group_diversity.py --make-task-list 906 \\
        --out-dir outputs/probe/m6r-full
    python scripts/probe_group_diversity.py \\
        --tasks outputs/probe/m6r-full/tasks.jsonl \\
        --out-dir outputs/probe/m6r-full \\
        --attempts-per-task 4 --compare-n 4 \\
        --temperature 0.7 --top-p 0.9 --workers 24 \\
        --filter-training-pool --filter-n 4 \\
        --parquet-out outputs/probe/m6r-full/train_filtered.parquet

    # 此命令只生成探针结果与 parquet 副本；绝不改写 data/grpo/train.parquet。
    # 断点后重跑同一命令会跳过已经完成的 task_id × attempt_index。
    # 默认不加 --filter-band：核心条件 reward_varying 已等价于"该组有非零 advantage"，
    # 加带会误杀"4 条都没买成但 shaped 分有差异"的组（它们同样有梯度）。
    # 想换过滤口径时，直接读 difficulty_labels.jsonl 离线重算，不必重跑探针。

失败处理
--------
`collect_for_task` 会把异常吞进轨迹（`status="error"`），因此本脚本显式识别：

  - 服务不可达 → **预检阶段直接退出**，不会静默产出全 error 轨迹
  - 单条失败 → 记入 `<out-dir>/failures.jsonl`，**不写入** `trajectories.jsonl`，
    因此修好服务后重跑会自动重试这些 (task_id, attempt)
  - 基础设施故障（URLError / ShopHttpError / 环境资源耗尽）→ 立即停止
  - 连续 5 条失败 → 立即停止
  - 失败率 > `--max-failure-rate`（默认 0.2）→ 拒绝出报告并非零退出

**判断"跑得对不对"的最快方法**：看打印的「成功轨迹平均 X s/条」。
真实轨迹约 12–18 s/条；若只有 0.1 s/条，说明服务没起、每条都瞬间失败。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.rollout import (  # noqa: E402
    OpenAIChatClient,
    _is_infrastructure_failure,
    collect_for_task,
    completed_task_attempts,
)
from shopping_grpo.training.grpo.group_diversity import (  # noqa: E402
    DEFAULT_TOLERANCE,
    compare_n,
    group_trajectories,
    summarize_group,
)

DEFAULT_TOOLS_CONFIG = ROOT / "configs/tools.json"
DEFAULT_TRAIN_POOL = ROOT / "data/grpo/train.jsonl"
DEFAULT_LONGTAIL_PROBE = ROOT / "audit/longtail_probe.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", type=Path, help="任务清单 JSONL（每行 {\"task_id\": N}）")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--make-task-list",
        type=int,
        metavar="N",
        help="从 GRPO 训练池随机抽 N 题写入 <out-dir>/tasks.jsonl 后退出",
    )
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-only", action="store_true", help="只用已有 trajectories.jsonl 重算报告"
    )

    parser.add_argument("--attempts-per-task", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--context-compaction", action="store_true")
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)

    parser.add_argument("--model", default="shopping-agent")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--tools-config", type=Path, default=DEFAULT_TOOLS_CONFIG)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--compare-n",
        default=None,
        help="模拟比较的 n 值，逗号分隔；默认按 --attempts-per-task 自动推导（如 8 → 4,8）",
    )
    parser.add_argument(
        "--filter-training-pool",
        action="store_true",
        help=(
            "基于每题实际组内 reward 方差输出 difficulty_labels.jsonl 和 "
            "filtered_tasks.jsonl；不修改源数据"
        ),
    )
    parser.add_argument(
        "--filter-n",
        type=int,
        default=None,
        help="按每题前 N 个 attempt 评估训练信号；默认取 --compare-n 最大值",
    )
    parser.add_argument(
        "--filter-band",
        default=None,
        metavar="LO,HI",
        help=(
            "可选的购买成功率附加限制，范围 [0,1]，如 0.125,0.875。"
            "核心保留条件始终是有效且 reward_varying"
        ),
    )
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=ROOT / "data/grpo/train.parquet",
        help="过滤取行的源 parquet（task_id 从 extra_info 列读取）",
    )
    parser.add_argument(
        "--parquet-out",
        type=Path,
        default=None,
        help="写出过滤后的 parquet 副本；绝不覆盖源 parquet（需要 pyarrow）",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.2,
        help="轨迹失败率超过该值即拒绝出报告并非零退出（默认 0.2）",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过环境与模型服务的连通性预检",
    )
    return parser.parse_args()


def resolve_compare_n(args: argparse.Namespace) -> list[int]:
    if args.compare_n:
        values = [int(value) for value in str(args.compare_n).split(",") if value.strip()]
        if not values:
            raise SystemExit("--compare-n 解析为空")
        return values
    attempts = int(args.attempts_per_task)
    candidates = [value for value in (4, 8) if value <= attempts]
    return candidates or [attempts]


def preflight(args: argparse.Namespace) -> None:
    """连通性预检：服务没起时立刻给出可读的错误，而不是静默产出全 error 轨迹。"""
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen

    checks = [
        ("环境服务", args.base_url),
        ("模型服务", args.llm_base_url.rstrip("/") + "/models"),
    ]
    failures = []
    for name, url in checks:
        try:
            with urlopen(url, timeout=10) as response:
                status = getattr(response, "status", 200)
            if status >= 500:
                failures.append(f"  {name} 返回 HTTP {status}（上游不可用）：{url}")
            else:
                print(f"  预检 {name}: {url} -> HTTP {status}")
        except HTTPError as exc:
            # 4xx 说明服务本身活着（只是该路径没有）；5xx 说明上游没起
            if exc.code >= 500:
                failures.append(f"  {name} 返回 HTTP {exc.code}（上游不可用）：{url}")
            else:
                print(f"  预检 {name}: {url} -> HTTP {exc.code}（服务在线）")
        except (URLError, OSError) as exc:
            failures.append(f"  {name} 不可达：{url} ({exc})")
    if failures:
        raise SystemExit(
            "预检失败，请先启动服务：\n"
            + "\n".join(failures)
            + "\n\n  环境： bash scripts/start_environment.sh\n"
            "  模型： bash scripts/serve_model.sh <merged 模型目录>"
        )


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_filter_band(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise SystemExit("--filter-band 格式应为 LO,HI，例如 0.125,0.875")
    try:
        low, high = map(float, parts)
    except ValueError as exc:
        raise SystemExit("--filter-band 必须是两个数字") from exc
    if not (0.0 <= low <= high <= 1.0):
        raise SystemExit("--filter-band 必须满足 0 ≤ LO ≤ HI ≤ 1")
    return low, high


def build_difficulty_labels(
    grouped: dict[int, list[dict]],
    task_ids: list[int],
    *,
    filter_n: int,
    tolerance: float,
    pass_rate_band: tuple[float, float] | None = None,
) -> list[dict]:
    """用 GRPO 的实际组内 reward 方差给全池任务打标签。"""
    if filter_n < 2:
        raise ValueError("filter_n 必须至少为 2")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("训练池 task_id 有重复")

    labels = []
    for task_id in task_ids:
        items = list(grouped.get(int(task_id), []))
        items.sort(key=lambda item: int(item.get("attempt_index", 0)))
        if len(items) < filter_n:
            labels.append(
                {
                    "task_id": int(task_id),
                    "attempts_observed": len(items),
                    "filter_n": filter_n,
                    "eligible": False,
                    "filter_reason": "insufficient_attempts",
                }
            )
            continue
        selected = items[:filter_n]
        attempt_indices = [
            int(item.get("attempt_index", i)) for i, item in enumerate(selected)
        ]
        if len(set(attempt_indices)) != len(attempt_indices):
            raise ValueError(f"task_id={task_id} 的 attempt_index 重复")

        stats = summarize_group(selected, tolerance=tolerance)
        successes = sum(bool(value) for value in stats["purchase_success"])
        pass_rate = successes / filter_n
        inside_band = (
            pass_rate_band is None
            or pass_rate_band[0] <= pass_rate <= pass_rate_band[1]
        )
        eligible = bool(stats["kept"]) and inside_band
        reason = stats["drop_reason"]
        if stats["kept"] and not inside_band:
            reason = "outside_pass_rate_band"
        labels.append(
            {
                "task_id": int(task_id),
                "attempts_observed": len(items),
                "filter_n": filter_n,
                "pass_count": successes,
                "pass_rate": pass_rate,
                "reward_varying": bool(stats["reward_varying"]),
                "utility_min": float(stats["utility_min"]),
                "utility_max": float(stats["utility_max"]),
                "utility_std": float(stats["utility_std"]),
                "sampling_invalid_reasons": list(stats["sampling_invalid_reasons"]),
                "eligible": eligible,
                "filter_reason": None if eligible else reason,
            }
        )
    return labels


def write_filter_outputs(
    out_dir: Path,
    labels: list[dict],
    *,
    train_parquet: Path,
    parquet_out: Path | None = None,
) -> list[int]:
    """写出任务清单/标签与可选 parquet 子集；源训练池保持只读。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / "difficulty_labels.jsonl"
    task_path = out_dir / "filtered_tasks.jsonl"
    selected_ids = [int(row["task_id"]) for row in labels if row.get("eligible")]
    if not selected_ids:
        raise SystemExit("难度过滤没有保留任何任务；拒绝写出空训练集")

    with label_path.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with task_path.open("w", encoding="utf-8") as handle:
        for task_id in selected_ids:
            handle.write(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n")

    if parquet_out is not None:
        source = Path(train_parquet).expanduser().resolve()
        target = Path(parquet_out).expanduser().resolve()
        if source == target:
            raise SystemExit("parquet-out 不能与 train-parquet 相同；拒绝覆盖源训练集")
        if not source.is_file():
            raise SystemExit(f"源 parquet 不存在：{source}")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("写 parquet 需要 pyarrow；请使用仓库 setup 环境安装") from exc

        table = pq.read_table(source)
        if "extra_info" not in table.column_names:
            raise SystemExit("源 parquet 缺少 extra_info 列，无法映射 task_id")
        extra_info = table.column("extra_info").to_pylist()
        source_ids = []
        for row_index, value in enumerate(extra_info):
            if not isinstance(value, dict) or "task_id" not in value:
                raise SystemExit(f"源 parquet 第 {row_index} 行 extra_info.task_id 缺失")
            source_ids.append(int(value["task_id"]))
        unknown = sorted(set(selected_ids) - set(source_ids))
        if unknown:
            raise SystemExit(f"筛选出的 task_id 不在源 parquet 中：{unknown[:20]}")
        selected = set(selected_ids)
        mask = pa.array([task_id in selected for task_id in source_ids])
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            raise SystemExit("过滤后的 parquet 为空；拒绝输出")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(filtered, target)
        print(f"过滤 parquet：{table.num_rows} → {filtered.num_rows} 行：{target}")

    print(
        f"难度标签：保留 {len(selected_ids)}/{len(labels)} 题；"
        f"任务清单：{task_path}；标签：{label_path}"
    )
    return selected_ids


def training_tool_schemas(path: Path) -> list[dict]:

    """取出与 GRPO 训练完全一致的 tool schema。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw["tools"] if isinstance(raw, dict) else raw
    schemas = []
    for entry in entries:
        schema = entry.get("tool_schema", entry)
        schemas.append(schema)
    names = [item["function"]["name"] for item in schemas]
    if len(set(names)) != len(names):
        raise SystemExit(f"工具名重复：{names}")
    return schemas


def make_task_list(args: argparse.Namespace) -> None:
    pool = [int(row["task_id"]) for row in read_jsonl(args.train_pool)]
    if not pool:
        raise SystemExit(f"训练池为空：{args.train_pool}")
    count = min(int(args.make_task_list), len(pool))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(pool, count))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "tasks.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for task_id in chosen:
            handle.write(json.dumps({"task_id": task_id}) + "\n")
    print(f"已写入 {count} 题 -> {target}")

    # 顺带打印这批题在检索难度上的分布，便于判断样本是否有代表性
    if DEFAULT_LONGTAIL_PROBE.exists():
        probe = json.loads(DEFAULT_LONGTAIL_PROBE.read_text(encoding="utf-8"))
        ranks = {
            int(row["task_id"]): row.get("gold_rank_Q4")
            for row in probe.get("rows", [])
            if row.get("dataset") == "grpo_train"
        }
        buckets: dict[str, int] = {}
        for task_id in chosen:
            rank = ranks.get(task_id)
            key = (
                "未命中"
                if not rank
                else "rank1"
                if rank == 1
                else "rank2-3"
                if rank <= 3
                else "rank4-10"
                if rank <= 10
                else ">10"
            )
            buckets[key] = buckets.get(key, 0) + 1
        print(f"  样本的 gold_rank_Q4 分布：{dict(sorted(buckets.items()))}")


def build_client(args: argparse.Namespace) -> OpenAIChatClient:
    return OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )


def run_rollouts(args: argparse.Namespace, tasks: list[dict]) -> None:
    output_path = Path(args.out_dir) / "trajectories.jsonl"
    schemas = training_tool_schemas(args.tools_config)
    done = completed_task_attempts(output_path)
    pending_pairs = [
        (task, attempt_index)
        for task in tasks
        for attempt_index in range(int(args.attempts_per_task))
        if (int(task["task_id"]), attempt_index) not in done
    ]
    total = len(pending_pairs)
    if not total:
        print("没有待跑任务（已全部完成）")
        return
    print(
        f"待跑 {total} 条轨迹：{len(tasks)} 题 × {args.attempts_per_task} 次，"
        f"并发 {args.workers}，temperature={args.temperature}，top_p={args.top_p}"
    )

    started = time.perf_counter()
    finished = 0
    durations: list[float] = []
    failures: list[dict] = []
    consecutive_failures = 0
    failure_path = Path(args.out_dir) / "failures.jsonl"
    queue = iter(pending_pairs)
    max_consecutive = 5

    def worker(pair):
        task, attempt_index = pair
        begin = time.perf_counter()
        trajectory = collect_for_task(
            task,
            client=build_client(args),
            base_url=args.base_url,
            max_steps=args.max_steps,
            tools=schemas,
            attempt_index=attempt_index,
        )
        return trajectory, time.perf_counter() - begin

    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {}

        def submit_available() -> None:
            while len(futures) < int(args.workers):
                try:
                    pair = next(queue)
                except StopIteration:
                    return
                futures[executor.submit(worker, pair)] = pair

        submit_available()
        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                pair = futures.pop(future)
                task, attempt_index = pair
                try:
                    trajectory, elapsed = future.result()
                except Exception as exc:  # 连 collect_for_task 都抛了：直接停
                    print(
                        f"  基础设施故障，已停止：task_id={task['task_id']} "
                        f"attempt={attempt_index} {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    raise SystemExit(2) from exc

                # collect_for_task 会把异常吞进轨迹里，必须显式识别
                error = trajectory.get("error") or {}
                release_error = trajectory.get("release_error") or {}
                if error or release_error or trajectory.get("status") == "error":
                    detail = error or release_error
                    record = {
                        "task_id": int(task["task_id"]),
                        "attempt_index": int(attempt_index),
                        "status": trajectory.get("status"),
                        "error_type": detail.get("type"),
                        "error_message": detail.get("message"),
                        "steps_completed": len(trajectory.get("steps") or []),
                    }
                    failures.append(record)
                    append_jsonl(failure_path, [record])
                    consecutive_failures += 1
                    if len(failures) <= 3:
                        print(
                            f"  [失败] task_id={record['task_id']} "
                            f"attempt={record['attempt_index']} "
                            f"{record['error_type']}: {record['error_message']}",
                            file=sys.stderr,
                        )
                    if _is_infrastructure_failure(trajectory):
                        raise SystemExit(
                            "环境或模型服务不可用，已停止。失败明细见 "
                            f"{failure_path}\n"
                            f"  最后一条：{record['error_type']}: {record['error_message']}"
                        )
                    if consecutive_failures >= max_consecutive:
                        raise SystemExit(
                            f"连续 {max_consecutive} 条轨迹失败，已停止（疑似服务不可用）。"
                            f"失败明细见 {failure_path}\n"
                            f"  最后一条：{record['error_type']}: {record['error_message']}"
                        )
                else:
                    consecutive_failures = 0
                    append_jsonl(output_path, [trajectory])
                    durations.append(elapsed)

                finished += 1
                if finished % 10 == 0 or finished == total:
                    rate = (time.perf_counter() - started) / finished
                    print(
                        f"  {finished}/{total}  平均 {rate:.1f} s/条  "
                        f"预计剩余 {rate * (total - finished) / 60:.0f} 分钟"
                    )
            submit_available()

    wall = time.perf_counter() - started
    ok = finished - len(failures)
    if durations:
        print(
            f"完成 {finished} 条：成功 {ok}、失败 {len(failures)}，"
            f"总耗时 {wall / 60:.1f} 分钟，成功轨迹平均 "
            f"{statistics.mean(durations):.1f} s/条（中位 {statistics.median(durations):.1f}）"
        )
    else:
        print(f"完成 {finished} 条：全部失败（{len(failures)}）", file=sys.stderr)

    failure_rate = len(failures) / finished if finished else 1.0
    if failure_rate > float(args.max_failure_rate):
        raise SystemExit(
            f"轨迹失败率 {failure_rate:.0%} 超过阈值 {float(args.max_failure_rate):.0%}，"
            f"拒绝生成报告。失败明细见 {failure_path}\n"
            "  常见原因：环境或模型服务未启动；请先跑 bash scripts/start_environment.sh "
            "与 bash scripts/serve_model.sh <merged 模型目录>"
        )


def build_report(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    trajectories = read_jsonl(out_dir / "trajectories.jsonl")
    if not trajectories:
        raise SystemExit(f"没有轨迹可分析：{out_dir / 'trajectories.jsonl'}")

    failures = read_jsonl(out_dir / "failures.jsonl")
    grouped = group_trajectories(trajectories)
    # ThreadPoolExecutor 写回顺序按完成时间；统一按 attempt_index 排序，
    # 保证 n=4 与 n=8 的比较总是前 4 条 vs 前 8 条。
    grouped = {
        task_id: sorted(items, key=lambda item: int(item.get("attempt_index", 0)))
        for task_id, items in grouped.items()
    }
    n_values = resolve_compare_n(args)
    full_n = max(n_values)
    usable = {key: value for key, value in grouped.items() if len(value) >= full_n}

    filter_summary = None
    filter_requested = (
        args.filter_training_pool
        or args.filter_band is not None
        or args.parquet_out is not None
    )
    if filter_requested:
        task_source = args.tasks if args.tasks is not None else args.train_pool
        task_rows = read_jsonl(task_source)
        if not task_rows:
            raise SystemExit(f"难度过滤任务清单为空：{task_source}")
        task_ids = [int(row["task_id"]) for row in task_rows]
        filter_n = args.filter_n or full_n
        if filter_n < 2 or filter_n > int(args.attempts_per_task):
            raise SystemExit(
                f"--filter-n 必须在 2 到 --attempts-per-task 之间，当前 {filter_n}"
            )
        pass_rate_band = parse_filter_band(args.filter_band)
        labels = build_difficulty_labels(
            grouped,
            task_ids,
            filter_n=filter_n,
            tolerance=args.tolerance,
            pass_rate_band=pass_rate_band,
        )
        incomplete = [
            row["task_id"] for row in labels
            if row.get("filter_reason") == "insufficient_attempts"
        ]
        if incomplete:
            raise SystemExit(
                f"全池探针未完整：{len(incomplete)}/{len(labels)} 题未达到 n={filter_n}；"
                f"例：{incomplete[:20]}。修复失败轨迹后续跑，不生成过滤训练集。"
            )
        selected_ids = write_filter_outputs(
            out_dir,
            labels,
            train_parquet=args.train_parquet,
            parquet_out=args.parquet_out,
        )
        filter_summary = {
            "filter_n": filter_n,
            "pass_rate_band": pass_rate_band,
            "eligible_tasks": len(selected_ids),
            "total_tasks": len(labels),
            "excluded_tasks": len(labels) - len(selected_ids),
            "parquet_out": str(args.parquet_out) if args.parquet_out else None,
        }

    if not usable:
        counts = sorted({len(value) for value in grouped.values()})
        hint = (
            f"  · 若只是冒烟测试，显式加 --compare-n {max(counts)}（当前 --attempts-per-task "
            f"{args.attempts_per_task}）\n"
            if max(counts) < full_n
            else ""
        )
        raise SystemExit(
            f"没有一道题凑满 {full_n} 条轨迹（实际每题 {counts} 条）。\n"
            + hint
            + f"  · 正式探针请用 --attempts-per-task {full_n}"
        )

    group_rows = [
        summarize_group(items, tolerance=args.tolerance, n=full_n)
        for items in usable.values()
    ]
    # 每次重算覆盖，避免 --report-only 重复运行时累积重复行
    with (out_dir / "groups.jsonl").open("w", encoding="utf-8") as handle:
        for row in group_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "probe": {
            "trajectories": len(trajectories),
            "tasks_usable": len(usable),
            "tasks_total": len(grouped),
            "attempts_per_task": int(args.attempts_per_task),
            "attempts_observed": Counter(
                len(value) for value in grouped.values()
            ).most_common(1)[0][0],
            "failed_trajectories": len(failures),
            "failure_error_types": dict(
                sorted(
                    Counter(
                        str(row.get("error_type")) for row in failures
                    ).items()
                )
            ),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_steps": args.max_steps,
            "tools_config": str(args.tools_config),
            "base_url": args.base_url,
            "llm_base_url": args.llm_base_url,
            "tolerance": args.tolerance,
        },
        "n_comparison": compare_n(usable, n_values, tolerance=args.tolerance),
        "training_pool_filter": filter_summary,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    probe = report["probe"]
    lines = [
        "# 行为多样性探针报告",
        "",
        f"- 题目数：{probe['tasks_usable']} 组（共 {probe['trajectories']} 条轨迹，每题实测 {probe['attempts_observed']} 次）",
        f"- 采样：temperature={probe['temperature']}，top_p={probe['top_p']}，max_steps={probe['max_steps']}",
        f"- 工具 schema：`{probe['tools_config']}`",
        f"- 失败轨迹：{probe['failed_trajectories']} 条 {json.dumps(probe['failure_error_types'], ensure_ascii=False)}",
        "",
        "## 1. 组内信号（决定 GRPO 有没有梯度）",
        "",
        "| n | 组数 | 零方差组占比 | 无效组占比 | **有效组占比** | 全成功组占比 | 全失败组占比 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in report["n_comparison"].items():
        signal = entry["signal"]
        lines.append(
            f"| {key} | {entry['num_groups_usable']} | "
            f"{signal['constant_reward_group_rate'] * 100:.1f}% | "
            f"{signal['sampling_invalid_group_rate'] * 100:.1f}% | "
            f"**{signal['kept_group_rate'] * 100:.1f}%** | "
            f"{signal['all_purchase_success_group_rate'] * 100:.1f}% | "
            f"{signal['no_purchase_success_group_rate'] * 100:.1f}% |"
        )
    lines += [
        "",
        "> 零方差组与无效组都会被 `dynamic_sampling` 直接丢弃、重新采题；"
        "有效组占比越低，达到同样有效批量所需的实际 rollout 越多。",
        "",
        "## 2. 组内行为多样性",
        "",
        "| n | 平均不同「打开商品集合」数 | ≥2 种的组占比 | 平均不同「购买商品」数 | ≥2 种的组占比 | 平均不同「搜索词集合」数 | 平均不同「工具序列」数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in report["n_comparison"].items():
        diversity = entry["diversity"]
        lines.append(
            f"| {key} | {diversity['mean_distinct_open_sets']:.2f} | "
            f"{diversity['rate_groups_with_ge2_open_sets'] * 100:.1f}% | "
            f"{diversity['mean_distinct_purchases']:.2f} | "
            f"{diversity['rate_groups_with_ge2_purchases'] * 100:.1f}% | "
            f"{diversity['mean_distinct_query_sets']:.2f} | "
            f"{diversity['mean_distinct_tool_sequences']:.2f} |"
        )
    lines += ["", "## 3. 行为与稳定性", ""]
    for key, entry in report["n_comparison"].items():
        behaviour = entry["behaviour"]
        reward = entry["reward"]
        lines += [
            f"**{key}**",
            "",
            f"- 平均工具步数：{behaviour['mean_steps_per_trajectory']:.2f}",
            f"- guard 阻断 / 组：{behaviour['guard_blocked_per_group']:.2f}",
            f"- 工具调用截断 / 组：{behaviour['truncations_per_group']:.2f}",
            f"- 上下文压缩 / 组：{behaviour['compactions_per_group']:.2f}",
            f"- 平均终局奖励：{reward['mean_terminal_utility']:.4f}",
            f"- 终局类型计数：{json.dumps(reward['reward_type_counts'], ensure_ascii=False)}",
            "",
        ]
    filter_summary = report.get("training_pool_filter")
    if filter_summary:
        lines += [
            "## 4. M6 重训练任务过滤",
            "",
            f"- 探针组大小 n={filter_summary['filter_n']}；"
            f"保留 {filter_summary['eligible_tasks']}/{filter_summary['total_tasks']} 题，"
            f"排除 {filter_summary['excluded_tasks']} 题。",
            "- 核心规则：无 sampling-invalid 且组内终局 utility 方差大于 tolerance。",
        ]
        if filter_summary["pass_rate_band"] is not None:
            low, high = filter_summary["pass_rate_band"]
            lines.append(f"- 附加购买成功率范围：[{low:.3f}, {high:.3f}]。")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.make_task_list:
        make_task_list(args)
        return 0

    if not args.report_only:
        if not args.tasks:
            raise SystemExit("需要 --tasks（或先用 --make-task-list 生成）")
        tasks = read_jsonl(args.tasks)
        if not tasks:
            raise SystemExit(f"任务清单为空：{args.tasks}")
        if not args.skip_preflight:
            print("预检服务连通性…")
            preflight(args)
        run_rollouts(args, tasks)

    report = build_report(args)
    print()
    print(render_markdown(report))
    print(f"报告已写入 {Path(args.out_dir) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
