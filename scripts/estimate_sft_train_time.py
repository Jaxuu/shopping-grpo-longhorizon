#!/usr/bin/env python3
"""用实测标定外推 SFT 课程各阶段的耗时，不靠感觉。

标定数据来自 M5 在**同类 48 GB 卡**上的实测（plans/m5-linux-sft-checklist.md §1.1、
plans/m5-sft-ab-runbook.md §3.6）：

    阶段      平均字符   s/step
    A          6,963      50.2
    B          7,793      56.2
    C          9,371      72.0

三点拟合出 s/step 与「平均字符数」的关系，再把目标数据的平均字符数代进去。

为什么「平均字符数」是合法的自变量：训练是 per_device_batch=1 × grad_accum=8，
一个 optimizer step 就是 8 条样本各做一次前向/反向，**不存在 batch 内 padding**，
所以 step 耗时 ∝ 8 条样本的长度之和 ∝ 平均长度。长序列的 attention 是 O(L²)，
因此另给一个幂律拟合（指数 >1）作为上界。

用法：
    python scripts/estimate_sft_train_time.py --data data/sft_m10r/capped/all.jsonl \\
        --labels data/sft_m10r/capped/difficulty_labels.jsonl \\
        --manifest data/sft_m10r/capped/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_sft_curriculum import (  # noqa: E402
    behavior_bucket,
    behavior_profile,
    _bucket,
)

# 实测点。前三行是 M5 在同类 48 GB 卡上的标定（plans/m5-linux-sft-checklist.md §1.1、
# plans/m5-sft-ab-runbook.md §3.6）；第四行是 M10-R stage A 在 2026-10-23 00:43 的实测
# （16 步 / 31:53 = 119.6 s/step，平均字符 18,726），用来把外推锚回真实速度——
# 只用 M5 三点时它把 A 段高估到 158 s/step，实际是 120。
CALIBRATION = [(6963, 50.2), (7793, 56.2), (9371, 72.0), (18726, 119.6)]
EFFECTIVE_BATCH = 8  # per_device_train_batch_size=1 × gradient_accumulation_steps=8


def chars(row: dict) -> int:
    """一条轨迹的字符数：所有 message 正文 + 工具调用参数。"""
    total = 0
    for message in row.get("messages") or []:
        total += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            total += len(str((call.get("function") or {}).get("arguments") or ""))
    return total


def approx_tokens(row: dict) -> float:
    """粗估 token 数：ASCII 约 4 字符/token，非 ASCII 约 1 字符/token。

    只用来判断「会不会被 max_length 丢掉」，不用于计时。
    """
    total = 0.0
    for message in row.get("messages") or []:
        text = str(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            text += str((call.get("function") or {}).get("arguments") or "")
        wide = sum(1 for ch in text if ord(ch) > 127)
        total += wide + (len(text) - wide) / 4.0
    return total


def fit_linear(points):
    """最小二乘：s/step = a + b × chars。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def fit_power(points):
    """幂律：s/step = exp(a) × chars^b（b>1 反映 attention 的 O(L²)）。"""
    pts = [(math.log(x), math.log(y)) for x, y in points]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def predict(avg_chars: float, linear, power) -> dict:
    a_lin, b_lin = linear
    a_pow, b_pow = power
    lin = a_lin + b_lin * avg_chars
    # 过原点按比例（最保守的下界：假设没有固定开销）
    prop = avg_chars * (CALIBRATION[0][1] / CALIBRATION[0][0])
    pow_ = math.exp(a_pow + b_pow * math.log(avg_chars))
    return {"linear": lin, "power": pow_, "proportional": prop}


def bucket_stats(rows, labels_by_id, mode, max_length):
    groups = defaultdict(list)
    for row in rows:
        if mode == "difficulty":
            label = labels_by_id.get(int(row["task_id"]))
            if label is None:
                continue
            groups[_bucket(label)].append(row)
        else:
            groups[behavior_bucket(behavior_profile(row))].append(row)
    stats = {}
    for name, items in groups.items():
        values = sorted(chars(row) for row in items)
        tokens = sorted(approx_tokens(row) for row in items)
        stats[name] = {
            "rows": len(items),
            "mean_chars": statistics.mean(values),
            "p50": values[len(values) // 2],
            "p90": values[int(len(values) * 0.9)],
            "max": values[-1],
            "mean_tokens": statistics.mean(tokens),
            "max_tokens": tokens[-1],
            "over_max_length": sum(1 for value in tokens if value > max_length),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--manifest", type=Path, help="课程清单；给了就按阶段算")
    parser.add_argument("--bucket-mode", choices=("behavior", "difficulty"), default="behavior")
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument(
        "--stage-counts",
        help="没有真实池子时按「阶段:桶=行数」显式给组成，"
        '例如 \'{"a":{"exploratory":457},"b":{"exploratory":457,"template":630,"other":2}}\'。'
        "字符数仍取自 --data 实测的分桶均值。",
    )
    parser.add_argument("--json", action="store_true", help="只输出机器可读结果")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels_by_id = {}
    if args.labels and args.labels.exists():
        for line in args.labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                label = json.loads(line)
                labels_by_id[int(label["task_id"])] = label

    stats = bucket_stats(rows, labels_by_id, args.bucket_mode, args.max_length)
    linear = fit_linear(CALIBRATION)
    power = fit_power(CALIBRATION)

    print(f"数据：{args.data}（{len(rows)} 行）")
    print(f"标定：M5 48 GB 卡实测 {CALIBRATION}")
    print(f"拟合：s/step = {linear[0]:.1f} + {linear[1]:.5f}×字符  |  幂律指数 {power[1]:.3f}")
    print()
    print("| 桶 | 行数 | 平均字符 | p90 | 最大 | 平均token | 超过 max_length |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, item in sorted(stats.items(), key=lambda kv: -kv[1]["rows"]):
        print(
            f"| {name} | {item['rows']} | {item['mean_chars']:.0f} | {item['p90']} | "
            f"{item['max']} | {item['mean_tokens']:.0f} | {item['over_max_length']} |"
        )

    stages = []
    if args.stage_counts:
        for stage, counts in json.loads(args.stage_counts).items():
            members = [b for b in counts if b in stats]
            total_rows = sum(counts[b] for b in members)
            if not total_rows:
                continue
            avg = sum(stats[b]["mean_chars"] * counts[b] for b in members) / total_rows
            stages.append((stage, {"epochs": 1.0}, members, total_rows, avg))
    elif args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for stage, config in (manifest.get("stages") or {}).items():
            members = [b for b in config["buckets"] if b in stats]
            total_rows = sum(stats[b]["rows"] for b in members)
            if not total_rows:
                continue
            avg = sum(stats[b]["mean_chars"] * stats[b]["rows"] for b in members) / total_rows
            stages.append((stage, config, members, total_rows, avg))
    else:
        avg = statistics.mean(chars(row) for row in rows)
        stages.append(("all", {"epochs": 1.0}, ["*"], len(rows), avg))

    print()
    print("| 阶段 | 桶 | 行数 | 平均字符 | 步数 | s/step（拟合） | 耗时 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    total_hours = 0.0
    for stage, config, members, total_rows, avg in stages:
        steps = math.ceil(total_rows / EFFECTIVE_BATCH)
        timings = predict(avg, linear, power)
        # 三个模型给三条线，取最小/最大作为区间——加了实测点之后幂律可能低于比例模型，
        # 所以不能假定 proportional 一定是下界。
        low, high = min(timings.values()), max(timings.values())
        hours = steps * (low + high) / 2 / 3600
        total_hours += hours
        print(
            f"| {stage} | {'+'.join(members)} | {total_rows} | {avg:.0f} | {steps} | "
            f"{timings['linear']:.0f} | {steps * low / 60:.0f}–{steps * high / 60:.0f} 分钟 |"
        )
    print()
    print(f"合计（不含加载/分词/合并）：约 {total_hours:.1f} 小时")
    lo = sum(
        math.ceil(s[3] / EFFECTIVE_BATCH) * min(predict(s[4], linear, power).values())
        for s in stages
    ) / 3600
    hi = sum(
        math.ceil(s[3] / EFFECTIVE_BATCH) * max(predict(s[4], linear, power).values())
        for s in stages
    ) / 3600
    print(f"区间 {lo:.1f}–{hi:.1f} 小时")

    if args.json:
        print(json.dumps({"stages": [(s[0], s[3], s[4]) for s in stages]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
