"""把 SFT 课程清单（manifest.json）翻译成一张人看的表。

回答两个问题：

1. **三阶段各训多少数据、什么 lr** —— `stages` 是累积的：a ⊂ b ⊂ c，
   后一阶段包含前一阶段的全部桶，只是 lr 更低。
2. **每个桶里是什么题** —— 桶不是按「轨迹类型」分的，是按难度标签：
   * `foundation`：`difficulty == simple`
   * `constraints`：`medium` 且**不需要**换词、**不需要**对比
   * `strategy`：其余（hard，或 medium 里需要换词/对比的）

第 3 条最容易被误解：**「翻页 / 换词 / 多开」这些新采的轨迹几乎全进 `strategy`**，
而 `strategy` 只在最后一阶段（c）才第一次出现。所以新数据的实际曝光是
「1 个 epoch × 最低 lr」。想知道新数据被稀释到什么程度，看这张表就够。

用法::

    python scripts/show_sft_manifest.py data/sft_m10r/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGE_LABEL_DIFFICULTY = {"a": "A 打基础", "b": "B 加约束", "c": "C 加策略", "full": "单阶段全量"}
STAGE_LABEL_BEHAVIOR = {"a": "A 学探索", "b": "B 全量收敛", "full": "单阶段全量"}
BUCKET_LABEL = {
    "foundation": "基础（simple）",
    "constraints": "约束（medium 且不需换词/对比）",
    "strategy": "策略（hard 或需换词/对比）",
    "exploratory": "探索（翻页/换词/开≥2）",
    "template": "模板（单搜单开）",
    "other": "其他",
}


def stage_labels(manifest: dict) -> dict:
    mode = manifest.get("bucket_mode", "difficulty")
    return STAGE_LABEL_BEHAVIOR if mode == "behavior" else STAGE_LABEL_DIFFICULTY


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(manifest: dict) -> str:
    buckets = manifest.get("buckets") or {}
    stages = manifest.get("stages") or {}
    counts = manifest.get("counts") or {}
    lines = [
        f"总行数 {counts.get('rows', 0)} · train {counts.get('train', 0)}"
        f" · validation {counts.get('validation', 0)}"
        f" · 与评测集重叠 {counts.get('evaluation_overlap', 0)}",
        "",
        "## 桶（按行为分：探索 / 模板 / 其他）"
        if manifest.get("bucket_mode") == "behavior"
        else "## 桶（按难度标签分，不是按轨迹类型分）",
        "",
        "| 桶 | 行数 | train | validation | 占池子 |",
        "|---|---:|---:|---:|---:|",
    ]
    total = counts.get("rows") or sum(int(b.get("rows", 0)) for b in buckets.values())
    for name, bucket in buckets.items():
        rows = int(bucket.get("rows", 0))
        share = f"{rows / total:.1%}" if total else "—"
        lines.append(
            f"| {BUCKET_LABEL.get(name, name)} | {rows} |"
            f" {len(bucket.get('train_task_ids') or [])} |"
            f" {len(bucket.get('validation_task_ids') or [])} | {share} |"
        )

    labels = stage_labels(manifest)
    single = manifest.get("stage_plan") == "single"
    lines += [
        "",
        "## 阶段（单阶段：全量分布 1 个 epoch）"
        if single
        else "## 阶段（累积：后一阶段包含前一阶段的全部桶）",
        "",
        "| 阶段 | 包含的桶 | 累计 train 行 | 本阶段新增 | epochs | lr |",
        "|---|---|---:|---:|---:|---:|",
    ]
    previous = 0
    for name, stage in stages.items():
        train_rows = int(stage.get("train_rows", 0))
        selected = " + ".join(stage.get("buckets") or [])
        lines.append(
            f"| {labels.get(name, name)} | {selected} | {train_rows} |"
            f" {train_rows - previous} | {stage.get('epochs', 1.0)} |"
            f" {stage.get('learning_rate')} |"
        )
        previous = train_rows

    flags = manifest.get("review_flags") or {}
    flagged = {name: len(ids or []) for name, ids in flags.items() if ids}
    if flagged:
        lines += ["", "## 复核标记（不阻塞训练，只是提示）", ""]
        for name, count in sorted(flagged.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {name}: {count} 题")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="prepare_sft_curriculum.py 产出的 manifest.json")
    args = parser.parse_args()
    print(render(load(args.manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
