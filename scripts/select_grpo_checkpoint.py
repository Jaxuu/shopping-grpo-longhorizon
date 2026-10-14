#!/usr/bin/env python3
"""按验证集指标挑选 GRPO checkpoint，而不是默认导出最后一步。

训练时 veRL 每 `trainer.test_freq` 步在验证集上评一次，验证标量随
`optimizer_step` 事件写进 `<output>/training_diagnostics.jsonl` 的 `metrics` 字段。
本脚本把每个验证点抽出来排序，选出验证分最高的 checkpoint。

`docs/grpo.md` 明确要求不要假定最后一步最好；RL 后期常见熵塌缩 / 策略跑偏，
取最后一步会白丢分数。**不要用评测集（Final-200）来做这个选择** —— 那是考卷，
用它选题等于信息泄漏。

用法：

    # 先看 diagnostics 里到底有哪些 val 指标（键名由 veRL 决定，第一步先确认）
    python scripts/select_grpo_checkpoint.py outputs/models/grpo --list-metrics

    # 选 checkpoint
    python scripts/select_grpo_checkpoint.py outputs/models/grpo

    # 机器可读
    python scripts/select_grpo_checkpoint.py outputs/models/grpo --json

判据分三层：① 主指标落在「最高分 − 1/val_size」带宽内的 step 视为并列；
② 并列时由副指标（同一验证集上的连续信号）分出高低；
③ 副指标亦并列则取步数最大者（训练更充分）。验证集只有 `--val-size` 条，
1/val_size 就是「差一条题」的量级，所以第一层不能用严格相等来判并列。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DIAGNOSTICS_NAME = "training_diagnostics.jsonl"
VAL_PREFIX = "val/"
DEFAULT_VAL_SIZE = 49
DEFAULT_EXPORT_OUTPUT = "outputs/models/grpo-merged"


def _export_target(export_output: Path | None) -> str:
    """导出目标路径统一用正斜杠，因为它会被直接拼进 bash 命令。"""
    return DEFAULT_EXPORT_OUTPUT if export_output is None else export_output.as_posix()

# 主指标自动推断顺序：优先"任务成功率"口径，其次 reward 均值。
PRIMARY_PREFERENCES = (
    "val/reward/purchase_success_rate",
    "val/reward/shaped_mean",
    "val/reward/terminal_utility_mean",
    "val/reward/full_mean",
    "val/score",
    "val/accuracy",
)

# 上面没命中时按关键词兜底匹配。
PRIMARY_KEYWORDS = ("purchase_success", "success_rate", "accuracy", "score", "acc")

SECONDARY_PREFERENCES = (
    "val/reward/shaped_mean",
    "val/reward/terminal_utility_mean",
    "val/trajectory/done_rate",
    "val/reward/full_mean",
)


def _scalar(value: Any) -> float | None:
    """只接受有限标量；bool 与 list 一律拒绝。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def load_val_metrics(diagnostics: Path) -> dict[int, dict[str, float]]:
    """返回 `{step: {val_metric: value}}`，同一步出现多条记录时取最后一条。"""
    if not diagnostics.is_file():
        raise SystemExit(f"训练诊断文件不存在：{diagnostics}")
    per_step: dict[int, dict[str, float]] = {}
    with diagnostics.open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{diagnostics}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict) or record.get("event") != "optimizer_step":
                continue
            step = record.get("global_step")
            if not isinstance(step, int):
                continue
            raw_metrics = record.get("metrics")
            if not isinstance(raw_metrics, dict):
                continue
            values: dict[str, float] = {}
            for key, raw_value in raw_metrics.items():
                if not isinstance(key, str) or not key.startswith(VAL_PREFIX):
                    continue
                number = _scalar(raw_value)
                if number is not None:
                    values[key] = number
            if values:
                per_step[step] = values
    return per_step


def available_metrics(per_step: dict[int, dict[str, float]]) -> dict[str, list[int]]:
    """返回 `{metric: [出现过的 step...]}`（step 升序）。"""
    seen: dict[str, list[int]] = {}
    for step in sorted(per_step):
        for key in per_step[step]:
            seen.setdefault(key, []).append(step)
    return seen


def pick_primary(available: dict[str, list[int]]) -> str | None:
    for key in PRIMARY_PREFERENCES:
        if key in available:
            return key
    lowered = sorted((key.lower(), key) for key in available)
    for keyword in PRIMARY_KEYWORDS:
        for low, original in lowered:
            if keyword in low:
                return original
    return None


def pick_secondary(available: dict[str, list[int]], primary: str | None) -> str | None:
    for key in SECONDARY_PREFERENCES:
        if key in available and key != primary:
            return key
    return None


def build_rows(
    per_step: dict[int, dict[str, float]], output: Path
) -> list[dict[str, Any]]:
    rows = []
    for step in sorted(per_step):
        actor = output / f"global_step_{step}" / "actor"
        rows.append(
            {
                "step": step,
                "metrics": per_step[step],
                # 统一正斜杠：这条路径会被直接拼进 bash 导出命令。
                "checkpoint": actor.as_posix(),
                "checkpoint_available": actor.is_dir(),
            }
        )
    return rows


def select_best(
    rows: list[dict[str, Any]],
    primary: str,
    tie_band: float,
    secondary: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """三层决胜，返回 `(选中的 row, 主指标并列集合, 决胜依据)`。

    1. 主指标落在「最高分 − tie_band」内的 step 构成并列集合；
    2. 并列时若副指标可用，取副指标严格最高者（副指标是同一验证集上的连续信号，
       分辨率高于 0/1 成功率，比「取靠后」更能分辨只差一条题的两个点）；
    3. 副指标也并列（或不可用）时取步数最大者 —— 训练更充分。
    """
    candidates = [
        row for row in rows if primary in row["metrics"] and row["checkpoint_available"]
    ]
    if not candidates:
        return None, [], "none"
    best = max(row["metrics"][primary] for row in candidates)
    tied = [row for row in candidates if row["metrics"][primary] >= best - tie_band]
    if len(tied) == 1:
        return tied[0], tied, "primary"
    if secondary:
        scored = [row for row in tied if secondary in row["metrics"]]
        if scored:
            best_secondary = max(row["metrics"][secondary] for row in scored)
            finalists = [
                row
                for row in scored
                if row["metrics"][secondary] >= best_secondary - 1e-12
            ]
            if len(finalists) < len(tied):
                return max(finalists, key=lambda row: row["step"]), tied, "secondary"
    return max(tied, key=lambda row: row["step"]), tied, "step"


def _format_count(value: float, val_size: int) -> str:
    """把成功率折算成 `k/val_size`，便于肉眼判断是不是差一条题。"""
    scaled = value * val_size
    if abs(scaled - round(scaled)) < 1e-6:
        return f"{round(scaled)}/{val_size}"
    return "—"


def render_report(
    rows: list[dict[str, Any]],
    primary: str,
    secondary: str | None,
    chosen: dict[str, Any] | None,
    tied: list[dict[str, Any]],
    decided_by: str,
    tie_band: float,
    val_size: int,
    export_output: Path | None,
) -> str:
    lines: list[str] = []
    header = f"{'step':>6}  {primary:<44} {'命中':>8}"
    if secondary:
        header += f"  {secondary:<40}"
    header += "  checkpoint"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        primary_value = row["metrics"].get(primary)
        primary_text = "—" if primary_value is None else f"{primary_value:.4f}"
        hits = "—" if primary_value is None else _format_count(primary_value, val_size)
        line = f"{row['step']:>6}  {primary_text:<44} {hits:>8}"
        if secondary:
            secondary_value = row["metrics"].get(secondary)
            secondary_text = "—" if secondary_value is None else f"{secondary_value:.4f}"
            line += f"  {secondary_text:<40}"
        line += "  " + ("有" if row["checkpoint_available"] else "缺失")
        lines.append(line)

    lines.append("")
    missing = [row["step"] for row in rows if not row["checkpoint_available"]]
    if missing:
        lines.append(
            f"注意：step {missing} 有验证分但没有 `global_step_*/actor`，已排除。"
            "检查 `trainer.save_freq` 是否 ≤ 且对齐 `trainer.test_freq`。"
        )
        lines.append("")

    if chosen is None:
        lines.append("没有可用的验证点（需要主指标存在且 checkpoint 齐全）。")
        return "\n".join(lines)

    tied_steps = [row["step"] for row in tied]
    lines.append(f"主指标：{primary}")
    if secondary:
        lines.append(f"副指标：{secondary}")
    lines.append(f"并列带宽：±{tie_band:.4f}（= 1/{val_size}，约一条验证题）")
    if decided_by == "primary":
        lines.append("决胜：主指标唯一最高，无并列")
    elif decided_by == "secondary":
        lines.append(f"决胜：主指标并列 {tied_steps} → 由副指标 {secondary} 分出高低")
    elif decided_by == "step":
        lines.append(f"决胜：主指标并列 {tied_steps} → 副指标亦并列 → 取步数最大者")
    else:
        lines.append(f"决胜：{decided_by}")
    lines.append("")
    chosen_hits = _format_count(chosen["metrics"][primary], val_size)
    lines.append(
        f"选定 step {chosen['step']}：{primary} = {chosen['metrics'][primary]:.4f}"
        + (f"（{chosen_hits}）" if chosen_hits != "—" else "")
    )
    if secondary and secondary in chosen["metrics"]:
        lines.append(f"            {secondary} = {chosen['metrics'][secondary]:.4f}")
    lines.append("")
    lines.append("导出命令：")
    lines.append(
        f"  bash scripts/export_grpo.sh {chosen['checkpoint']} {_export_target(export_output)}"
    )
    return "\n".join(lines)


def build_payload(
    rows: list[dict[str, Any]],
    primary: str,
    secondary: str | None,
    chosen: dict[str, Any] | None,
    tied: list[dict[str, Any]],
    decided_by: str,
    tie_band: float,
    val_size: int,
    export_output: Path | None,
) -> dict[str, Any]:
    target = _export_target(export_output)
    return {
        "primary_metric": primary,
        "secondary_metric": secondary,
        "decided_by": decided_by,
        "tie_band": tie_band,
        "val_size": val_size,
        "rows": rows,
        "tied_steps": [row["step"] for row in tied],
        "selected_step": None if chosen is None else chosen["step"],
        "selected_checkpoint": None if chosen is None else chosen["checkpoint"],
        "selected_metrics": None if chosen is None else chosen["metrics"],
        "export_command": (
            None
            if chosen is None
            else f"bash scripts/export_grpo.sh {chosen['checkpoint']} {target}"
        ),
    }


def render_metric_list(
    available: dict[str, list[int]], primary: str | None, secondary: str | None
) -> str:
    lines = [f"发现 {len(available)} 个 `val/` 标量指标：", ""]
    width = max(len(key) for key in available)
    for key in sorted(available):
        steps = available[key]
        tag = ""
        if key == primary:
            tag = "  <- 主指标"
        elif key == secondary:
            tag = "  <- 副指标"
        lines.append(f"  {key:<{width}}  {len(steps):>3} 个验证点 {steps}{tag}")
    lines.append("")
    lines.append("若自动推断的主指标不对，用 `--primary-metric <键名>` 指定。")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output",
        type=Path,
        help="训练输出目录（含 training_diagnostics.jsonl 与 global_step_*）",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help=f"诊断文件路径，默认 <output>/{DIAGNOSTICS_NAME}",
    )
    parser.add_argument("--primary-metric", help="主指标键名，默认自动推断")
    parser.add_argument("--secondary-metric", help="副指标键名，默认自动推断")
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help=f"验证集条数，用于折算命中数与并列带宽（默认 {DEFAULT_VAL_SIZE}）",
    )
    parser.add_argument(
        "--tie-band",
        type=float,
        help="并列带宽，默认 1/val-size；主指标落在最高分-带宽内的 step 视为并列",
    )
    parser.add_argument(
        "--export-output",
        type=Path,
        help="导出目标目录，仅用于打印命令，默认 outputs/models/grpo-merged",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="只列出 diagnostics 里所有 val/ 指标及覆盖的验证点，不选择",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser()
    diagnostics = (args.diagnostics or output / DIAGNOSTICS_NAME).expanduser()

    if args.val_size <= 0:
        raise SystemExit("--val-size 必须为正整数")
    tie_band = args.tie_band if args.tie_band is not None else 1.0 / args.val_size
    if tie_band < 0:
        raise SystemExit("--tie-band 不能为负")

    per_step = load_val_metrics(diagnostics)
    if not per_step:
        raise SystemExit(
            f"{diagnostics} 里没有任何带 `{VAL_PREFIX}` 指标的 optimizer_step 记录。\n"
            "  可能原因：① `trainer.test_freq` 未触发（设成 0 或大于总步数）；"
            "② 训练尚未跑到第一个验证点；③ 该步被 dynamic sampling 跳过（无 optimizer_step）。"
        )
    available = available_metrics(per_step)
    primary = args.primary_metric or pick_primary(available)
    if primary is None:
        raise SystemExit(
            "无法自动推断主指标，请用 --primary-metric 指定。可用键：\n  "
            + "\n  ".join(sorted(available))
        )
    if primary not in available:
        raise SystemExit(
            f"主指标 {primary} 不在 diagnostics 里。可用键：\n  "
            + "\n  ".join(sorted(available))
        )
    secondary = args.secondary_metric or pick_secondary(available, primary)
    if args.secondary_metric and args.secondary_metric not in available:
        raise SystemExit(f"副指标 {args.secondary_metric} 不在 diagnostics 里")

    if args.list_metrics:
        print(render_metric_list(available, primary, secondary))
        return

    rows = build_rows(per_step, output)
    chosen, tied, decided_by = select_best(rows, primary, tie_band, secondary)
    if args.json:
        print(
            json.dumps(
                build_payload(
                    rows,
                    primary,
                    secondary,
                    chosen,
                    tied,
                    decided_by,
                    tie_band,
                    args.val_size,
                    args.export_output,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(
        render_report(
            rows,
            primary,
            secondary,
            chosen,
            tied,
            decided_by,
            tie_band,
            args.val_size,
            args.export_output,
        )
    )


if __name__ == "__main__":
    main()
