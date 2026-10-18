"""`scripts/audit_ratio_noise.py` 的纯 CPU 单元测试。

固定四件事：
  1. `norm_cdf` 的对称性与已知点；
  2. `predicted_clipfrac` 随 σ 单调递增（噪声越大，尾部越厚）；
  3. `summarize` 的判读分档：实测 clipfrac 远小于 k3 预测、又远大于 k1 预测 ⇒ 重尾；
  4. `load_steps` 只收 `optimizer_step` 且有 `ppo_kl` / `pg_clipfrac` 的记录。
"""

from __future__ import annotations

import json
import math

import pytest

from scripts.audit_ratio_noise import (
    DEFAULT_CLIP_HIGH,
    load_steps,
    main,
    norm_cdf,
    predicted_clipfrac,
    summarize,
)


def test_norm_cdf_known_points():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.0) == pytest.approx(0.8413447, abs=1e-6)
    assert norm_cdf(-1.0) == pytest.approx(1.0 - 0.8413447, abs=1e-6)
    assert norm_cdf(-10.0) < 1e-20


def test_predicted_clipfrac_monotone():
    values = [predicted_clipfrac(sigma, DEFAULT_CLIP_HIGH) for sigma in (0.001, 0.01, 0.05, 0.2)]
    assert values == sorted(values)
    assert predicted_clipfrac(0.0, DEFAULT_CLIP_HIGH) == 0.0


def test_summarize_flags_heavy_tail():
    # 实测：kl≈0.0035、clipfrac≈5.5e-4、每步 ~96k token（第一轮的真实量级）
    steps = [
        {"step": float(i), "kl": 0.0035, "clipfrac": 5.5e-4, "tokens": 96000.0}
        for i in range(1, 11)
    ]
    summary = summarize(steps)
    assert summary["steps"] == 10
    assert summary["ratio_k3"] > 2.0  # k3 预测远高于实测
    assert summary["ratio_k1"] < 0.5  # k1 预测远低于实测
    assert "重尾" in summary["verdict"]
    assert summary["tail_tokens_per_step"] == pytest.approx(52.8)


def test_summarize_single_model_fit():
    # 构造一个「k1 自洽」的点：kl 很小、clipfrac 与 k1 预测同量级
    kl = 0.02
    sigma_k1 = kl * math.sqrt(math.pi / 2.0)
    clipfrac = predicted_clipfrac(sigma_k1, DEFAULT_CLIP_HIGH)
    steps = [{"step": 1.0, "kl": kl, "clipfrac": clipfrac, "tokens": 1000.0}]
    summary = summarize(steps)
    assert 0.5 <= summary["ratio_k1"] <= 2.0


def test_load_steps_filters_records(tmp_path):
    path = tmp_path / "training_diagnostics.jsonl"
    records = [
        {"event": "generation_batch", "global_step": 1, "metrics": {"actor/ppo_kl": 1.0}},
        {
            "event": "optimizer_step",
            "global_step": 1,
            "metrics": {"actor/ppo_kl": 0.003, "actor/pg_clipfrac": 5e-4, "perf/total_num_tokens": 96000},
        },
        {"event": "optimizer_step", "global_step": 2, "metrics": {"actor/ppo_kl": 0.004}},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    steps = load_steps(path)
    assert len(steps) == 1
    assert steps[0]["kl"] == pytest.approx(0.003)
    assert steps[0]["tokens"] == pytest.approx(96000)


def test_load_steps_empty_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_steps(path)


def test_main_cross_run_table(tmp_path):
    for name, kl, clipfrac in (("run-a", 0.0033, 5.5e-4), ("run-b", 0.0035, 1.04e-3)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "event": "optimizer_step",
                    "global_step": 1,
                    "metrics": {
                        "actor/ppo_kl": kl,
                        "actor/pg_clipfrac": clipfrac,
                        "perf/total_num_tokens": 96000,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    out = tmp_path / "audit.md"
    assert main([str(tmp_path / "run-a.jsonl"), str(tmp_path / "run-b.jsonl"), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "跨轮对比" in text
    assert "ratio 噪声审计" in text
