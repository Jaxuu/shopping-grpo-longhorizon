#!/usr/bin/env python3
"""让模型在「票数前 k 名」的候选之间做一次**显式比较**，看能不能把 64.5% 抬到 70%。

为什么这条不是集成技巧
----------------------
项目的核心诊断是：这个模型**从不比较候选**——Final-200 里「买之前打开过 ≥2 个商品」
的题是 **0 道**。它看到搜索页就直接买第一件。所以失败不是「不会挑」，是**没做「挑」
这个动作**。把候选缩到 2 个、明确要求它比较，测的就是这个动作本身。

输入只用两样东西：
1. 用户的 `instruction`（模型本来就看得见）；
2. 候选商品的详情页文本（来自模型自己的轨迹观测）。
**不使用** `initial_result.goal_options`（那是答案的一部分）、不使用 `terminal_result.goal`。

判据（跑之前定死）
------------------
Final-200 实测：多数票 129，目标 140。只看票数前 2 名、由完美裁判挑能到 143
（`analyze_ensemble_headroom.py`）。所以：

- 比较器在「多数票判错且正确答案在票数前 2 名」的题里判对 **≥11** ⇒ 总分 ≥140，达成；
- 判对 <11 ⇒ 这条不够，换更强的信号，或者接受 70% 不可达。

`--scope all` 会把比较器套到**全部**有 ≥2 个候选的题上，直接量端到端分数（不再外推），
代价是多花几分钟推理。要判断「会不会把本来对的题改错」，用这个。

用法::

    .venv/bin/python scripts/eval_pairwise_selector.py outputs/evaluation/*/ \\
        --headroom outputs/analysis/headroom.json
    .venv/bin/python scripts/eval_pairwise_selector.py outputs/evaluation/*/ \\
        --headroom outputs/analysis/headroom.json --scope all --out outputs/analysis/pairwise.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HEADROOM = ROOT / "outputs" / "analysis" / "headroom.json"
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
DEFAULT_MODEL = os.environ.get("SERVED_MODEL_NAME", "shopping-agent")
TARGET_RATE = 0.70
# 预登记门槛：14 道「正确答案在前 2 名」的题里必须判对 11 道，总分才到 140。
MIN_HITS = 11


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ens = _load("eval_ensemble", ROOT / "scripts" / "eval_ensemble.py")


def split_identity(key: str) -> tuple[str, dict[str, str]]:
    """`asin|a=1&b=2` → (asin, {"a": "1", "b": "2"})；无规格时选项为空。"""
    asin, _, rest = key.partition("|")
    options: dict[str, str] = {}
    if rest:
        for pair in rest.split("&"):
            name, _, value = pair.partition("=")
            if name:
                options[name] = value
    return asin, options


def product_details(runs: Mapping[str, Sequence[Mapping]]) -> dict[tuple[int, str], str]:
    """(task_id, asin) → 详情页文本。

    优先取「已经选过规格」的那一版（`selected_options` 非空），因为比较器要看到最终
    买到手的到底是什么；没有就取最后一版。
    """
    index: dict[tuple[int, str], str] = {}
    selected: set[tuple[int, str]] = set()
    for rows in runs.values():
        for row in rows:
            task_id = row.get("task_id")
            if task_id is None:
                continue
            task_id = int(task_id)
            for step in row.get("steps") or ():
                text = step.get("observation")
                if not isinstance(text, str) or "page_type: product_detail" not in text:
                    continue
                match = re.search(r"^asin:\s*(\S+)", text, re.MULTILINE)
                if not match:
                    continue
                key = (task_id, match.group(1).strip())
                has_options = not re.search(r"^selected_options:\s*\{\}\s*$", text, re.MULTILINE)
                if key in selected and not has_options:
                    continue
                index[key] = text
                if has_options:
                    selected.add(key)
    return index


def instructions(runs: Mapping[str, Sequence[Mapping]]) -> dict[int, str]:
    """task_id → 用户需求原文。只取 `instruction`，绝不带 goal。"""
    out: dict[int, str] = {}
    for rows in runs.values():
        for row in rows:
            task_id = row.get("task_id")
            text = (row.get("initial_result") or {}).get("instruction")
            if task_id is None or not text:
                continue
            out.setdefault(int(task_id), str(text))
    return out


NOISE_PREFIXES = (
    "可点击的按钮",
    "搜索功能是否可用",
    "[SHOPPING_OBSERVATION_V2]",
    # 详情页里的 selected_options 是「某个模型当时选的」，与本题候选的规格未必一致，
    # 留着会和下面的「本次已选规格」打架。候选的规格统一由那一行给出。
    "selected_options",
)


def clean_detail(text: str, *, max_chars: int = 1600) -> str:
    """去掉详情页里的噪声行（按钮列表很长且与判断无关），并按上限截断。"""
    kept = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(NOISE_PREFIXES)
    ]
    joined = "\n".join(kept).strip()
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars] + "\n…（截断）"
    return joined


def build_prompt(
    instruction: str,
    candidates: Sequence[tuple[str, str, dict[str, str], str]],
    *,
    max_detail_chars: int = 1600,
) -> str:
    """candidates = [(标签, asin, 已选规格, 详情页文本)]。标签是 A/B/C…"""
    labels = " / ".join(label for label, *_ in candidates)
    parts = [
        "你在帮用户完成一次购物。下面是用户的完整需求，以及几款候选商品（都是真实商品页）。",
        f"请判断哪一款最符合需求，只回答一个字母：{labels}。",
        "",
        "【用户需求】",
        instruction.strip(),
        "",
    ]
    for label, asin, options, detail in candidates:
        parts.append(f"【候选 {label}】")
        parts.append(clean_detail(detail, max_chars=max_detail_chars))
        if options:
            chosen = "，".join(f"{key}={value}" for key, value in sorted(options.items()))
            parts.append(f"本次已选规格：{chosen}")
        else:
            parts.append("本次已选规格：无")
        parts.append("")
    parts.append(f"哪一款最符合需求？只回答一个字母：{labels}。")
    return "\n".join(parts)


def parse_choice(text: str, labels: Sequence[str]) -> str | None:
    """从模型输出里取第一个独立出现的标签。取不到返回 None（按判错算，不放水）。"""
    if not text:
        return None
    pattern = r"(?<![A-Za-z])(" + "|".join(re.escape(label) for label in labels) + r")(?![A-Za-z])"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def chat(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 32,
    timeout: int = 120,
) -> str:
    """一次 OpenAI 兼容的 chat 调用。先试关闭 thinking，服务端不认就去掉那个参数重试。"""
    url = base_url.rstrip("/") + "/chat/completions"
    base_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for extra in ({"chat_template_kwargs": {"enable_thinking": False}}, {}):
        payload = {**base_payload, **extra}
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return str(data["choices"][0]["message"].get("content") or "")
        except urllib.error.HTTPError:
            if extra:
                continue
            raise
    return ""


def check_service(base_url: str, api_key: str, model: str, *, timeout: int = 5) -> list[str]:
    """开跑前确认 :8000 上真有服务、且模型名对得上。返回可用 model id 列表。

    没有这一步，vLLM 没起时会在第一题上抛 40 行 traceback，看不出是「服务没开」
    还是「脚本有 bug」。失败一律 SystemExit，把该敲的命令打出来。
    """
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"连不上 {base_url}（{exc.reason}）—— vLLM 没在跑。\n"
            "先起服务（2B 权重 4.4GB，约 2–3 分钟；就绪判据是 /v1/models 返回带 id 的 JSON）：\n"
            "  bash scripts/serve_model.sh outputs/models/sft-m10r/stage-full/merged\n"
        ) from None
    ids = [str(item.get("id")) for item in (data.get("data") or []) if item.get("id")]
    if model not in ids:
        hint = f"加 --model {ids[0]} 重跑。" if ids else "服务起来了但没加载任何模型。"
        raise SystemExit(f"服务在跑，但没有模型 {model!r}。实际可用：{ids}\n{hint}")
    return ids


def _top_keys(votes: Mapping[str, int], top_k: int) -> list[str]:
    """票数降序、同票按身份字典序 —— 不依赖 JSON 里的键顺序。"""
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _ in ranked][:top_k]


def disputed_tasks(headroom: Mapping, *, top_k: int) -> list[dict]:
    """从 headroom.json 里取「多数票判错、且正确答案在票数前 k 名」的题。"""
    flag = "correct_in_top2" if top_k <= 2 else "correct_in_top3"
    picked = []
    for item in headroom.get("diagnosis", {}).get("wrong_tasks", []):
        if not item.get(flag):
            continue
        keys = _top_keys(item.get("votes", {}), top_k)
        if len(keys) < 2:
            continue
        picked.append(
            {"task_id": item["task_id"], "candidates": keys, "good_keys": item["good_keys"]}
        )
    return picked


def all_tasks(headroom: Mapping, *, top_k: int) -> list[dict]:
    """所有至少有 2 个不同候选的题（含多数票本来就判对的），用来量会不会改错。"""
    picked = []
    for item in headroom.get("diagnosis", {}).get("wrong_tasks", []):
        keys = _top_keys(item.get("votes", {}), top_k)
        if len(keys) >= 2:
            picked.append(
                {
                    "task_id": item["task_id"],
                    "candidates": keys,
                    "good_keys": item["good_keys"],
                    "plurality_wrong": True,
                }
            )
    for item in headroom.get("diagnosis", {}).get("right_tasks", []):
        keys = _top_keys(item.get("votes", {}), top_k)
        if len(keys) >= 2:
            picked.append(
                {
                    "task_id": item["task_id"],
                    "candidates": keys,
                    "good_keys": item["good_keys"],
                    "plurality_wrong": False,
                }
            )
    return picked


def run_selector(
    tasks: Sequence[Mapping],
    *,
    details: Mapping[tuple[int, str], str],
    instruction_by_task: Mapping[int, str],
    base_url: str,
    api_key: str,
    model: str,
    top_k: int,
    verbose: bool = True,
) -> list[dict]:
    results = []
    for task in tasks:
        task_id = task["task_id"]
        instruction = instruction_by_task.get(task_id)
        if not instruction:
            continue
        labels = [chr(ord("A") + i) for i in range(len(task["candidates"]))]
        candidates = []
        missing = False
        for label, key in zip(labels, task["candidates"], strict=False):
            asin, options = split_identity(key)
            detail = details.get((task_id, asin))
            if detail is None:
                missing = True
                break
            candidates.append((label, asin, options, detail))
        if missing:
            continue
        prompt = build_prompt(instruction, candidates)
        answer = chat(
            prompt, base_url=base_url, api_key=api_key, model=model, max_tokens=8 + 4 * top_k
        )
        choice = parse_choice(answer, labels)
        picked = task["candidates"][labels.index(choice)] if choice else None
        results.append(
            {
                "task_id": task_id,
                "choice": choice,
                "picked": picked,
                "hit": bool(picked is not None and picked in task["good_keys"]),
                "plurality_wrong": task.get("plurality_wrong", True),
                "answer": answer.strip()[:120],
            }
        )
        if verbose:
            mark = "✅" if results[-1]["hit"] else "❌"
            said = results[-1]["answer"]
            tail = "" if said == choice else f"  「{said}」"
            print(f"{mark} task {task_id} → {choice or '解析失败'}{tail}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--headroom", type=Path, default=DEFAULT_HEADROOM)
    parser.add_argument("--reference", type=Path, default=ens.DEFAULT_REFERENCE)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--scope", choices=("disputed", "all"), default="disputed")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过 /v1/models 探活（第三方网关不列模型时用）",
    )
    args = parser.parse_args(argv)

    headroom = json.loads(args.headroom.read_text(encoding="utf-8"))
    reference_set = set(ens.probe.task_ids_of(args.reference))
    runs = ens.resolve_runs(args.runs, min_tasks=args.min_tasks)
    for label in list(runs):
        ids = {int(row["task_id"]) for row in runs[label] if row.get("task_id") is not None}
        if len(ids & reference_set) / max(1, len(reference_set)) < args.min_overlap:
            del runs[label]
    if not runs:
        raise SystemExit("没有可用的评测目录。")

    tasks = disputed_tasks(headroom, top_k=args.top_k)
    if args.scope == "all":
        tasks = all_tasks(headroom, top_k=args.top_k)
    if not tasks:
        raise SystemExit("headroom.json 里没有符合条件的题（先跑 analyze_ensemble_headroom.py）。")

    print(f"# 候选比较器：{len(tasks)} 题，取票数前 {args.top_k} 名，模型 {args.model}")
    print()
    if not args.skip_preflight:
        check_service(args.base_url, args.api_key, args.model)

    results = run_selector(
        tasks,
        details=product_details(runs),
        instruction_by_task=instructions(runs),
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
    if not results:
        raise SystemExit("一题都没跑成（候选商品详情页在轨迹里找不到？）")

    hit = sum(1 for item in results if item["hit"])
    n = len(results)
    print()
    print(f"## 结果：判对 {hit}/{n}（{hit / n:.1%}）")

    if args.scope == "disputed":
        base = headroom.get("rules", {})
        plurality = next((v for k, v in base.items() if "多数票（平票算失败）" in k), None)
        if plurality is not None:
            projected = plurality + hit
            total = headroom["n"]
            target = round(total * TARGET_RATE)
            print()
            print(
                f"多数票 {plurality}/{total} + 比较器救回 {hit} ⇒ 投影 **{projected}/{total} "
                f"= {projected / total:.1%}**（目标 {target}）"
            )
            if hit >= MIN_HITS:
                print(f"**达标**：判对 {hit} ≥ 预登记门槛 {MIN_HITS}。")
            else:
                print(f"**未达标**：判对 {hit} < 门槛 {MIN_HITS}，这条不够，不要外推。")
            print()
            print("> 注意这是**上界**：只在「多数票判错」的题上跑，没有量会不会把本来对的题改错。")
            print("> 要端到端数字，加 `--scope all`。")
    else:
        wrong_fixed = sum(1 for item in results if item["hit"] and item["plurality_wrong"])
        broken = sum(1 for item in results if not item["hit"] and not item["plurality_wrong"])
        print()
        print(f"救回 {wrong_fixed} 题，改错 {broken} 题 ⇒ 净 **{wrong_fixed - broken:+d}**")
        print("（改错 = 多数票本来判对、被比较器换掉了。这一列才是能不能真上线的关键。）")

    if args.out:
        payload = {
            "scope": args.scope,
            "top_k": args.top_k,
            "model": args.model,
            "hit": hit,
            "n": n,
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print()
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
