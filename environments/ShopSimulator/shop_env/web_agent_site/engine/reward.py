"""Lightweight, consumer-oriented terminal Reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from web_agent_site.engine.comparators import (
    COMPARATOR_VERSION,
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_category,
    compare_core_functions,
    compare_model,
    comparison,
    load_brand_aliases,
    normalize_text,
)
from web_agent_site.engine.constraints import (
    explicit_price_preference_from_instruction,
)
from web_agent_site.engine.reward_features import (
    REWARD_FEATURE_VERSION,
    normalize_option_text,
)
from web_agent_site.engine.variant_price import (
    VARIANT_PRICE_VERSION,
    candidate_options_for_evaluation,
    compare_required_options,
    resolve_variant_price,
)


# Reward v4 = parser coverage (shopping-budget-parser-v4) + informational
# layers (soft price proximity, process evidence). The hard-gate semantics
# (``category`` + ``budget``, strict ``actual <= upper``) are byte-for-byte
# the v3 semantics; see docs/reward-v4.md for the 修 bug / 改标准 split.
#
# Reward v5 = one numeric branch only: ``wrong_purchase`` stops being the
# constant -0.85 and becomes a soft band driven by ``match_score x budget
# proximity`` (see docs/reward-v5.md). Classification, hard gates,
# ``purchase_success`` and ``reward_valid`` are untouched.
REWARD_VERSION = "shopsimulator-reward-v5"
SUPPORTED_REWARD_VERSIONS = (
    "shopsimulator-reward-v3",
    "shopsimulator-reward-v4",
    "shopsimulator-reward-v5",
)
PROCESS_EVIDENCE_VERSION = "shopping-process-evidence-v1"
DIMENSION_WEIGHTS = {
    "brand": 0.35,
    "model": 0.25,
    "core_functions": 0.25,
    "key_options": 0.15,
}
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_purchase_base": -0.30,
    "partial_purchase_scale": 0.55,
    "partial_purchase_cap": 0.25,
    "graceful_stop": -0.15,
    "early_abstain": -0.35,
    "max_steps": -0.50,
    "repeat_loop": -0.65,
    # v5: wrong_purchase 由常数改为 band [base, cap]。`wrong_purchase` 保留为
    # driver = 0 时的退化值（与 v4 逐字节一致），因此它仍是该档的**地板**。
    # cap 必须严格低于 repeat_loop(-0.65)，否则「买错」会好于「循环/步数耗尽」。
    "wrong_purchase": -0.85,
    "wrong_purchase_base": -0.85,
    "wrong_purchase_scale": 0.20,
    "wrong_purchase_cap": -0.66,
    "reward_unverifiable": 0.0,
}
KNOWN_ACCEPTABLE_MATCH_THRESHOLD = 0.70
KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD = 0.75


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    hard_gates: dict
    weighted_score: float
    evidence: dict

    def to_dict(self):
        payload = asdict(self)
        preference_scoring = self.evidence.get("preference_scoring") or {}
        dimensions = preference_scoring.get("dimensions") or {}
        payload.update(
            {
                "reward_version": REWARD_VERSION,
                "terminal_utility": self.reward,
                "purchase_success": self.reward_type
                in {"gold_purchase", "valid_alternative_purchase"},
                "sampling_invalid": not self.reward_valid,
                "evidence_coverage": float(
                    preference_scoring.get("evidence_coverage", 0.0)
                ),
                "dimension_scores": {
                    name: float(result.get("score", 0.0))
                    for name, result in dimensions.items()
                },
            }
        )
        return payload


def _price_gate(price_resolution: object, goal: dict) -> dict:
    upper = goal.get("price_upper")
    if upper is None:
        return comparison(
            PASS,
            comparator="budget_not_declared_v1",
            required=None,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="instruction",
            evidence=price_resolution,
        )
    if (
        not isinstance(price_resolution, dict)
        or price_resolution.get("status") != PASS
    ):
        return comparison(
            UNVERIFIABLE,
            comparator="variant_price_budget_v1",
            required=upper,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="variant_price",
            evidence=price_resolution,
        )
    try:
        actual = float(price_resolution["price"])
        limit = float(upper)
    except (KeyError, TypeError, ValueError):
        status = UNVERIFIABLE
        actual = price_resolution.get("price")
        limit = upper
    else:
        if (
            not math.isfinite(actual)
            or not math.isfinite(limit)
            or actual < 0
            or limit <= 0
        ):
            status = UNVERIFIABLE
        else:
            status = PASS if actual <= limit else FAIL
    return comparison(
        status,
        comparator="variant_price_budget_v1",
        required=limit,
        actual=actual,
        source_field="variant_price",
        evidence=price_resolution,
    )


def _budget_proximity(budget_gate: dict) -> float:
    """Reward v5 shaping axis: ``min(1, required / actual)``.

    Only the *numeric* terminal reward consumes this value; the hard gate
    status, ``reward_type`` and ``purchase_success`` are unaffected.

    * Budget gate PASS ⇒ ``required / actual >= 1`` is clipped to 1.0, so the
      factor is neutral.
    * Budget not declared (``required is None``) ⇒ 1.0, which degenerates the
      band to a pure ``match_score`` axis.
    * Unverifiable / non-finite inputs ⇒ 1.0 (neutral, never a penalty).
    """
    try:
        required = float(budget_gate.get("required"))
        actual = float(budget_gate.get("actual"))
    except (AttributeError, TypeError, ValueError):
        return 1.0
    if (
        not math.isfinite(required)
        or not math.isfinite(actual)
        or required <= 0
        or actual <= 0
    ):
        return 1.0
    return min(1.0, required / actual)


def _brand_gate(expected: str, product: dict) -> dict:
    aliases = load_brand_aliases()
    canonical = aliases.get(normalize_text(expected), normalize_text(expected))
    candidate_brand_text = normalize_text(
        " ".join(
            str(value)
            for value in (
                product.get("brand"),
                product.get("shop_name"),
            )
            if value
        )
    )
    candidate_aliases = {
        candidate_canonical
        for alias, candidate_canonical in aliases.items()
        if len(alias) >= 2 and alias in candidate_brand_text
    }
    matched = (
        canonical in candidate_aliases
        or normalize_text(expected) in candidate_brand_text
    )
    return comparison(
        PASS if matched else FAIL,
        comparator="explicit_brand_alias_text_v1",
        required=expected,
        actual=candidate_brand_text,
        source_field="brand|shop_name",
        evidence={"alias_match": matched},
    )


def _score_requirements(requirements, evaluator) -> dict:
    results = [evaluator(requirement) for requirement in requirements]
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _option_dimension(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    required = goal.get("required_options_by_key") or {}
    unresolved = goal.get("unresolved_option_requirements") or []
    results = [
        compare_required_options(
            product,
            {axis: requirement},
            selected_options,
        )
        for axis, requirement in required.items()
    ]
    selected_values = {
        normalize_option_text(value)
        for value in (
            selected_options.values()
            if isinstance(selected_options, dict)
            else ()
        )
    }
    results.extend(
        comparison(
            (
                PASS
                if normalize_option_text(item.get("value"))
                in selected_values
                else FAIL
            ),
            comparator="exact_option_value_fallback_v1",
            required=item.get("value"),
            actual=list(
                selected_options.values()
                if isinstance(selected_options, dict)
                else ()
            ),
            source_field="instruction_options",
            evidence=item,
        )
        for item in unresolved
    )
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _preference_dimensions(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    dimensions = {
        "brand": _score_requirements(
            goal.get("expected_brand") or [],
            lambda value: _brand_gate(value, product),
        ),
        "model": _score_requirements(
            goal.get("expected_model") or [],
            lambda value: compare_model(value, product),
        ),
        "core_functions": _score_requirements(
            goal.get("expected_core_functions") or [],
            lambda value: compare_core_functions([value], product),
        ),
        "key_options": _option_dimension(
            product,
            goal,
            selected_options,
        ),
    }
    active_weight = sum(
        DIMENSION_WEIGHTS[name]
        for name, result in dimensions.items()
        if result["active"]
    )
    if active_weight:
        match_score = sum(
            DIMENSION_WEIGHTS[name] * result["score"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
        evidence_coverage = sum(
            DIMENSION_WEIGHTS[name] * result["coverage"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
    else:
        match_score = 1.0
        evidence_coverage = 1.0
    all_satisfied = (
        math.isclose(match_score, 1.0, abs_tol=1.0e-8)
        and math.isclose(evidence_coverage, 1.0, abs_tol=1.0e-8)
    )
    return {
        "weights": DIMENSION_WEIGHTS,
        "active_weight": active_weight,
        "match_score": match_score,
        "evidence_coverage": evidence_coverage,
        "all_satisfied": all_satisfied,
        "dimensions": dimensions,
    }


def _soft_price_preference(goal: dict, price_resolution: object) -> dict:
    """Reward v4 informational layer ("改标准", evidence-only).

    Scores how close the paid price is to the *nominal* value of a soft
    "左右/上下" preference. It never feeds the hard gates, the preference
    match score, or the terminal reward; training and reporting layers may
    consume it (M6/M7).
    """
    nominal = explicit_price_preference_from_instruction(
        goal.get("instruction_text")
    )
    inactive = {
        "active": False,
        "nominal": None,
        "actual": None,
        "proximity": None,
    }
    if nominal is None or not isinstance(price_resolution, dict):
        return inactive
    if price_resolution.get("status") != PASS:
        return inactive
    try:
        actual = float(price_resolution["price"])
    except (KeyError, TypeError, ValueError):
        return inactive
    if not math.isfinite(actual) or actual < 0 or nominal <= 0:
        return inactive
    proximity = max(0.0, 1.0 - abs(actual - nominal) / nominal)
    return {
        "active": True,
        "nominal": float(nominal),
        "actual": actual,
        "proximity": proximity,
    }


def _process_evidence(
    hard_gates: dict,
    preferences: dict,
    soft_price: dict,
    asin_match: bool,
) -> dict:
    """Compact terminal diagnostics for the training side (M3-7, M6 consumer).

    Additive output only: the terminal reward, reward_type classification and
    hard-gate semantics are identical to v3 (P1).
    """
    return {
        "version": PROCESS_EVIDENCE_VERSION,
        "hard_gates": {
            name: gate["status"] for name, gate in hard_gates.items()
        },
        "match_score": float(preferences["match_score"]),
        "evidence_coverage": float(preferences["evidence_coverage"]),
        "dimension_scores": {
            name: float(result.get("score", 0.0))
            for name, result in preferences["dimensions"].items()
        },
        "soft_price_preference": dict(soft_price),
        "target_asin_match": bool(asin_match),
    }


def _evaluate(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: object,
) -> tuple[dict, dict]:
    hard_gates = {
        "category": compare_category(goal.get("category"), product),
        "budget": _price_gate(price_resolution, goal),
    }
    preferences = _preference_dimensions(
        product,
        goal,
        selected_options,
    )
    return hard_gates, preferences


def _explicit_price_resolution(price: object) -> dict:
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = None
    return {
        "status": (
            PASS
            if value is not None
            and math.isfinite(value)
            and value >= 0
            else UNVERIFIABLE
        ),
        "price": value,
        "version": VARIANT_PRICE_VERSION,
        "method": "explicit_test_price",
        "evidence": {},
    }


def evaluate_purchase(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: dict | None = None,
    price: object = None,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    """Score a purchase using two hard gates and continuous soft matching."""
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if price_resolution is None:
        price_resolution = (
            _explicit_price_resolution(price)
            if price is not None
            else resolve_variant_price(product, selected_options)
        )
    hard_gates, preferences = _evaluate(
        product,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
    )
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    soft_price = _soft_price_preference(goal, price_resolution)
    process_evidence = _process_evidence(
        hard_gates,
        preferences,
        soft_price,
        asin_match,
    )
    hard_unverifiable = any(
        gate["status"] == UNVERIFIABLE for gate in hard_gates.values()
    )
    hard_failed = any(
        gate["status"] == FAIL for gate in hard_gates.values()
    )
    if hard_unverifiable:
        reward_type = "reward_unverifiable"
        reward_valid = False
        reward = values[reward_type]
    elif hard_failed:
        reward_type = "wrong_purchase"
        reward_valid = True
        # v5: 档位内 soft 排序。driver = 0 时与 v4 的常数 -0.85 完全一致；
        # cap 保证该档仍严格低于 repeat_loop / max_steps。
        driver = preferences["match_score"] * _budget_proximity(
            hard_gates["budget"]
        )
        reward = min(
            values["wrong_purchase_cap"],
            values["wrong_purchase_base"]
            + values["wrong_purchase_scale"] * driver,
        )
    elif preferences["all_satisfied"]:
        reward_type = (
            "gold_purchase" if asin_match else "valid_alternative_purchase"
        )
        reward_valid = True
        reward = values[reward_type]
    else:
        reward_type = "partial_alternative_purchase"
        reward_valid = True
        reward = min(
            values["partial_purchase_cap"],
            values["partial_purchase_base"]
            + values["partial_purchase_scale"]
            * preferences["match_score"],
        )
    return RewardResult(
        reward=float(reward),
        reward_type=reward_type,
        reward_valid=reward_valid,
        termination_reason=reward_type,
        target_asin_match=asin_match,
        hard_gates=hard_gates,
        weighted_score=float(preferences["match_score"]),
        evidence={
            "reward_feature_version": goal.get("reward_feature_version"),
            "expected_reward_feature_version": REWARD_FEATURE_VERSION,
            "comparator_version": COMPARATOR_VERSION,
            "variant_price_version": VARIANT_PRICE_VERSION,
            "price_resolution": price_resolution,
            "preference_scoring": preferences,
            "soft_price_preference": soft_price,
            "process_evidence": process_evidence,
        },
    )


def evaluate_candidate_eligibility(product: dict, goal: dict) -> dict:
    """Determine whether an opened candidate is sufficiently acceptable to block abstention."""
    selected, option_resolution = candidate_options_for_evaluation(
        product,
        goal.get("required_options_by_key"),
    )
    price_resolution = resolve_variant_price(product, selected)
    hard_gates, preferences = _evaluate(
        product,
        goal,
        selected_options=selected,
        price_resolution=price_resolution,
    )
    hard_pass = all(
        gate["status"] == PASS for gate in hard_gates.values()
    )
    known_acceptable = (
        hard_pass
        and preferences["match_score"]
        >= KNOWN_ACCEPTABLE_MATCH_THRESHOLD
        and preferences["evidence_coverage"]
        >= KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD
    )
    return {
        "status": PASS if known_acceptable else FAIL,
        "known_acceptable": known_acceptable,
        # Compatibility field for the Environment v2.1 session state.
        "known_valid": known_acceptable,
        "selected_options": selected,
        "option_resolution": option_resolution,
        "price_resolution": price_resolution,
        "hard_gates": hard_gates,
        "match_score": preferences["match_score"],
        "evidence_coverage": preferences["evidence_coverage"],
    }


def evaluate_abstain(
    *,
    effective_result_sets: int,
    opened_candidates: int,
    known_acceptable_candidates: int | None = None,
    known_valid_candidates: int | None = None,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    known_count = int(
        known_acceptable_candidates
        if known_acceptable_candidates is not None
        else known_valid_candidates or 0
    )
    eligible = (
        int(effective_result_sets) >= 2
        and int(opened_candidates) >= 2
        and known_count == 0
    )
    reward_type = "graceful_stop" if eligible else "early_abstain"
    return RewardResult(
        reward=float(values[reward_type]),
        reward_type=reward_type,
        reward_valid=True,
        termination_reason=reward_type,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={
            "eligible": eligible,
            "effective_result_sets": int(effective_result_sets),
            "opened_candidates": int(opened_candidates),
            "known_acceptable_candidates": known_count,
        },
    )


def fixed_termination(
    reason: str,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return RewardResult(
        reward=float(values[reason]),
        reward_type=reason,
        reward_valid=True,
        termination_reason=reason,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={},
    )
