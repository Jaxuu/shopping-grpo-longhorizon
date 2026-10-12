"""Task-pool admission contract for ShopSimulator training pools.

``shopping-task-admission-v1`` is the single gate that a task must pass before
it may enter a training pool. It exists because the M1 audit
(``audit/report.md``) showed the same class of defects in every pool that was
built without a gate:

* SFT carries ``gold_purchase`` verification, so it only leaked 2 defect
  classes (zero-price, cross-pool overlap);
* the GRPO pool carried no gate at all, so it leaked all of them
  (unsolvable specs, zero-price, gold that cannot win, overlap).

Applying one shared contract at every pool boundary is the fix; adding
per-defect patches to each builder is not.

Gate status:

* **A1** spec resolvable      -- ``unresolved_option_requirements`` is empty;
* **A2** price evaluable      -- the gold variant price resolves, and the
  purchase is not ``reward_unverifiable``;
* **A3** non-zero price       -- no variant is priced at or below zero;
* **A4** gold can win         -- buying the gold product yields
  ``gold_purchase`` under the current reward version;
* **A6** no cross-pool overlap -- the ``task_id`` is not in the held-out set.

**A5 (text normalisation) is reported as a warning, not a rejection.** The M1
follow-up check showed the scoring path already folds ``\\xa0``/full-width
spaces through ``normalize_option_text`` (NFKC), so a task such as
``evaluation:3647`` scores correctly today. Whitespace differences are a
data-representation smell, not a functional defect, and are therefore recorded
in ``warnings`` rather than blocking admission.

A4 is version-sensitive: it must be re-evaluated after any reward change,
because a wider budget parser activates the budget gate on tasks where it
previously passed by default.
"""

from __future__ import annotations

import re
import unicodedata

from web_agent_site.engine.reward import REWARD_VERSION, evaluate_purchase
from web_agent_site.engine.variant_price import (
    candidate_options_for_evaluation,
    resolve_variant_price,
)


CONTRACT_VERSION = "shopping-task-admission-v1"

_NON_ASCII_WHITESPACE = re.compile(
    r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000\ufeff]"
)


def has_non_ascii_whitespace(*values) -> bool:
    """True when any value still carries NBSP / full-width / zero-width space."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            if has_non_ascii_whitespace(*value):
                return True
            continue
        if _NON_ASCII_WHITESPACE.search(str(value)):
            return True
    return False


def _zero_price(product) -> bool:
    entries = [
        entry
        for values in (product.get("customization_options") or {}).values()
        for entry in (values or [])
        if isinstance(entry, dict)
    ]
    variant_prices = [entry.get("price") for entry in entries]
    pricing = list(product.get("pricing") or [])
    zero_variant = any(
        price is not None and price <= 0 for price in variant_prices
    )
    zero_pricing = any(price is not None and price <= 0 for price in pricing)
    return zero_variant or zero_pricing


def admit_task(task_id, goal, product, *, held_out_task_ids=()) -> dict:
    """Judge one task against ``shopping-task-admission-v1``.

    ``goal`` must already carry the compiled reward features
    (``compile_reward_features``), ``price_upper`` and ``category``.
    Returns ``{"admitted", "reasons", "warnings", "detail"}`` and never mutates
    its inputs.
    """
    goal = goal if isinstance(goal, dict) else {}
    product = product if isinstance(product, dict) else {}
    held_out = {int(value) for value in (held_out_task_ids or ())}

    reasons: list[str] = []
    warnings: list[str] = []
    detail: dict = {"contract": CONTRACT_VERSION, "reward_version": REWARD_VERSION}

    # A1 -- the task's required spec must exist on the gold product.
    unresolved = list(goal.get("unresolved_option_requirements") or [])
    detail["unresolved_options"] = unresolved
    if unresolved:
        reasons.append("A1_spec_unresolvable")

    # A3 -- a zero price makes the budget gate vacuous.
    zero_price = _zero_price(product)
    detail["zero_price"] = zero_price
    if zero_price:
        reasons.append("A3_zero_price")

    # A2 / A4 -- the gold purchase must be evaluable and must actually win.
    selected, _selection = candidate_options_for_evaluation(
        product, goal.get("required_options_by_key")
    )
    price_resolution = resolve_variant_price(product, selected)
    detail["price_status"] = price_resolution.get("status")
    detail["price_method"] = price_resolution.get("method")
    if price_resolution.get("status") != "pass":
        reasons.append("A2_price_unverifiable")
        detail["gold_reward_type"] = None
    else:
        result = evaluate_purchase(
            product,
            goal,
            selected_options=selected,
            price_resolution=price_resolution,
        )
        detail["gold_reward_type"] = result.reward_type
        detail["gold_hard_gates"] = {
            name: gate["status"] for name, gate in result.hard_gates.items()
        }
        if result.reward_type == "reward_unverifiable":
            reasons.append("A2_price_unverifiable")
        elif result.reward_type != "gold_purchase":
            reasons.append("A4_gold_cannot_win")

    # A5 -- representation smell only (see module docstring).
    if has_non_ascii_whitespace(
        goal.get("instruction_text"),
        goal.get("attributes"),
        goal.get("goal_options"),
    ):
        warnings.append("A5_non_ascii_whitespace")

    # A6 -- training pools must stay disjoint.
    detail["held_out"] = int(task_id) in held_out
    if int(task_id) in held_out:
        reasons.append("A6_cross_pool_overlap")

    return {
        "task_id": int(task_id),
        "admitted": not reasons,
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "detail": detail,
    }


__all__ = [
    "CONTRACT_VERSION",
    "admit_task",
    "has_non_ascii_whitespace",
]
