"""Pure, reproducible goal helpers for Environment v2."""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata


CONSTRAINT_CONTRACT_VERSION = "shopping-task-constraints-v1"

# Budget parser contract version. Bumped from v3 to v4: the *coverage* of the
# natural-language price parser is extended (Chinese numerals, colloquial units,
# prefix-less ranges, unit prices, non-ASCII whitespace). The hard-gate
# *semantics* are unchanged: an upper bound is still an upper bound and a
# lower-bound-only phrasing still yields ``None``.
BUDGET_PARSER_VERSION = "shopping-budget-parser-v4"

_CN_DIGITS = {
    "零": 0, "〇": 0,
    "一": 1, "壹": 1,
    "二": 2, "两": 2, "贰": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}
_CN_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_CN_CHARS = "".join(_CN_DIGITS) + "".join(_CN_UNITS) + "万亿"

_AMOUNT = rf"(?:\d+(?:\.\d+)?|[{_CN_CHARS}]+)"
_SCALE = r"(?:万|千|[kK])"
_PRICE_UNIT = r"(?:块钱|块|元|圆|钱)"
_BUDGET_CUE = (
    r"(?:预算|价格|售价|价钱|价位|总价|费用|成本|花费|标价|报价|单价|"
    r"开销|定金|到手价|预算价)"
)
_UPPER_PREFIX = (
    r"(?:不超过|不能超过|不要超过|不得超过|不可超过|别超过|别超|不超|"
    r"不高于|最高|至多|封顶|(?<!不)低于|少于|不到|小于|不贵于|控制在)"
)
_UPPER_SUFFIX = r"(?:以内|以下|之内|以里|内)"
_AROUND_SUFFIX = r"(?:左右|上下|出头|多一些|来块|来)"
_LOWER_PREFIX = r"(?:不低于|不少于|至少|最低|(?<!不)高于|超过|多于)"
_LOWER_SUFFIX = r"(?:以上|起|起步)"
_PER_UNIT = r"每(?:[^\s\d，。；]{0,6})?"


def _clean_annotation_values(value):
    """Normalize task-authored annotations without inferring new constraints."""
    if not isinstance(value, list):
        return [], False
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned, True


def compile_task_constraint_contract(instruction_record):
    """Compile the task's existing annotations into a fail-closed contract.

    Environment v2 uses only fields already attached to the current natural
    language instruction. It deliberately does not infer brand/model or read
    extra attributes from the target product.
    """
    if not isinstance(instruction_record, dict):
        instruction_record = {}
    attributes, attributes_valid = _clean_annotation_values(
        instruction_record.get("attributes")
    )
    options, options_valid = _clean_annotation_values(
        instruction_record.get("instruction_options")
    )
    instruction_valid = bool(
        re.sub(r"\s+", "", str(instruction_record.get("instruction") or ""))
    )
    complete = instruction_valid and attributes_valid and options_valid
    return {
        "hard_constraints": {
            "complete": complete,
            "contract_version": CONSTRAINT_CONTRACT_VERSION,
            "annotation_source": "instruction.attributes",
            "core_functions": attributes,
            "brand": [],
            "model": [],
            "key_specs": [],
            # Options continue to be checked by Reward v2's key_options gate.
            "annotated_option_count": len(options),
        },
        # The current task data does not label hard versus soft preferences.
        # Do not guess that distinction from keywords.
        "weighted_preferences": [],
    }


# --------------------------------------------------------------------------- #
# Budget parsing (shopping-budget-parser-v4)
# --------------------------------------------------------------------------- #
def _normalize_price_text(instruction):
    """Fold non-ASCII whitespace and separators before pattern matching."""
    text = unicodedata.normalize("NFKC", str(instruction or ""))
    text = re.sub(r"[\s\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000\ufeff]+", " ", text)
    return text.replace(",", "").replace("，", "").strip()


def _cn_numeral_to_int(amount):
    """Convert an Arabic or Chinese numeral token into a number."""
    if amount is None:
        return None
    text = str(amount).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    # Chinese shorthand omits the trailing unit: 两千五 = 2500, 一百二 = 120.
    if len(text) >= 2 and text[-1] in _CN_DIGITS and text[-2] in _CN_UNITS:
        base = _cn_numeral_to_int(text[:-1])
        if base is not None:
            return base + _CN_DIGITS[text[-1]] * (_CN_UNITS[text[-2]] / 10)
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        elif char == "万":
            section = (section + number) * 10000
            total += section
            section = 0
            number = 0
        else:
            return None
    return float(total + section + number)


def _amount_value(amount, scale):
    base = _cn_numeral_to_int(amount)
    if base is None:
        return None
    normalized_scale = str(scale or "").casefold()
    if normalized_scale == "万":
        base *= 10000
    elif normalized_scale in {"千", "k"}:
        base *= 1000
    return base


_RANGE_RE = re.compile(
    rf"({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?\s*"
    rf"(?:-|—|–|~|～|至|到)\s*"
    rf"({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?"
)
_UPPER_PREFIX_RE = re.compile(
    rf"{_UPPER_PREFIX}\s*(?:在)?\s*({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?"
)
_UPPER_SUFFIX_RE = re.compile(
    rf"({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?\s*{_UPPER_SUFFIX}"
)
_AROUND_RE = re.compile(
    rf"({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?\s*{_AROUND_SUFFIX}"
)
_AROUND_BEFORE_UNIT_RE = re.compile(
    rf"({_AMOUNT})\s*({_SCALE})?\s*(?:多|来)\s*{_PRICE_UNIT}"
)
_PER_UNIT_RE = re.compile(
    rf"{_PER_UNIT}\s*({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}"
)
_WAN_SHORTHAND_RE = re.compile(
    rf"({_AMOUNT})\s*万\s*({_AMOUNT})\s*(?:千)?\s*{_PRICE_UNIT}?"
)
_CUE_AMOUNT_RE = re.compile(
    rf"{_BUDGET_CUE}\s*(?:控制|大概|大约|约|差不多|接近|卡|是|为|在|要|有|花|需|需要|得)*\s*"
    rf"({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?"
)
_AMOUNT_CUE_RE = re.compile(
    rf"({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}\s*(?:的)?\s*"
    rf"{_BUDGET_CUE}"
)
_LOWER_RE = re.compile(
    rf"(?:{_LOWER_PREFIX}\s*(?:在)?\s*({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?"
    rf"|({_AMOUNT})\s*({_SCALE})?\s*{_PRICE_UNIT}?\s*{_LOWER_SUFFIX})"
)
_CUE_NEAR_RE = re.compile(_BUDGET_CUE)
_PRICE_UNIT_RE = re.compile(_PRICE_UNIT)
_OPEN_ENDED_RE = re.compile(rf"{_AMOUNT}\s*{_SCALE}?\s*\+")


def _price_context(text, start, end, before=8):
    """True when the span carries a price unit or sits right after a cue.

    Bare numbers such as ``6g``, ``60厘米`` or ``SPF50+`` are not budgets, so a
    candidate only counts when it is either unit-tagged or cue-led.
    """
    if _PRICE_UNIT_RE.search(text[start:end]):
        return True
    return bool(_CUE_NEAR_RE.search(text[max(0, start - before): start]))


def explicit_budget_from_instruction(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous.

    v4 extends v3's coverage without relaxing the hard gate:

    * Chinese numerals (``七十元`` / ``一千二百元`` / ``两千五``);
    * colloquial units (``块`` / ``块钱``);
    * prefix-less ranges (``350 到 400 元之间``);
    * per-unit prices (``每平方米一千元``);
    * non-ASCII whitespace (``\\xa0`` / full-width space).

    Ambiguous or lower-bound-only phrasings still return ``None``.
    """
    text = _normalize_price_text(instruction)
    if not text:
        return None

    # Open-ended *lower* bounds such as "4k+" describe a floor, not a ceiling.
    for match in _OPEN_ENDED_RE.finditer(text):
        if _price_context(text, match.start(), match.end()):
            return None

    # Shorthand "1万2千" = 10000 + 2000.
    for match in _WAN_SHORTHAND_RE.finditer(text):
        if not _price_context(text, match.start(), match.end()):
            continue
        high = _amount_value(match.group(1), "万")
        low = _amount_value(match.group(2), "千")
        if high is not None and low is not None and high + low > 0:
            return high + low

    # Ranges are explicit upper bounds; parse them before single values so that
    # "10-20" yields 20 and is never mistaken for an upper bound of 10.
    for match in _RANGE_RE.finditer(text):
        if not _price_context(text, match.start(), match.end()):
            continue
        low = _amount_value(match.group(1), match.group(2))
        high = _amount_value(match.group(3), match.group(4))
        if low is not None and high is not None and high >= low > 0:
            return high

    # "左右 / 上下 / 出头" are soft preferences; keep v3's deterministic +10%.
    # Checked before the strict markers so "价格控制在 450 元左右" keeps the
    # frozen v3 tolerance instead of collapsing to the round number.
    for pattern in (_AROUND_RE, _AROUND_BEFORE_UNIT_RE):
        for match in pattern.finditer(text):
            if not _price_context(text, match.start(), match.end()):
                continue
            value = _amount_value(match.group(1), match.group(2))
            if value is not None and value > 0:
                return value * 1.1

    # Explicit upper-bound markers ("不超过 70 块", "400 元以内").
    for pattern in (_UPPER_PREFIX_RE, _UPPER_SUFFIX_RE):
        for match in pattern.finditer(text):
            if not _price_context(text, match.start(), match.end()):
                continue
            value = _amount_value(match.group(1), match.group(2))
            if value is not None and value > 0:
                return value

    # Per-unit prices ("每平方米一千元") bound the unit price.
    for match in _PER_UNIT_RE.finditer(text):
        value = _amount_value(match.group(1), match.group(2))
        if value is not None and value > 0:
            return value

    # Budget cue with an adjacent amount ("预算是四百七十元", "预算约 75 元").
    for pattern in (_CUE_AMOUNT_RE, _AMOUNT_CUE_RE):
        for match in pattern.finditer(text):
            value = _amount_value(match.group(1), match.group(2))
            if value is not None and value > 0:
                return value

    # A lone lower-bound phrasing with no ceiling is not a budget.
    if _LOWER_RE.search(text):
        return None

    return None


def _price_range_above(price):
    if price <= 100:
        step = 3
    elif price <= 1000:
        step = 10
    elif price <= 5000:
        step = 50
    elif price <= 10000:
        step = 100
    else:
        step = 4
    base = math.ceil(price / 10) * 10
    return [base + index * 10 for index in range(step)]


def deterministic_price_upper(asin, instruction, price):
    explicit = explicit_budget_from_instruction(instruction)
    if explicit is not None:
        return explicit
    price_range = _price_range_above(float(price))
    if len(price_range) < 2:
        return 10000000
    digest = hashlib.sha256(
        f"{asin}\0{instruction}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    _, upper = sorted(rng.sample(price_range, 2))
    return upper
