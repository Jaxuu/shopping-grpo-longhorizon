import unittest

from scripts.label_sft_difficulty_offline import (
    HEURISTIC_MODE,
    label_row,
    price_band,
    trajectory_features,
)


def _messages(*, searches, opens, steps_extra=0):
    messages = []
    for index in range(searches):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"s{index}",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": f'{{"query": "q{index}"}}',
                        },
                    }
                ],
            }
        )
    for index in range(opens):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"o{index}",
                        "type": "function",
                        "function": {"name": "open_product", "arguments": f'{{"asin": "a{index}"}}'},
                    }
                ],
            }
        )
    for index in range(steps_extra):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"x{index}",
                        "type": "function",
                        "function": {"name": "prev_page", "arguments": "{}"},
                    }
                ],
            }
        )
    return messages


class OfflineLabelTests(unittest.TestCase):
    def test_trajectory_features_counts_distinct_queries(self):
        features = trajectory_features(_messages(searches=3, opens=2))
        self.assertEqual(features["searches"], 3)
        self.assertEqual(features["distinct_queries"], 3)
        self.assertEqual(features["opens"], 2)
        self.assertEqual(features["steps"], 5)

    def test_price_band_thresholds(self):
        self.assertEqual(price_band(None), "unknown")
        self.assertEqual(price_band(20.0), "low")
        self.assertEqual(price_band(100.0), "medium")
        self.assertEqual(price_band(900.0), "high")

    def test_high_ambiguity_multi_candidate_is_hard(self):
        features = trajectory_features(_messages(searches=2, opens=3))
        label = label_row(
            task_id=1, features=features, ambiguity=120, axes=1, price_upper=100.0, domain="浴霸"
        )
        self.assertEqual(label["difficulty"], "hard")
        self.assertTrue(label["needs_candidate_comparison"])
        self.assertEqual(label["label_input_mode"], HEURISTIC_MODE)
        self.assertEqual(label["confidence"], 0.4)

    def test_low_ambiguity_direct_purchase_is_simple(self):
        features = trajectory_features(_messages(searches=1, opens=1))
        label = label_row(
            task_id=2, features=features, ambiguity=None, axes=0, price_upper=None, domain="水彩笔"
        )
        self.assertEqual(label["difficulty"], "simple")
        self.assertFalse(label["needs_query_rewrite"])
        self.assertEqual(label["price_band"], "unknown")

    def test_long_trajectory_escalates_difficulty(self):
        features = trajectory_features(_messages(searches=1, opens=1, steps_extra=16))
        label = label_row(
            task_id=3, features=features, ambiguity=None, axes=0, price_upper=50.0, domain="灯"
        )
        self.assertEqual(label["difficulty"], "hard")
        self.assertEqual(label["trajectory_complexity"], "hard")


if __name__ == "__main__":
    unittest.main()
