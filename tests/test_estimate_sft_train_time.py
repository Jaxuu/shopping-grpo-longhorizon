import unittest

from scripts.estimate_sft_train_time import (
    CALIBRATION,
    EFFECTIVE_BATCH,
    fit_linear,
    fit_power,
    predict,
)


class EstimateTrainTimeTest(unittest.TestCase):
    def test_fits_reproduce_the_calibration_points(self):
        """标定点是实测值，拟合必须能把它复现（误差 < 3%）。"""
        linear = fit_linear(CALIBRATION)
        for chars, seconds in CALIBRATION:
            predicted = linear[0] + linear[1] * chars
            self.assertLess(abs(predicted - seconds) / seconds, 0.03)

    def test_power_fit_beats_one_reflecting_attention_cost(self):
        """指数 >1 才符合 attention 的 O(L²)；等于 1 说明退化成线性。"""
        _, exponent = fit_power(CALIBRATION)
        self.assertGreater(exponent, 1.05)

    def test_longer_rows_never_estimate_faster(self):
        linear = fit_linear(CALIBRATION)
        power = fit_power(CALIBRATION)
        short = predict(7000, linear, power)
        long = predict(20000, linear, power)
        for key in short:
            self.assertGreater(long[key], short[key])

    def test_effective_batch_matches_the_training_recipe(self):
        """per_device_train_batch_size=1 × gradient_accumulation_steps=8。"""
        self.assertEqual(EFFECTIVE_BATCH, 8)


if __name__ == "__main__":
    unittest.main()
