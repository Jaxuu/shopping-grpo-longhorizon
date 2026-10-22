import unittest

from scripts.estimate_sft_train_time import (
    CALIBRATION,
    EFFECTIVE_BATCH,
    fit_linear,
    fit_power,
    predict,
)

M5_POINTS = CALIBRATION[:3]  # 只用 M5 三点做「拟合器本身是否正确」的检查
MEASURED = CALIBRATION[3]  # M10-R stage A 实测（2026-10-23 00:43）


class EstimateTrainTimeTest(unittest.TestCase):
    def test_linear_fit_reproduces_its_own_input_points(self):
        """三点拟合必须能把这三点本身复现（误差 < 3%），否则拟合器就是错的。"""
        linear = fit_linear(M5_POINTS)
        for chars, seconds in M5_POINTS:
            predicted = linear[0] + linear[1] * chars
            self.assertLess(abs(predicted - seconds) / seconds, 0.03)

    def test_full_fit_predicts_the_measured_stage_a_point(self):
        """加入实测点后，拟合在 18,726 字符处要落在 119.6 s 附近（< 5%）。

        这是这个脚本唯一有意义的验收：它必须能解释真实观测。
        四个点一起拟合时最大误差在 stage C（7.2%），所以不对全部点卡 3%。
        """
        linear = fit_linear(CALIBRATION)
        predicted = linear[0] + linear[1] * MEASURED[0]
        self.assertLess(abs(predicted - MEASURED[1]) / MEASURED[1], 0.05)

    def test_measured_scaling_is_sub_linear(self):
        """实测的 s/step 增长慢于字符数增长（指数 < 1）。

        M5 的 A→C 两点看起来像超线性，但加入 M10-R 的长轨迹点后整体是**次线性**的。
        所以幂律不能当成「attention 是 O(L²)，所以会更慢」的依据去外推到数据范围之外——
        它只是三条参考线里的一条。
        """
        _, exponent = fit_power(CALIBRATION)
        self.assertLess(exponent, 1.0)
        self.assertGreater(exponent, 0.5)

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
