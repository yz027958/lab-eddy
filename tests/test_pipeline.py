#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端冒烟测试：不需要网络，用合成 SST 跑通「标签 -> 训练 -> 评估 -> 对比」。

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lab_eddy_rf as base  # noqa: E402
import lab_eddy_unet as unet  # noqa: E402


def make_synthetic(path: str, days: int = 23, h: int = 24, w: int = 24) -> str:
    """合成一个「有暖涡/冷涡/缺测」的小数据集并写缓存。"""
    rng = np.random.default_rng(0)
    lat = np.linspace(20.0, 20.0 + 0.25 * (h - 1), h, dtype=np.float32)
    lon = np.linspace(140.0, 140.0 + 0.25 * (w - 1), w, dtype=np.float32)
    sst = 20.0 + 0.3 * rng.standard_normal((days, h, w)).astype(np.float32)
    # 涡旋振幅随时间振荡：这样「减时间平均」不会把涡旋本身扣掉，
    # 两个时期都能拿到真实的暖/冷标签
    amp = (1.6 * np.cos(2 * np.pi * np.arange(days) / 7.0)).astype(np.float32)
    sst[:, 4:9, 4:9] += amp[:, None, None]    # 暖涡块（面积 25 >= MIN_AREA）
    sst[:, 14:19, 14:19] -= amp[:, None, None]  # 冷涡块
    sst[:, 0:2, 0:2] = np.nan    # 缺测
    dates = [f"2024-01-{d:02d}" for d in range(1, days + 1)]
    data = base.EddyData(sst=sst, lat=lat, lon=lon, dates=dates)
    data.save(path)
    return path


class TestLabelsAndPostprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = make_synthetic(os.path.join(self.tmp.name, "sst_cache.npz"))
        self.data = base.EddyData.load(self.cache)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rule_labels(self) -> None:
        anom = np.array([[0.0, 0.5, -0.5, np.nan]], dtype=np.float32)
        lab = base.rule_labels(anom)
        self.assertEqual(lab.tolist(), [[0, 1, 2, -1]])

    def test_anomaly_has_no_leak(self) -> None:
        """背景场只由训练日决定。"""
        tr = list(range(0, 5))
        bg = base.anomaly_background(self.data.sst, tr)
        self.assertTrue(np.isfinite(bg).all())

    def test_detections_area_and_radius(self) -> None:
        mask = np.zeros((24, 24), dtype=np.int64)
        mask[4:9, 4:9] = 1          # 25 像素 -> 保留
        mask[20, 20] = 1            # 1 像素 -> 面积过小丢弃
        dets = base.detections(mask, self.data.lat, self.data.lon)
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d.area, 25)
        self.assertAlmostEqual(d.radius_km, np.sqrt(25 / np.pi) * base.CELL_KM, places=4)
        self.assertTrue(20.0 <= d.lat <= 26.0)
        self.assertTrue(139.0 <= d.lon <= 146.0)

    def test_detections_max_radius(self) -> None:
        mask = np.zeros((24, 24), dtype=np.int64)
        mask[:, :] = 1  # 576 px -> 半径 ~ 375 km > 250
        self.assertEqual(base.detections(mask, self.data.lat, self.data.lon), [])

    def test_match_is_one_to_one(self) -> None:
        a = [base.Eddy(1, 21.0, 141.0, 60.0, 20, 1.0)]
        b = [
            base.Eddy(1, 21.1, 141.1, 60.0, 20, 1.0),
            base.Eddy(1, 21.2, 141.2, 60.0, 20, 1.0),
        ]
        pairs, up, ur = base.match_detections(a, b)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(up), 0)
        self.assertEqual(len(ur), 1)

    def test_prf(self) -> None:
        m = base.prf(3, 1, 2)
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.6)
        self.assertAlmostEqual(m["f1"], 2 * 0.75 * 0.6 / 1.35)


class TestModel(unittest.TestCase):
    def test_param_budget(self) -> None:
        """方案 §3.1：参数量 <= 2M。"""
        m = unet.UNet(in_ch=3, n_cls=3, base=16, depth=4)
        n = unet.count_parameters(m)
        self.assertLess(n, 2_000_000, f"参数量 {n} 超出 2M 预算")

    def test_forward_shape(self) -> None:
        m = unet.UNet(in_ch=3, n_cls=3, base=8, depth=4)
        with __import__("torch").no_grad():
            out = m(__import__("torch").zeros(2, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (2, 3, 32, 32))

    def test_loss_finite_with_nan_mask(self) -> None:
        import torch

        crit = unet.DiceCELoss(np.array([1.0, 3.0, 1.5], dtype=np.float32))
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.randint(0, 3, (2, 8, 8))
        target[0, :4, :4] = -1      # 缺测
        loss, parts = crit(logits, target)
        self.assertTrue(np.isfinite(parts["loss"]))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in [logits]))


class TestEndToEnd(unittest.TestCase):
    """真的跑一遍 rf -> unet -> compare_models。"""

    def test_pipeline(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            cache = make_synthetic(os.path.join(tmp.name, "sst_cache.npz"))
            rf_json = os.path.join(tmp.name, "rf_results.json")
            unet_json = os.path.join(tmp.name, "unet_results.json")
            weights = os.path.join(tmp.name, "unet_weights.pt")

            rc = base.main(["--cache", cache, "--out", rf_json, "--no-download", "--trees", "20"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(rf_json))

            rc = unet.main([
                "--cache", cache, "--out", unet_json, "--weights", weights, "--no-download",
                "--epochs", "1", "--batch", "8", "--patch", "16", "--stride", "16",
                "--base", "8", "--dropout", "0.0", "--threads", "2",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(weights))

            with open(rf_json, encoding="utf-8") as f:
                rf = json.load(f)
            with open(unet_json, encoding="utf-8") as f:
                uu = json.load(f)
            for k in ("data", "split", "rule", "class_counts", "train_time_s", "pixel",
                      "confusion_matrix", "detection", "threshold_drift",
                      "daily_count_corr", "timing"):
                self.assertIn(k, rf)
                self.assertIn(k, uu)
            self.assertEqual(len(uu["confusion_matrix"]), 3)
            self.assertLess(uu["model"]["params"], 2_000_000)

            import compare_models

            table = compare_models.render(rf, uu)
            self.assertIn("U-Net", table)
            self.assertIn("一致度", table)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)