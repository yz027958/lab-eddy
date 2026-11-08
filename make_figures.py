#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_figures.py -- 生成报告用的图

产出（默认写到 ``figures/``）：

- ``pred_vs_rule_<date>.png``：同一测试日的 SST / 规则法标签 / U-Net 标签 三联图；
- ``training_curve.png``：训练 loss 与测试期 IoU 曲线（只作展示，不参与选择）；
- ``confusion.png``：随机森林与 U-Net 的混淆矩阵（行=参考，列=预测）。

用法::

    python make_figures.py
    python make_figures.py --date 2024-01-25 --outdir figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Windows 自带中文字体，避免图上出现方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

#: 0 背景 / 1 暖涡 / 2 冷涡 / 缺测
LABEL_CMAP = ListedColormap(["#ffffff", "#d73027", "#4575b4", "#e0e0e0"])
LABEL_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], LABEL_CMAP.N)

from lab_eddy_rf import (  # noqa: E402
    CACHE_NAME,
    CLASS_NAMES,
    TRAIN_END_DAY,
    EddyData,
    anomaly_background,
    compute_anomaly,
    rule_labels,
)
from lab_eddy_unet import UNET_RESULTS, UNET_WEIGHTS, Normalizer, UNet, predict_day  # noqa: E402


def load_predictor(weights: str, device: torch.device):
    """从 ``unet_weights.pt`` 复原模型、标准化器与背景场。"""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = UNet(in_ch=cfg["in_ch"], n_cls=cfg["n_cls"], base=cfg["base"],
                 depth=cfg["depth"], skip=cfg["skip"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    norm = Normalizer.from_state(ckpt["normalizer"])
    bg = np.asarray(ckpt["background"], dtype=np.float32)
    return model, norm, bg, cfg


def plot_prediction(data: EddyData, anom: np.ndarray, pred: np.ndarray, labels: np.ndarray,
                    day: int, out: str) -> None:
    """三联图：SST / 规则标签 / U-Net 标签。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ext = [float(data.lon[0]), float(data.lon[-1]), float(data.lat[0]), float(data.lat[-1])]

    im0 = axes[0].imshow(data.sst[day], origin="lower", extent=ext, cmap="RdYlBu_r")
    axes[0].set_title(f"SST  {data.dates[day]}")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="degC")
    axes[0].set_ylabel("lat")

    for ax, m, title in ((axes[1], labels[day], "规则法标签"), (axes[2], pred, "U-Net 预测")):
        ax.imshow(np.where(m < 0, 3, m), origin="lower", extent=ext,
                  cmap=LABEL_CMAP, norm=LABEL_NORM, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("lon")
    axes[0].legend(handles=[
        Patch(color="#d73027", label="暖涡"),
        Patch(color="#4575b4", label="冷涡"),
        Patch(color="#ffffff", ec="0.6", label="背景"),
        Patch(color="#e0e0e0", label="缺测"),
    ], loc="lower right", fontsize=7, frameon=True, framealpha=0.85)

    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_training_curve(hist: dict, out: str) -> None:
    """loss / dice / IoU 三个子图。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    ep = np.arange(1, len(hist.get("loss", [])) + 1)
    axes[0].plot(ep, hist.get("loss", []), label="total")
    axes[0].plot(ep, hist.get("dice", []), label="dice term")
    axes[0].plot(ep, hist.get("ce", []), label="CE term")
    axes[0].set_title("training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, hist.get("lr", []))
    axes[1].set_title("learning rate")
    axes[1].set_xlabel("epoch")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ep, hist.get("val_iou", []), marker="o", ms=2)
    axes[2].set_title("test-period macro IoU (monitor only)")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(rf: dict, unet: dict, out: str) -> None:
    """并排两张混淆矩阵（行 = 参考）。"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, obj, title in ((axes[0], rf, "Random Forest"), (axes[1], unet, "U-Net")):
        cm = np.asarray(obj["confusion_matrix"], dtype=float)
        row = cm.sum(axis=1, keepdims=True)
        cmn = cm / np.maximum(row, 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), CLASS_NAMES, rotation=30, fontsize=8)
        ax.set_yticks(range(3), CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("reference")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=7, color="black" if cmn[i, j] < 0.55 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成报告用图")
    p.add_argument("--cache", default=CACHE_NAME)
    p.add_argument("--weights", default=UNET_WEIGHTS)
    p.add_argument("--rf", default="rf_results.json")
    p.add_argument("--unet", default=UNET_RESULTS)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--date", default="", help="要画哪一天（缺省取最后一个测试日）")
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.exists(args.weights):
        print(f"[skip] 没有 {args.weights}，先跑 lab_eddy_unet.py", file=sys.stderr)
        return 1

    data = EddyData.load(args.cache)
    train_idx = list(range(0, min(TRAIN_END_DAY, len(data.dates))))
    bg = anomaly_background(data.sst, train_idx)
    anom = compute_anomaly(data.sst, bg)
    labels = rule_labels(anom)

    device = torch.device(args.device)
    model, norm, _bg, cfg = load_predictor(args.weights, device)
    use_anom = cfg.get("channels", "sst_anom") == "sst_anom"
    patch = cfg.get("patch", args.patch)
    stride = cfg.get("stride", args.stride)

    test_idx: List[int] = list(range(min(TRAIN_END_DAY, len(data.dates)), len(data.dates)))
    if args.date:
        if args.date not in data.dates:
            print(f"[err] {args.date} 不在数据里", file=sys.stderr)
            return 1
        day = data.dates.index(args.date)
    else:
        day = test_idx[-1]

    img = norm(data.sst[day], anom[day], use_anom=use_anom)
    pred = predict_day(model, img, patch, stride, device)
    out = os.path.join(args.outdir, f"pred_vs_rule_{data.dates[day]}.png")
    plot_prediction(data, anom, pred, labels, day, out)
    print(f"[fig] {out}")

    with open(args.unet, encoding="utf-8") as f:
        unet = json.load(f)
    hist = unet.get("model", {}).get("history")
    if hist:
        out = os.path.join(args.outdir, "training_curve.png")
        plot_training_curve(hist, out)
        print(f"[fig] {out}")

    if os.path.exists(args.rf):
        with open(args.rf, encoding="utf-8") as f:
            rf = json.load(f)
        out = os.path.join(args.outdir, "confusion.png")
        plot_confusion(rf, unet, out)
        print(f"[fig] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())