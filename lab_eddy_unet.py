#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lab_eddy_unet.py -- 涡旋检测：U-Net 逐像素三分类

与 ``lab_eddy_rf.py`` 共用同一份数据缓存、同一套规则标签、同一个后处理
（``detections``）、同一套匹配（``match_detections``）与指标（``prf``），
因此 ``unet_results.json`` 与 ``rf_results.json`` 的 schema 完全一致，
可以直接喂给 ``compare_models.py``。

用法::

    python lab_eddy_unet.py                     # 96x96 切片, 60 epoch
    python lab_eddy_unet.py --epochs 5          # 先跑通流水线（方案 §8 要求）
    python lab_eddy_unet.py --no-skip           # 消融：去掉 skip connection
    python lab_eddy_unet.py --channels sst_only # 消融：只给 SST，去掉距平通道
    python lab_eddy_unet.py --loss ce           # 消融：换纯 CE

依赖：numpy / scipy / scikit-learn（沿用基线） / torch。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_eddy_rf import (  # noqa: E402  -- 共享基线模块
    CACHE_NAME,
    CLASS_NAMES,
    SEED,
    TRAIN_END_DAY,
    EddyData,
    anomaly_background,
    build_meta,
    class_counts_of,
    class_weights_from_counts,
    compute_anomaly,
    evaluate,
    load_data,
    rule_labels,
    valid_mask,
)

UNET_RESULTS = "unet_results.json"
UNET_WEIGHTS = "unet_weights.pt"


# ----------------------------------------------------------------------------
# 1. 切片
# ----------------------------------------------------------------------------


def padded_size(size: int, patch: int, stride: int) -> int:
    """把 ``size`` 补到能覆盖所有切片起点所需的最小尺寸。"""
    if size <= patch:
        return patch
    return patch + int(math.ceil((size - patch) / stride)) * stride


def patch_starts(size: int, patch: int, stride: int) -> List[int]:
    """切片起点列表（含末端对齐的那一片）。"""
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if not starts or starts[-1] != size - patch:
        starts.append(max(size - patch, 0))
    return sorted(set(int(s) for s in starts))


def pad_field(field: np.ndarray, patch: int, stride: int, mode: str = "reflect") -> np.ndarray:
    """反射填充到切片网格尺寸（方案 §2.3：边角用反射填充）。"""
    h, w = field.shape[-2:]
    nh, nw = padded_size(h, patch, stride), padded_size(w, patch, stride)
    ph, pw = nh - h, nw - w
    if ph == 0 and pw == 0:
        return field
    # np.pad 的 reflect 要求填充宽度 < 边长，否则退化为 symmetric
    m = mode
    if ph >= h or pw >= w:
        m = "symmetric"
    pad = [(0, 0)] * (field.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(field, pad, mode=m)


# ----------------------------------------------------------------------------
# 2. 输入通道
# ----------------------------------------------------------------------------


class Normalizer:
    """通道标准化：只用训练集统计量（方案 §2.4）。"""

    def __init__(self, mu: np.ndarray, sd: np.ndarray):
        self.mu = mu.astype(np.float32)
        self.sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    @classmethod
    def fit(cls, sst: np.ndarray, anom: np.ndarray, idx: Sequence[int]) -> "Normalizer":
        ok = np.concatenate([valid_mask(sst[i]).ravel() for i in idx])
        s = np.concatenate([sst[i].ravel() for i in idx])[ok]
        a = np.concatenate([anom[i].ravel() for i in idx])[ok]
        mu = np.array([np.nanmean(s), np.nanmean(a)], dtype=np.float32)
        sd = np.array([np.nanstd(s), np.nanstd(a)], dtype=np.float32)
        return cls(mu, sd)

    def __call__(self, sst: np.ndarray, anom: np.ndarray, use_anom: bool = True) -> np.ndarray:
        """``(2|3, H, W)`` float32，缺测像素置 0 并由掩膜通道标记。"""
        if use_anom:
            ch = np.stack(
                [
                    (sst - self.mu[0]) / self.sd[0],
                    (anom - self.mu[1]) / self.sd[1],
                    valid_mask(sst).astype(np.float32),
                ]
            )
        else:
            ch = np.stack([(sst - self.mu[0]) / self.sd[0], valid_mask(sst).astype(np.float32)])
        return np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def state(self) -> Dict[str, List[float]]:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_state(cls, st: Dict[str, List[float]]) -> "Normalizer":
        return cls(np.array(st["mu"], dtype=np.float32), np.array(st["sd"], dtype=np.float32))


def make_patches(img: np.ndarray, patch: int, stride: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """``(C,H,W)`` -> ``(N,C,patch,patch)`` 与起点列表。"""
    img = pad_field(img, patch, stride)
    h, w = img.shape[-2:]
    starts = [(i, j) for i in patch_starts(h, patch, stride) for j in patch_starts(w, patch, stride)]
    out = np.stack([img[:, i : i + patch, j : j + patch] for i, j in starts])
    return out.astype(np.float32), starts


# ----------------------------------------------------------------------------
# 3. 网络
# ----------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x2，U-Net 的基本块。"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """4 级下采样的 U-Net，解码器与编码器同级特征 concat。

    Args:
        in_ch: 输入通道数（3 = SST / 距平 / 有效掩膜）。
        n_cls: 输出类别数（3）。
        base: 第一级通道数，逐级翻倍。
        depth: 下采样级数（方案要求 4）。
        skip: 是否使用 skip connection（消融开关）。
        bottleneck: 瓶颈通道数，缺省 ``base * 2**depth``。
        dropout: 瓶颈处 dropout，0 关闭。

    Note:
        方案 §3.1 同时要求通道 ``[32, 64, 128, 256]`` + 瓶颈 512 且参数量 < 2M，
        这两个条件在「每级两个 3x3 卷积」下无法同时满足（该配置约 7.8M）。
        这里默认 ``base=16``（``[16,32,64,128]`` + 瓶颈 256，约 1.9M）以满足
        CPU 可训的上限；``--base 32`` 可复现方案原表（约 7.8M）。
    """

    def __init__(
        self,
        in_ch: int = 3,
        n_cls: int = 3,
        base: int = 16,
        depth: int = 4,
        skip: bool = True,
        bottleneck: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.skip = skip
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth)]
        bott = bottleneck if bottleneck is not None else chans[-1] * 2

        self.enc = nn.ModuleList()
        cin = in_ch
        for c in chans:
            self.enc.append(DoubleConv(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(DoubleConv(cin, bott))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = bott
        for c in reversed(chans):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(DoubleConv(c * (2 if skip else 1), c))
            cin = c

        self.head = nn.Conv2d(cin, n_cls, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for block in self.enc:
            x = block(x)
            feats.append(x)
            x = self.pool(x)
        x = self.dropout(self.bottleneck(x))
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            f = feats[-1 - i]
            # 尺寸兜底（奇数尺寸时最大池化会让 skip 和上采样差 1 像素）
            if x.shape[-2:] != f.shape[-2:]:
                x = F.interpolate(x, size=f.shape[-2:], mode="bilinear", align_corners=False)
            if self.skip:
                x = torch.cat([x, f], dim=1)
            x = dec(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# 4. 损失：0.5*(1-Dice) + CE(weighted)
# ----------------------------------------------------------------------------


class DiceCELoss(nn.Module):
    """方案 §5 的混合损失。

    ``L = dice_weight * (1 - Dice_macro) + ce_weight * CE(weighted)``

    - Dice 按类计算，只有出现在 batch 里的类参与平均（空类不拉低 loss）；
    - CE 用类权重，``ignore_index=-1`` 让缺测像素权重为 0。
    """

    def __init__(self, class_weights: np.ndarray | None = None, dice_weight: float = 0.5,
                 ce_weight: float = 1.0, mode: str = "dice_ce", eps: float = 1e-6):
        super().__init__()
        self.dice_weight = 0.0 if mode == "ce" else dice_weight
        self.ce_weight = ce_weight if mode in ("dice_ce", "ce") else 0.0
        self.mode = mode
        self.eps = eps
        w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", w if w is not None else torch.ones(3))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``logits (B,C,H,W)``，``target (B,H,W)``（-1 = 缺测）。"""
        ce = logits.new_zeros(())
        if self.ce_weight > 0:
            ce = F.cross_entropy(logits, target, weight=self.class_weights,
                                 ignore_index=-1, reduction="mean")

        dice = logits.new_zeros(())
        if self.dice_weight > 0:
            prob = F.softmax(logits, dim=1)
            valid = (target >= 0).unsqueeze(1).to(prob.dtype)
            vals = []
            for c in range(prob.shape[1]):
                m = ((target == c) & (target >= 0)).unsqueeze(1).to(prob.dtype)
                if float(m.sum()) == 0:
                    continue  # 空类不参与
                p = prob[:, c : c + 1] * valid
                inter = float((p * m).sum())
                denom = float(p.sum()) + float(m.sum())
                vals.append(1.0 - (2 * inter + self.eps) / (denom + self.eps))
            if vals:
                dice = torch.stack([torch.as_tensor(v, dtype=prob.dtype, device=prob.device) for v in vals]).mean()

        total = self.dice_weight * dice + self.ce_weight * ce
        return total, {"loss": float(total.detach()), "dice": float(dice.detach()), "ce": float(ce.detach())}


# ----------------------------------------------------------------------------
# 5. 数据装配
# ----------------------------------------------------------------------------


def build_train_arrays(
    data: EddyData,
    anom: np.ndarray,
    norm: Normalizer,
    idx: Sequence[int],
    patch: int,
    stride: int,
    use_anom: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """把训练日切成切片并打乱，返回 ``(X (N,C,p,p), y (N,p,p))``。"""
    xs, ys = [], []
    labels = rule_labels(anom)
    for i in idx:
        img = norm(data.sst[i], anom[i], use_anom=use_anom)
        xp, _ = make_patches(img, patch, stride)
        yp, _ = make_patches(labels[i].astype(np.float32)[None], patch, stride)
        xs.append(xp)
        ys.append(yp[:, 0].astype(np.int64))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


@torch.no_grad()
def predict_day(
    model: nn.Module,
    img: np.ndarray,
    patch: int,
    stride: int,
    device: torch.device,
    batch: int = 16,
) -> np.ndarray:
    """整幅拼接推理：重叠区按 logits 平均（soft voting）。"""
    xp, starts = make_patches(img, patch, stride)
    h, w = pad_field(img, patch, stride).shape[-2:]
    n_cls = model.head.out_channels
    acc = np.zeros((n_cls, h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for s in range(0, len(xp), batch):
        chunk = torch.from_numpy(xp[s : s + batch]).to(device)
        out = F.softmax(model(chunk), dim=1).cpu().numpy()
        for k, (i, j) in enumerate(starts[s : s + batch]):
            acc[:, i : i + patch, j : j + patch] += out[k]
            cnt[i : i + patch, j : j + patch] += 1.0
    cnt = np.maximum(cnt, 1.0)
    prob = acc / cnt[None]
    pred = np.argmax(prob, axis=0).astype(np.int64)
    return pred[: img.shape[-2], : img.shape[-1]]


def train(
    X: np.ndarray,
    y: np.ndarray,
    model: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    class_weights: np.ndarray,
    device: torch.device,
    mode: str = "dice_ce",
    seed: int = SEED,
    log_every: int = 1,
    val_fn=None,
) -> Dict[str, List[float]]:
    """训练循环，返回逐 epoch 的 loss 曲线。"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = DiceCELoss(class_weights, mode=mode).to(device)

    n = X.shape[0]
    hist = {"loss": [], "dice": [], "ce": [], "lr": [], "val_iou": []}
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        tot = {"loss": 0.0, "dice": 0.0, "ce": 0.0}
        nb = 0
        t0 = time.perf_counter()
        for s in range(0, n, batch_size):
            sel = order[s : s + batch_size]
            xb = torch.from_numpy(X[sel]).to(device)
            yb = torch.from_numpy(y[sel]).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss, parts = crit(logits, yb)
            loss.backward()
            opt.step()
            for k in tot:
                tot[k] += parts[k]
            nb += 1
        mean_loss = tot["loss"] / max(nb, 1)
        sched.step(mean_loss)  # 只用训练 loss 调度：测试日不参与任何决策
        hist["loss"].append(mean_loss)
        hist["dice"].append(tot["dice"] / max(nb, 1))
        hist["ce"].append(tot["ce"] / max(nb, 1))
        hist["lr"].append(opt.param_groups[0]["lr"])
        viou = float(val_fn(model)) if val_fn is not None else float("nan")
        hist["val_iou"].append(viou)
        if ep % log_every == 0 or ep == epochs:
            print(
                f"  epoch {ep:3d}/{epochs}  loss={mean_loss:.4f}  "
                f"dice={hist['dice'][-1]:.4f}  ce={hist['ce'][-1]:.4f}  "
                f"lr={hist['lr'][-1]:.2e}  val_IoU={viou:.4f}  "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return hist


# ----------------------------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="涡旋检测：U-Net 逐像素三分类")
    p.add_argument("--cache", default=CACHE_NAME)
    p.add_argument("--out", default=UNET_RESULTS)
    p.add_argument("--weights", default=UNET_WEIGHTS)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--base", type=int, default=16, help="首级通道数（16 -> 1.9M，32 -> 7.8M）")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--loss", choices=["dice_ce", "ce"], default="dice_ce")
    p.add_argument("--channels", choices=["sst_anom", "sst_only"], default="sst_anom")
    p.add_argument("--no-skip", action="store_true", help="消融：去掉 skip connection")
    p.add_argument("--class-weights", default="auto",
                   help="auto 或 '1.0,3.0,1.5'（background,warm,cold）")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch CPU 线程数，0 = 默认")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--no-save", action="store_true", help="不写权重，只打印指标")
    args = p.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    # --- 数据 ---
    data = load_data(args.cache, force_download=args.force_download, no_download=args.no_download)
    n = len(data.dates)
    train_idx = list(range(0, min(TRAIN_END_DAY, n)))
    test_idx = list(range(min(TRAIN_END_DAY, n), n))
    print(f"[data] {data.shape}  训练 {len(train_idx)} 天 / 测试 {len(test_idx)} 天")

    bg = anomaly_background(data.sst, train_idx)
    anom = compute_anomaly(data.sst, bg)
    labels = rule_labels(anom)
    counts = class_counts_of(labels[train_idx])
    total = sum(counts.values())
    print("[labels] train pixels: " + ", ".join(
        f"{k}={v} ({v / max(total, 1) * 100:.1f}%)" for k, v in counts.items()))

    if args.class_weights == "auto":
        cw = class_weights_from_counts(counts)
    else:
        cw = np.array([float(x) for x in args.class_weights.split(",")], dtype=np.float32)
    print(f"[loss] class weights = {cw.round(3).tolist()}")

    norm = Normalizer.fit(data.sst, anom, train_idx)
    use_anom = args.channels == "sst_anom"
    in_ch = 3 if use_anom else 2

    X, y = build_train_arrays(data, anom, norm, train_idx, args.patch, args.stride, use_anom)
    print(f"[patches] patch={args.patch} stride={args.stride} -> train patches {X.shape[0]}")

    # --- 模型 ---
    model = UNet(in_ch=in_ch, n_cls=3, base=args.base, depth=args.depth,
                 skip=not args.no_skip, dropout=args.dropout).to(device)
    n_param = count_parameters(model)
    print(f"[model] U-Net base={args.base} depth={args.depth} skip={not args.no_skip} "
          f"params={n_param:,} ({n_param / 1e6:.2f}M)")

    # 监控用：只打印，不参与任何选择（避免测试日泄漏）
    def val_iou(m: nn.Module) -> float:
        m.eval()
        preds = []
        for i in test_idx:
            img = norm(data.sst[i], anom[i], use_anom=use_anom)
            preds.append(predict_day(m, img, args.patch, args.stride, device))
        preds = np.stack(preds)
        refs = labels[test_idx]
        ok = refs >= 0
        ious = []
        for c in (1, 2):
            pm, rm = (preds[ok] == c), (refs[ok] == c)
            inter = int(np.logical_and(pm, rm).sum())
            union = int(np.logical_or(pm, rm).sum())
            ious.append(inter / union if union else 0.0)
        m.train()
        return float(np.mean(ious))

    # --- 训练 ---
    t0 = time.perf_counter()
    hist = train(X, y, model, args.epochs, args.batch, args.lr, cw, device,
                 mode=args.loss, seed=args.seed, val_fn=val_iou)
    train_time = time.perf_counter() - t0
    print(f"[train] {train_time:.1f}s ({train_time / 60:.1f} min)")

    if not args.no_save:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "in_ch": in_ch, "n_cls": 3, "base": args.base, "depth": args.depth,
                    "skip": not args.no_skip, "dropout": args.dropout,
                    "patch": args.patch, "stride": args.stride, "channels": args.channels,
                },
                "normalizer": norm.state(),
                "background": bg,
                "history": hist,
            },
            args.weights,
        )
        print(f"[save] {args.weights}")

    # --- 推理 ---
    model.eval()
    preds = np.empty((len(test_idx), *data.sst.shape[1:]), dtype=np.int64)
    t_inf = time.perf_counter()
    for k, i in enumerate(test_idx):
        img = norm(data.sst[i], anom[i], use_anom=use_anom)
        preds[k] = predict_day(model, img, args.patch, args.stride, device)
    infer_time = time.perf_counter() - t_inf

    t_rule = time.perf_counter()
    refs = labels[test_idx]
    rule_time = time.perf_counter() - t_rule

    # --- 评估（与基线同一套函数）---
    result = build_meta(data, train_idx, test_idx, counts, train_time)
    result.update(
        evaluate(
            preds,
            refs,
            anom[test_idx],
            data.lat,
            data.lon,
            [data.dates[i] for i in test_idx],
            infer_time,
            rule_time,
        )
    )
    result["model"] = {
        "name": "unet",
        "params": int(n_param),
        "base_channels": args.base,
        "depth": args.depth,
        "skip": not args.no_skip,
        "channels": args.channels,
        "loss": args.loss,
        "class_weights": cw.tolist(),
        "patch": args.patch,
        "stride": args.stride,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "train_patches": int(X.shape[0]),
        "history": {k: [float(v) for v in vals] for k, vals in hist.items()},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}")
    print(f"       warm IoU={result['pixel']['warm']['iou']:.4f}  cold IoU={result['pixel']['cold']['iou']:.4f}")
    print(f"       detection macro F1={result['detection']['macro_f1']:.4f}")
    print(f"       infer {infer_time / max(len(test_idx), 1) * 1000:.0f} ms/day")
    return 0


if __name__ == "__main__":
    sys.exit(main())