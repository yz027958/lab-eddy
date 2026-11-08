#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_models.py -- 只读两个 json，出一张对比表

设计约束（方案 §7）：对比逻辑**不写进任何一个模型脚本**。本脚本只做两件事：

1. 校验两份 json 的关键 schema 一致；
2. 把方案 §7.1 要求的对照项渲染成 markdown / csv 表格。

用法::

    python compare_models.py                          # 读 rf_results.json / unet_results.json
    python compare_models.py --out compare.md          # 同时写文件
    python compare_models.py --rf a.json --unet b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Sequence, Tuple

#: 两份 json 必须都有的顶层字段（方案 §7 的 schema）
REQUIRED_KEYS = (
    "data", "split", "rule", "class_counts", "train_time_s",
    "pixel", "confusion_matrix", "detection", "threshold_drift",
    "daily_count_corr", "timing",
)

#: 对比行：(展示名, 取值函数, 格式, 越大越好?)
ROWS: List[Tuple[str, str, str, bool]] = [
    ("像素级 warm IoU", "pixel/warm/iou", "{:.4f}", True),
    ("像素级 warm Dice", "pixel/warm/dice", "{:.4f}", True),
    ("像素级 cold IoU", "pixel/cold/iou", "{:.4f}", True),
    ("像素级 cold Dice", "pixel/cold/dice", "{:.4f}", True),
    ("检测级 warm P", "detection/warm/micro/precision", "{:.4f}", True),
    ("检测级 warm R", "detection/warm/micro/recall", "{:.4f}", True),
    ("检测级 warm F1", "detection/warm/micro/f1", "{:.4f}", True),
    ("检测级 cold P", "detection/cold/micro/precision", "{:.4f}", True),
    ("检测级 cold R", "detection/cold/micro/recall", "{:.4f}", True),
    ("检测级 cold F1", "detection/cold/micro/f1", "{:.4f}", True),
    ("检测级 macro F1", "detection/macro_f1", "{:.4f}", True),
    ("半径 MAE (km)", "detection/warm/radius_mae_km", "{:.2f}", False),
    ("半径 MAE (km, cold)", "detection/cold/radius_mae_km", "{:.2f}", False),
    ("振幅 MAE (degC)", "detection/warm/amp_mae", "{:.3f}", False),
    ("逐日计数相关 r", "daily_count_corr", "{:.4f}", True),
    ("阈值漂移 IoU @0.3", "threshold_drift/thresh_0.3", "{:.4f}", True),
    ("阈值漂移 IoU @0.5", "threshold_drift/thresh_0.5", "{:.4f}", True),
    ("单日推理 (ms)", "timing/unet_infer_per_day_s", "{:.1f}", True),
    ("训练耗时 (s)", "train_time_s", "{:.1f}", False),
]


def dig(obj: Any, path: str) -> Any:
    """按 ``a/b/c`` 取值，缺失返回 ``None``。

    用 ``/`` 而不是 ``.`` 分隔：结果 json 里的键名本身带点（``thresh_0.3``），
    按点切分会把键名切碎。
    """
    cur = obj
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_schema(name: str, obj: Dict[str, Any]) -> List[str]:
    """返回缺失字段列表（空表示通过）。"""
    return [k for k in REQUIRED_KEYS if k not in obj]


def fmt(v: Any, path: str, f: str) -> str:
    if v is None:
        return "—"
    try:
        if path.endswith("unet_infer_per_day_s"):
            return f.format(float(v) * 1000.0)   # 秒 -> 毫秒
        return f.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def mark(v: float | None, other: float | None, higher_better: bool) -> str:
    """两列之间的胜者标记。"""
    if v is None or other is None:
        return ""
    if abs(v - other) < 1e-9:
        return "="
    win = v > other if higher_better else v < other
    return "**胜**" if win else ""


def render(rf: Dict[str, Any], unet: Dict[str, Any]) -> str:
    lines: List[str] = []
    a, b = rf["data"], unet["data"]
    lines.append(f"# 模型对比：随机森林基线 vs U-Net\n")
    lines.append(f"- 数据：{a['start']} ~ {a['end']}，网格 {a['grid']}，"
                 f"经纬度 {a['lat_range']} / {a['lon_range']}")
    lines.append(f"- 切分：训练 {len(rf['split']['train_dates'])} 天 / "
                 f"测试 {len(rf['split']['test_dates'])} 天（按日期切，不混）")
    r = rf["rule"]
    lines.append(f"- 规则标签：暖 > {r['anom_warm']}，冷 < {r['anom_cold']}；"
                 f"后处理 min_area={r['min_area']}，max_radius={r['max_radius_km']} km")
    lines.append("")
    lines.append("> 注意：标签来自规则法，因此下表所有数字都是**与规则法的一致度**，不是准确度；"
                 "像素级指标尤其不能当作独立验证。")
    lines.append("")

    lines.append("| 指标 | 随机森林 | U-Net | 更优 |")
    lines.append("|---|---:|---:|---:|")
    for label, path, f, hb in ROWS:
        va, vb = dig(rf, path), dig(unet, path)
        lines.append(f"| {label} | {fmt(va, path, f)} | {fmt(vb, path, f)} | "
                     f"{mark(vb if vb is not None else None, va, hb)} |")

    # 参数量（只有 U-Net 有）
    n_param = dig(unet, "model.params")
    if n_param:
        lines.append(f"| U-Net 参数量 | — | {int(n_param):,} | |")

    lines.append("")
    lines.append("## 类别计数（训练期有效像素）")
    lines.append("")
    lines.append("| 类别 | 随机森林 | U-Net |")
    lines.append("|---|---:|---:|")
    for k in ("background", "warm", "cold"):
        lines.append(f"| {k} | {rf['class_counts'].get(k)} | {unet['class_counts'].get(k)} |")

    lines.append("")
    lines.append("## 混淆矩阵（行=规则法参考，列=模型预测）")
    lines.append("")
    for title, obj in (("随机森林", rf), ("U-Net", unet)):
        cm = obj["confusion_matrix"]
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| 参考 \\ 预测 | background | warm | cold |")
        lines.append("|---|---:|---:|---:|")
        for i, name in enumerate(("background", "warm", "cold")):
            lines.append(f"| {name} | " + " | ".join(str(v) for v in cm[i]) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="对比两个模型的结果 json")
    p.add_argument("--rf", default="rf_results.json")
    p.add_argument("--unet", default="unet_results.json")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    with open(args.rf, encoding="utf-8") as f:
        rf = json.load(f)
    with open(args.unet, encoding="utf-8") as f:
        unet = json.load(f)

    bad = False
    for name, obj in ((args.rf, rf), (args.unet, unet)):
        miss = check_schema(name, obj)
        if miss:
            print(f"[schema] {name} 缺少字段: {miss}", file=sys.stderr)
            bad = True
    if bad:
        print("[schema] 两份 json 结构不一致，对比不可比（方案 §10 判据 1）", file=sys.stderr)
        return 2

    table = render(rf, unet)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())