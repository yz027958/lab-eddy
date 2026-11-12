#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lab_eddy_rf.py -- 涡旋检测基线：规则法标签 + 随机森林像素分类

本脚本在本实验里有双重身份：

1. **基线模型**：随机森林做逐像素三分类，产出 ``rf_results.json``；
2. **共享模块**：数据缓存、规则标签、后处理（``detections``）、匹配
   （``match_detections``）、指标（``prf``）以及全部关键常量都在这里。
   ``lab_eddy_unet.py`` 直接 ``from lab_eddy_rf import ...``，从而保证
   两个模型的「数据 / 标签 / 后处理参数 / 匹配逻辑」逐项一致。

用法::

    python lab_eddy_rf.py                      # 下载(或读缓存) -> 训练 -> 评估 -> rf_results.json
    python lab_eddy_rf.py --force-download     # 忽略缓存重新下载
    python lab_eddy_rf.py --no-download        # 完全离线（只读已有缓存）

依赖：numpy / scipy / scikit-learn（数据下载只用标准库）。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------------
# 0. 全局常量 —— 两个模型共用，改动任何一项都会让对比失真
# ----------------------------------------------------------------------------

SEED = 0

#: OISST v2.1 网格分辨率 0.25 度；27.75 km 是 25 度纬度处的格距
CELL_KM = 27.75

#: 后处理：与基线完全一致（方案 §6）
MIN_AREA = 12
MAX_RADIUS_KM = 250.0

#: 规则法标签阈值（方案 §2.5）
ANOM_WARM = 0.4
ANOM_COLD = -0.4

#: 检测匹配：预测中心落在参考涡旋半径的多少倍以内算命中
MATCH_RADIUS_FACTOR = 1.0

#: 类别名，顺序即标签值 0/1/2
CLASS_NAMES = ("background", "warm", "cold")

#: 数据窗口（方案 §2.2 选路 1：扩大窗口后切 96x96 切片）
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2024-01-31"
DEFAULT_LAT_RANGE = (20.0, 45.0)
DEFAULT_LON_RANGE = (140.0, 165.0)

#: 时间切分（方案 §2.3）：1–21 训练 / 22–31 测试，日期绝不混进两边
TRAIN_END_DAY = 21

#: ERDDAP 数据源（OISST v2.1 日平均，NOAA CoastWatch/PFEG 镜像）
ERDDAP_DATASET = "ncdcOisst21Agg"
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

CACHE_NAME = "sst_cache.npz"
RF_RESULTS = "rf_results.json"


# ----------------------------------------------------------------------------
# 1. 数据获取与缓存
# ----------------------------------------------------------------------------


@dataclass
class EddyData:
    """一个时空窗口内的 SST 场。

    Attributes:
        sst: ``(T, H, W)`` float32，缺测（陆地 / 云）为 ``NaN``。
        lat: ``(H,)`` float32，格点纬度。
        lon: ``(W,)`` float32，格点经度。
        dates: 长度 ``T`` 的 ``YYYY-MM-DD`` 字符串列表。
    """

    sst: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    dates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.sst.ndim == 3, "sst 必须是 (T, H, W)"
        assert self.sst.shape[1] == self.lat.size
        assert self.sst.shape[2] == self.lon.size

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.sst.shape

    def subset_days(self, idx: Sequence[int]) -> "EddyData":
        idx = list(idx)
        return EddyData(
            sst=self.sst[idx],
            lat=self.lat,
            lon=self.lon,
            dates=[self.dates[i] for i in idx],
        )

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            sst=self.sst,
            lat=self.lat,
            lon=self.lon,
            dates=np.array(self.dates, dtype="U10"),
        )

    @classmethod
    def load(cls, path: str) -> "EddyData":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                sst=z["sst"].astype(np.float32),
                lat=z["lat"].astype(np.float32),
                lon=z["lon"].astype(np.float32),
                dates=[str(d) for d in z["dates"]],
            )


def _date_range(start: str, end: str) -> List[str]:
    """含首尾的日期列表。"""
    from datetime import date, timedelta

    y0, m0, d0 = (int(x) for x in start.split("-"))
    y1, m1, d1 = (int(x) for x in end.split("-"))
    cur, last = date(y0, m0, d0), date(y1, m1, d1)
    out: List[str] = []
    while cur <= last:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def fetch_oisst_day(
    date: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    timeout: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 ERDDAP 取单日 SST 子集。

    用 ERDDAP 的 CSV 子集接口，只下载窗口内的格点（约 1 万行/日），
    避免为了 100x100 的窗口去下 40MB 全球 NetCDF，也免掉了 netCDF4 依赖。

    Returns:
        ``(sst(H, W), lat(H,), lon(W,))``，缺测为 ``NaN``。
    """
    t0 = f"{date}T12:00:00Z"
    query = (
        f"sst[({t0}):1:({t0})]"
        f"[0:1:0]"  # zlev 只有 0.0 m 一层，ERDDAP 要求这里用整数下标，写 0.0 会 400
        f"[({lat_range[0]}):1:({lat_range[1]})]"
        f"[({lon_range[0]}):1:({lon_range[1]})]"
    )
    url = f"{ERDDAP_BASE}/{ERDDAP_DATASET}.csv?" + urllib.parse.quote(query, safe="[]():,")

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
            break
        except Exception as exc:  # noqa: BLE001 - 网络抖动重试
            # ERDDAP 连续请求会被限流，表现为 SSL EOF / 502；退避到 40s 足够
            last_err = exc
            time.sleep(min(5.0 * (2 ** attempt), 40.0))
    else:  # pragma: no cover - 只在网络彻底不可用时触发
        raise RuntimeError(f"下载 {date} 失败: {last_err}")

    reader = csv.reader(io.StringIO(raw))
    next(reader)  # 第一行是列名
    rows = []
    for r in reader:
        # ERDDAP 的 CSV 有「列名 + 单位」两行表头，且可能带尾随空行；
        # 只保留第 3 列能解析成数字的行。
        if len(r) < 5:
            continue
        try:
            float(r[2])
        except ValueError:
            continue
        rows.append(r)
    if not rows:
        raise RuntimeError(f"{date} 返回空数据")

    lats = np.array(sorted({float(r[2]) for r in rows}), dtype=np.float32)
    lons = np.array(sorted({float(r[3]) for r in rows}), dtype=np.float32)
    grid = np.full((lats.size, lons.size), np.nan, dtype=np.float32)
    li = {v: i for i, v in enumerate(lats)}
    lo = {v: i for i, v in enumerate(lons)}
    for r in rows:
        val = r[4]
        if val == "" or val.upper() == "NAN":
            continue
        grid[li[float(r[2])], lo[float(r[3])]] = float(val)
    return grid, lats, lons


def build_cache(
    path: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    lat_range: Tuple[float, float] = DEFAULT_LAT_RANGE,
    lon_range: Tuple[float, float] = DEFAULT_LON_RANGE,
    verbose: bool = True,
) -> EddyData:
    """逐日下载并写出 ``sst_cache.npz``。

    每下一天就把已完成的切片写进 ``<path>.part.npz``，网络中断后重跑会从断点继续
    （ERDDAP 偶发 SSL EOF，31 天的下载不值得从头再来）。
    """
    dates = _date_range(start, end)
    part = path + ".part.npz"

    done: Dict[str, np.ndarray] = {}
    lat = lon = None
    if os.path.exists(part):
        try:
            z = np.load(part, allow_pickle=True)
            lat, lon = z["lat"], z["lon"]
            done = {str(d): g for d, g in zip(z["dates"], z["sst"])}
            print(f"[resume] {part} 已有 {len(done)} 天", flush=True)
        except Exception:  # noqa: BLE001 - 断点文件损坏就重下
            done, lat, lon = {}, None, None

    planes: List[np.ndarray] = []
    for i, d in enumerate(dates, 1):
        if d in done:
            grid = done[d]
        else:
            grid, la, lo = fetch_oisst_day(d, lat_range, lon_range)
            time.sleep(0.6)  # 别把 ERDDAP 打急了
            if lat is None:
                lat, lon = la, lo
            done[d] = grid
            np.savez_compressed(part, sst=np.stack([done[k] for k in dates if k in done]),
                                lat=lat, lon=lon,
                                dates=np.array([k for k in dates if k in done]))
        if grid.shape != (lat.size, lon.size):
            raise RuntimeError(f"{d} 网格尺寸不一致: {grid.shape} vs {(lat.size, lon.size)}")
        planes.append(grid)
        if verbose:
            print(f"  [{i}/{len(dates)}] {d}  {grid.shape}  "
                  f"有效 {np.isfinite(grid).mean() * 100:.1f}%", flush=True)
    data = EddyData(sst=np.stack(planes).astype(np.float32), lat=lat, lon=lon, dates=dates)
    data.save(path)
    if os.path.exists(part):
        os.remove(part)
    return data


def load_data(
    cache_path: str = CACHE_NAME,
    force_download: bool = False,
    no_download: bool = False,
    **kw,
) -> EddyData:
    """读缓存；缓存缺失时按需下载。"""
    if os.path.exists(cache_path) and not force_download:
        return EddyData.load(cache_path)
    if no_download:
        raise FileNotFoundError(f"缺少 {cache_path}，且指定了 --no-download")
    print(f"[data] 未找到缓存，开始下载 OISST -> {cache_path}")
    return build_cache(cache_path, **kw)


# ----------------------------------------------------------------------------
# 2. 距平 / 规则标签 / 特征
# ----------------------------------------------------------------------------


def anomaly_background(sst: np.ndarray, train_idx: Sequence[int]) -> np.ndarray:
    """背景场 = 训练日的时间平均（逐格点 nanmean）。

    方案里写的是「减 31 天背景」；这里只用训练日（1–21 日）算背景，
    避免测试日的 SST 通过背景场泄漏进特征。两处口径一致即可对比。
    """
    with warnings.catch_warnings():
        # 全缺测的格点会让 nanmean 报 "Mean of empty slice"，属预期
        warnings.simplefilter("ignore", RuntimeWarning)
        bg = np.nanmean(sst[list(train_idx)], axis=0)
    return np.where(np.isfinite(bg), bg, 0.0).astype(np.float32)


def compute_anomaly(sst: np.ndarray, background: np.ndarray) -> np.ndarray:
    """SST 距平，缺测保持 ``NaN``。"""
    return (sst - background[None, :, :]).astype(np.float32)


def rule_labels(anom: np.ndarray, warm: float = ANOM_WARM, cold: float = ANOM_COLD) -> np.ndarray:
    """规则法像素标签：``0`` 背景 / ``1`` 暖涡 / ``2`` 冷涡；缺测为 ``-1``。

    方案 §2.5：像素级标签不做连通域过滤（过滤只发生在后处理）。
    """
    label = np.zeros(anom.shape, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        label[anom > warm] = 1
        label[anom < cold] = 2
    label[~np.isfinite(anom)] = -1
    return label


def valid_mask(sst: np.ndarray) -> np.ndarray:
    """``True`` = 该格点有观测。"""
    return np.isfinite(sst)


def pixel_features(sst: np.ndarray, anom: np.ndarray) -> np.ndarray:
    """RF 的像素特征：``[sst, anom, valid]`` -> ``(N, 3)``。"""
    flat = np.stack([sst.ravel(), anom.ravel(), valid_mask(sst).ravel().astype(np.float32)], axis=1)
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def standardize_stats(feats: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """只用训练期有效像素统计均值 / 标准差。"""
    sel = feats[mask]
    mu = sel.mean(axis=0)
    sd = sel.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


# ----------------------------------------------------------------------------
# 3. 后处理：掩膜 -> 涡旋列表（与方案 §6 逐参数一致）
# ----------------------------------------------------------------------------


@dataclass
class Eddy:
    """一个检出涡旋。"""

    cls: int
    lat: float
    lon: float
    radius_km: float
    area: int
    amplitude: float


#: 切分超限大块时，局部极值种子的最小间距（像素）
SPLIT_SEED_FILTER = 7


def _eddy_from_index(
    idx: np.ndarray,
    cls: int,
    lat: np.ndarray,
    lon: np.ndarray,
    anom: np.ndarray | None,
    cell_km: float,
) -> Eddy:
    """由一组像素索引算出中心/半径/振幅。"""
    i_c = float(idx[:, 0].mean())
    j_c = float(idx[:, 1].mean())
    la = float(np.interp(i_c, np.arange(lat.size), lat))
    lo = float(np.interp(j_c, np.arange(lon.size), lon))
    if anom is None:
        amp = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            amp = float(np.nanmean(np.abs(anom[idx[:, 0], idx[:, 1]])))
    return Eddy(cls=cls, lat=la, lon=lo,
                radius_km=float(np.sqrt(idx.shape[0] / np.pi) * cell_km),
                area=int(idx.shape[0]), amplitude=amp)


def split_component(
    binary: np.ndarray,
    anom: np.ndarray,
    cls: int,
    size: int = SPLIT_SEED_FILTER,
) -> np.ndarray | None:
    """把一个超限连通域按距平的局部极值切开。

    做法：区域内找暖涡的局部极大 / 冷涡的局部极小作为种子，然后按
    「最近的种子」把每个像素分给某一块（欧氏 Voronoi，等价于用极值点做
    marker 的分水岭）。用距平而不是距离变换，是因为涡旋的物理中心就是距平
    极值点，这样切出来的每一块都有明确的物理含义。

    Returns:
        ``(H, W)`` 整数标签（0 = 不在该连通域内），或 ``None``（种子不足，切不开）。
    """
    if anom is None:
        return None
    fill = -np.inf if cls == 1 else np.inf
    g = np.where(binary, anom, fill).astype(np.float64)
    if cls == 1:
        seeds = (g == ndimage.maximum_filter(g, size=size, mode="nearest")) & binary
    else:
        seeds = (g == ndimage.minimum_filter(g, size=size, mode="nearest")) & binary
    slab, n_seed = ndimage.label(seeds, structure=np.ones((3, 3), dtype=bool))
    if n_seed < 2:
        return None
    # distance_transform_edt 返回「最近的 0 元素」的索引，0 正好是种子
    nearest = ndimage.distance_transform_edt(slab == 0, return_indices=True)[1]
    piece = slab[nearest[0], nearest[1]]
    piece[~binary] = 0
    return piece


def detections(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    anom: np.ndarray | None = None,
    min_area: int = MIN_AREA,
    max_radius_km: float = MAX_RADIUS_KM,
    cell_km: float = CELL_KM,
    classes: Sequence[int] = (1, 2),
    split_large: bool = False,
    stats: Dict[str, int] | None = None,
) -> List[Eddy]:
    """把像素掩膜转成涡旋列表。

    ``scipy.ndimage.label`` 八连通 -> 面积过滤 -> 半径过滤 -> 中心/半径/振幅。

    Args:
        mask: ``(H, W)`` 整数标签图，值与 ``classes`` 对应。
        anom: 可选，用于计算振幅（区域内平均 |距平|），缺省则振幅为 ``nan``。
        split_large: 超过 ``max_radius_km`` 的连通域默认**整块丢弃**；
            打开后先按距平极值切成若干块再过滤。规则法标签在这种窗口里
            会产生上千公里的连通冷区，整块丢弃会让检测级指标只覆盖极少数像素。
        stats: 可选，回填各种计数（``over_limit`` / ``split_ok`` / ``split_failed``）。

    Returns:
        ``Eddy`` 列表，顺序为 ``(cls, lat)`` 稳定排序，便于复现。
    """
    out: List[Eddy] = []
    structure = np.ones((3, 3), dtype=bool)
    if stats is None:
        stats = {}
    stats.setdefault("over_limit", 0)
    stats.setdefault("split_ok", 0)
    stats.setdefault("split_failed", 0)

    for cls in classes:
        binary = mask == cls
        if not binary.any():
            continue
        lab, n = ndimage.label(binary, structure=structure)
        for k in range(1, n + 1):
            comp = lab == k
            idx = np.argwhere(comp)
            if idx.shape[0] < min_area:
                continue
            if np.sqrt(idx.shape[0] / np.pi) * cell_km <= max_radius_km:
                out.append(_eddy_from_index(idx, cls, lat, lon, anom, cell_km))
                continue
            # 超限大块
            stats["over_limit"] += 1
            if not split_large:
                continue
            piece = split_component(comp, anom, cls)
            if piece is None:
                stats["split_failed"] += 1
                continue
            ok = 0
            for pid in range(1, int(piece.max()) + 1):
                pidx = np.argwhere(piece == pid)
                if pidx.shape[0] < min_area:
                    continue
                if np.sqrt(pidx.shape[0] / np.pi) * cell_km > max_radius_km:
                    continue
                out.append(_eddy_from_index(pidx, cls, lat, lon, anom, cell_km))
                ok += 1
            stats["split_ok"] += 1 if ok else 0
            if not ok:
                stats["split_failed"] += 1

    out.sort(key=lambda e: (e.cls, e.lat, e.lon))
    return out


def coverage_of_detections(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cell_km: float = CELL_KM,
    min_area: int = MIN_AREA,
    max_radius_km: float = MAX_RADIUS_KM,
    classes: Sequence[int] = (1, 2),
    split_large: bool = False,
    anom: np.ndarray | None = None,
) -> Dict[str, Dict[str, float]]:
    """检测级指标到底覆盖了多少已标注像素。

    返回每个类别 ``{"pixels": 总标注像素, "covered": 能形成检测的像素,
    "ratio": covered/pixels, "n_comp": 连通域数, "n_over": 超限连通域数}``。
    比例很低就说明检测级指标只代表一小撮像素，不能当总体性能读。
    """
    structure = np.ones((3, 3), dtype=bool)
    out: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        binary = mask == cls
        lab, n = ndimage.label(binary, structure=structure)
        covered = 0
        n_over = 0
        for k in range(1, n + 1):
            comp = lab == k
            area = int(comp.sum())
            if area < min_area:
                continue
            if np.sqrt(area / np.pi) * cell_km > max_radius_km:
                n_over += 1
                if split_large and anom is not None:
                    piece = split_component(comp, anom, cls)
                    if piece is not None:
                        for pid in range(1, int(piece.max()) + 1):
                            pa = int((piece == pid).sum())
                            if pa >= min_area and np.sqrt(pa / np.pi) * cell_km <= max_radius_km:
                                covered += pa
                continue
            covered += area
        total = int(binary.sum())
        out[CLASS_NAMES[cls]] = {
            "pixels": total,
            "covered": covered,
            "ratio": (covered / total) if total else 0.0,
            "n_comp": n,
            "n_over": n_over,
        }
    return out


def absorption_stats(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    cells: Sequence[int],
    cell_km: float = CELL_KM,
    min_area: int = MIN_AREA,
    max_radius_km: float = MAX_RADIUS_KM,
) -> Dict[str, Dict[str, float]]:
    """参考里那些「本来能检出」的涡旋，有多少被预测的巨型连通域吞掉了。

    「吞掉」的定义：参考涡旋的多数像素在预测里落在同一个**超过半径上限**的
    连通域内。这个数直接暴露后处理的脆性——像素判得再准，只要有一圈误报
    把小块和大块连起来，整块就会被丢弃。
    """
    structure = np.ones((3, 3), dtype=bool)
    out: Dict[str, Dict[str, float]] = {}
    for cls in cells:
        ref_bin = ref_mask == cls
        pred_bin = pred_mask == cls
        rlab, rn = ndimage.label(ref_bin, structure=structure)
        plab, pn = ndimage.label(pred_bin, structure=structure)
        if pn:
            parea = np.bincount(plab.ravel())
        else:
            parea = np.zeros(1, dtype=int)
        n_ref = n_absorbed = 0
        for k in range(1, rn + 1):
            idx = np.argwhere(rlab == k)
            area = idx.shape[0]
            if area < min_area or np.sqrt(area / np.pi) * cell_km > max_radius_km:
                continue
            n_ref += 1
            pk = plab[idx[:, 0], idx[:, 1]]
            pk = pk[pk > 0]
            if pk.size == 0:
                continue
            host = int(np.bincount(pk).argmax())
            if np.sqrt(parea[host] / np.pi) * cell_km > max_radius_km:
                n_absorbed += 1
        out[CLASS_NAMES[cls]] = {
            "n_ref": n_ref,
            "n_absorbed": n_absorbed,
            "rate": (n_absorbed / n_ref) if n_ref else 0.0,
        }
    return out


def great_circle_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """球面距离（km），入参单位为度。"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def match_detections(
    pred: Sequence[Eddy],
    ref: Sequence[Eddy],
    radius_factor: float = MATCH_RADIUS_FACTOR,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """一对一带权匹配（按距离贪心）。

    判据：预测中心与参考中心距离 <= ``radius_factor * 参考半径``。
    每个参考涡旋最多匹配一个预测，反之亦然。

    Returns:
        ``(pairs, unmatched_pred, unmatched_ref)``，其中 ``pairs`` 是
        ``(pred_index, ref_index, distance_km)``。
    """
    if not pred or not ref:
        return [], list(range(len(pred))), list(range(len(ref)))

    plat = np.array([e.lat for e in pred])
    plon = np.array([e.lon for e in pred])
    rlat = np.array([e.lat for e in ref])
    rlon = np.array([e.lon for e in ref])
    rrad = np.array([e.radius_km for e in ref])

    dist = great_circle_km(plat[:, None], plon[:, None], rlat[None, :], rlon[None, :])
    thr = radius_factor * rrad[None, :]
    cand = np.argwhere(dist <= thr)
    if cand.size == 0:
        return [], list(range(len(pred))), list(range(len(ref)))

    order = np.argsort(dist[cand[:, 0], cand[:, 1]])
    used_p, used_r = set(), set()
    pairs: List[Tuple[int, int, float]] = []
    for o in order:
        i, j = int(cand[o][0]), int(cand[o][1])
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        pairs.append((i, j, float(dist[i, j])))
    pairs.sort()
    return (
        pairs,
        [i for i in range(len(pred)) if i not in used_p],
        [j for j in range(len(ref)) if j not in used_r],
    )


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """由计数得 precision / recall / F1。"""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


# ----------------------------------------------------------------------------
# 4. 评估（两个模型共用同一套口径）
# ----------------------------------------------------------------------------


def pixel_metrics(pred: np.ndarray, ref: np.ndarray, classes: Sequence[int] = (1, 2)) -> Dict[str, Dict[str, float]]:
    """逐类像素级 IoU / Dice 与混淆计数（只统计有效像素，``ref >= 0``）。"""
    ok = ref >= 0
    p, r = pred[ok], ref[ok]
    out: Dict[str, Dict[str, float]] = {}
    for c in classes:
        pm, rm = (p == c), (r == c)
        tp = int(np.logical_and(pm, rm).sum())
        fp = int(np.logical_and(pm, ~rm).sum())
        fn = int(np.logical_and(~pm, rm).sum())
        union = tp + fp + fn
        out[CLASS_NAMES[c]] = {
            "iou": tp / union if union else 0.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def confusion_matrix(pred: np.ndarray, ref: np.ndarray, n: int = 3) -> List[List[int]]:
    """``n x n`` 混淆矩阵，行 = 参考，列 = 预测，只统计有效像素。"""
    ok = ref >= 0
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (ref[ok].astype(np.int64), pred[ok].astype(np.int64)), 1)
    return cm.tolist()


def _detection_block(
    preds: np.ndarray,
    refs: np.ndarray,
    anoms: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dates: Sequence[str],
    split: bool,
) -> Tuple[Dict[str, object], Dict[int, Dict[str, List[int]]]]:
    """跑一遍检测级评估，返回 (汇总字典, 逐日计数)。"""
    detection: Dict[str, object] = {}
    daily_counts = {c: {"pred": [], "ref": []} for c in (1, 2)}
    radius_err = {c: [] for c in (1, 2)}
    amp_err = {c: [] for c in (1, 2)}
    tp = {c: 0 for c in (1, 2)}
    fp = {c: 0 for c in (1, 2)}
    fn = {c: 0 for c in (1, 2)}

    for t, _d in enumerate(dates):
        p_list = detections(preds[t], lat, lon, anoms[t], split_large=split)
        r_list = detections(refs[t], lat, lon, anoms[t], split_large=split)
        for c in (1, 2):
            pc = [e for e in p_list if e.cls == c]
            rc = [e for e in r_list if e.cls == c]
            pairs, _up, _ur = match_detections(pc, rc)
            tp[c] += len(pairs)
            fp[c] += len(pc) - len(pairs)
            fn[c] += len(rc) - len(pairs)
            daily_counts[c]["pred"].append(len(pc))
            daily_counts[c]["ref"].append(len(rc))
            for i, j, _dist in pairs:
                radius_err[c].append(abs(pc[i].radius_km - rc[j].radius_km))
                if np.isfinite(pc[i].amplitude) and np.isfinite(rc[j].amplitude):
                    amp_err[c].append(abs(pc[i].amplitude - rc[j].amplitude))

    for c in (1, 2):
        name = CLASS_NAMES[c]
        m = prf(tp[c], fp[c], fn[c])
        detection[name] = {
            "micro": m,
            "macro_f1": m["f1"],
            "radius_mae_km": float(np.mean(radius_err[c])) if radius_err[c] else 0.0,
            "amp_mae": float(np.mean(amp_err[c])) if amp_err[c] else 0.0,
        }
    detection["macro_f1"] = float(np.mean([detection[CLASS_NAMES[c]]["macro_f1"] for c in (1, 2)]))
    return detection, daily_counts


def evaluate(
    preds: np.ndarray,
    refs: np.ndarray,
    anoms: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dates: Sequence[str],
    infer_time_s: float,
    rule_time_s: float,
) -> Dict[str, object]:
    """把逐日预测拼成方案 §7 要求的评估字典。

    Args:
        preds: ``(T, H, W)`` 预测标签（0/1/2）。
        refs: ``(T, H, W)`` 规则法参考标签（0/1/2，缺测 -1）。
        anoms: ``(T, H, W)`` 距平，用于振幅与阈值漂移。
    """
    pixel = pixel_metrics(preds, refs)
    cm = confusion_matrix(preds, refs)

    # --- 检测级 ---
    detection, daily_counts = _detection_block(preds, refs, anoms, lat, lon, dates, split=False)
    # 超限大块先切再过滤的版本（方案常量不变，只多加一步切分）
    detection_split, _ = _detection_block(preds, refs, anoms, lat, lon, dates, split=True)

    # --- 逐日检测总数相关 r（两类各算一个，取平均）---
    rs = []
    for c in (1, 2):
        a = np.asarray(daily_counts[c]["pred"], dtype=float)
        b = np.asarray(daily_counts[c]["ref"], dtype=float)
        rs.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0)
    daily_count_corr = float(np.nanmean(rs)) if rs else 0.0

    # --- 阈值漂移：固定预测，改变参考标签的阈值 ---
    drift: Dict[str, float] = {}
    for name, thr in (("thresh_0.3", 0.3), ("thresh_0.5", 0.5)):
        ref_alt = rule_labels(anoms, warm=thr, cold=-thr)
        m = pixel_metrics(preds, ref_alt)
        drift[name] = float(np.mean([m["warm"]["iou"], m["cold"]["iou"]]))

    # --- 诊断 1：参考里本来能检出的涡旋，有多少被预测的巨型块吞掉 ---
    absorbed = {c: {"n_ref": 0, "n_absorbed": 0} for c in (1, 2)}
    for t in range(len(dates)):
        st = absorption_stats(preds[t], refs[t], (1, 2))
        for c in (1, 2):
            absorbed[c]["n_ref"] += int(st[CLASS_NAMES[c]]["n_ref"])
            absorbed[c]["n_absorbed"] += int(st[CLASS_NAMES[c]]["n_absorbed"])
    absorption = {
        CLASS_NAMES[c]: {
            "n_ref": absorbed[c]["n_ref"],
            "n_absorbed": absorbed[c]["n_absorbed"],
            "rate": (absorbed[c]["n_absorbed"] / absorbed[c]["n_ref"]) if absorbed[c]["n_ref"] else 0.0,
        }
        for c in (1, 2)
    }

    # --- 诊断 2：检测级指标覆盖了多少已标注像素 ---
    cov = {n: {"pixels": 0, "covered": 0, "covered_split": 0, "n_comp": 0, "n_over": 0}
           for n in CLASS_NAMES[1:]}
    for t in range(len(dates)):
        a = coverage_of_detections(refs[t], lat, lon, split_large=False)
        b = coverage_of_detections(refs[t], lat, lon, split_large=True, anom=anoms[t])
        for n in cov:
            for k in ("pixels", "covered", "n_comp", "n_over"):
                cov[n][k] += int(a[n][k])
            cov[n]["covered_split"] += int(b[n]["covered"])
    coverage = {
        n: {
            **cov[n],
            "ratio": (cov[n]["covered"] / cov[n]["pixels"]) if cov[n]["pixels"] else 0.0,
            "ratio_split": (cov[n]["covered_split"] / cov[n]["pixels"]) if cov[n]["pixels"] else 0.0,
        }
        for n in cov
    }

    return {
        "pixel": pixel,
        "confusion_matrix": cm,
        "detection": detection,
        "detection_split": detection_split,
        "absorption": absorption,
        "coverage": coverage,
        "threshold_drift": drift,
        "daily_count_corr": daily_count_corr,
        "timing": {
            "unet_infer_per_day_s": float(infer_time_s / max(len(dates), 1)),
            "rule_per_day_s": float(rule_time_s / max(len(dates), 1)),
        },
    }


def build_meta(
    data: EddyData,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    class_counts: Dict[str, int],
    train_time_s: float,
) -> Dict[str, object]:
    """两个 json 共有的头部信息（data / split / rule / class_counts）。"""
    return {
        "data": {
            "source": "NOAA OISST v2.1 daily (AVHRR, 0.25 deg) via ERDDAP",
            "start": data.dates[0],
            "end": data.dates[-1],
            "grid": list(data.shape),
            "lat_range": [float(data.lat[0]), float(data.lat[-1])],
            "lon_range": [float(data.lon[0]), float(data.lon[-1])],
            "cell_km": CELL_KM,
        },
        "split": {
            "train_dates": [data.dates[i] for i in train_idx],
            "test_dates": [data.dates[i] for i in test_idx],
            "policy": "time-based, no date shared between splits",
        },
        "rule": {
            "anom_warm": ANOM_WARM,
            "anom_cold": ANOM_COLD,
            "min_area": MIN_AREA,
            "max_radius_km": MAX_RADIUS_KM,
            "cell_km": CELL_KM,
            "match_radius_factor": MATCH_RADIUS_FACTOR,
        },
        "class_counts": class_counts,
        "train_time_s": float(train_time_s),
    }


def class_counts_of(labels: np.ndarray, classes: Sequence[int] = (0, 1, 2)) -> Dict[str, int]:
    """有效像素上的类别计数。"""
    ok = labels >= 0
    return {CLASS_NAMES[c]: int((labels[ok] == c).sum()) for c in classes}


def class_weights_from_counts(counts: Dict[str, int], clamp: Tuple[float, float] = (0.5, 10.0)) -> np.ndarray:
    """按实测类频率的倒数定权，再裁到 ``clamp`` 区间。"""
    freq = np.array([counts[n] for n in CLASS_NAMES], dtype=np.float64)
    freq = np.maximum(freq, 1.0)
    w = freq.sum() / (len(CLASS_NAMES) * freq)
    return np.clip(w, clamp[0], clamp[1]).astype(np.float32)


# ----------------------------------------------------------------------------
# 5. 随机森林基线主流程
# ----------------------------------------------------------------------------


def train_random_forest(
    feats: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 200,
    seed: int = SEED,
    n_jobs: int = 1,
) -> RandomForestClassifier:
    """在训练日全部有效像素上拟合 RF。

    ``n_jobs=1`` 是默认值：joblib 的线程池在部分受限环境（无命名管道权限）会失败，
    需要跑满多核时显式传 ``--jobs -1``。
    """
    ok = labels >= 0
    X, y = feats[ok], labels[ok]
    # 下采样背景类，避免 60 万背景像素把训练拖死（只影响训练速度，不改变预测口径）
    rng = np.random.default_rng(seed)
    bg = np.flatnonzero(y == 0)
    if bg.size > 200_000:
        keep_bg = rng.choice(bg, 200_000, replace=False)
        keep = np.concatenate([keep_bg, np.flatnonzero(y != 0)])
        keep.sort()
        X, y = X[keep], y[keep]
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    clf.fit(X, y)
    return clf


def predict_days_rf(clf: RandomForestClassifier, data: EddyData, idx: Sequence[int], anom: np.ndarray) -> Tuple[np.ndarray, float]:
    """逐日整幅预测，返回 ``(T, H, W)`` 标签与总耗时。"""
    out = np.empty((len(idx), *data.sst.shape[1:]), dtype=np.int64)
    t0 = time.perf_counter()
    for k, i in enumerate(idx):
        feats = pixel_features(data.sst[i], anom[i])
        out[k] = clf.predict(feats).reshape(data.sst.shape[1:])
    return out, time.perf_counter() - t0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="涡旋检测基线：随机森林像素分类")
    p.add_argument("--cache", default=CACHE_NAME)
    p.add_argument("--out", default=RF_RESULTS)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--lat0", type=float, default=DEFAULT_LAT_RANGE[0])
    p.add_argument("--lat1", type=float, default=DEFAULT_LAT_RANGE[1])
    p.add_argument("--lon0", type=float, default=DEFAULT_LON_RANGE[0])
    p.add_argument("--lon1", type=float, default=DEFAULT_LON_RANGE[1])
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--jobs", type=int, default=1, help="RF 并行度，-1 = 全部核心")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)

    np.random.seed(SEED)

    data = load_data(
        args.cache,
        force_download=args.force_download,
        no_download=args.no_download,
        start=args.start,
        end=args.end,
        lat_range=(args.lat0, args.lat1),
        lon_range=(args.lon0, args.lon1),
    )
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

    # --- 训练 ---
    feats = np.concatenate([pixel_features(data.sst[i], anom[i]) for i in train_idx], axis=0)
    tr_labels = np.concatenate([labels[i].ravel() for i in train_idx], axis=0)
    t0 = time.perf_counter()
    clf = train_random_forest(feats, tr_labels, n_estimators=args.trees, n_jobs=args.jobs)
    train_time = time.perf_counter() - t0
    print(f"[train] RF {args.trees} trees, {train_time:.1f}s")

    # --- 测试 ---
    preds, infer_time = predict_days_rf(clf, data, test_idx, anom)
    t_rule = time.perf_counter()
    refs = labels[test_idx]
    rule_time = time.perf_counter() - t_rule

    result = build_meta(data, train_idx, test_idx, counts, train_time)
    result.update(evaluate(preds, refs, anom[test_idx], data.lat, data.lon,
                           [data.dates[i] for i in test_idx], infer_time, rule_time))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}")
    print(f"       warm IoU={result['pixel']['warm']['iou']:.4f}  cold IoU={result['pixel']['cold']['iou']:.4f}")
    print(f"       detection macro F1={result['detection']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())