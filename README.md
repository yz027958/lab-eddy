# lab-eddy —— 涡旋检测：随机森林基线 vs U-Net

一个可复现的小实验：**同一份 SST、同一套规则标签、同一个后处理、同一套匹配指标**，
对照「随机森林逐像素分类」与「U-Net 逐像素分割」两个模型。

- **参考工作**：Lguensat et al., *EddyNet: A Deep Neural Network For Pixel-Wise Classification of
  Oceanic Eddies* (IGARSS 2018, arXiv:1711.03954) —— 把涡旋检测写成逐像素三分类
  `{0: 非涡旋, 1: 反气旋涡, 2: 气旋涡}`；
  架构来自 Ronneberger et al., *U-Net* (MICCAI 2015, arXiv:1505.04597)。
- **本仓库实现的方案**：`UNet-实施方案与要求.md`（2026-11-04）。

> **先说最重要的一件事**：标签来自规则法（距平阈值 + 连通域），而规则法本身就是个阈值。
> 所以本仓库所有数字都是**与规则法的一致度**，不是**准确度**。
> 真正的验证需要人工复核的涡旋目录；没有它，像素级 IoU 高（基线 RF 已到 0.99+）几乎不构成优势。
> 因此主要论据放在**检测级指标**与**阈值漂移**上。

---

## 1. 目录

```
lab-eddy/
├── lab_eddy_rf.py       # 基线：规则标签 + 随机森林；同时是共享模块（数据/标签/后处理/指标）
├── lab_eddy_unet.py     # U-Net：切片 -> 训练 -> 整幅推理 -> 同一套评估
├── compare_models.py    # 只读两个 json，出对比表（对比逻辑不写进模型脚本）
├── make_figures.py      # 三联图 / 训练曲线 / 混淆矩阵
├── tests/test_pipeline.py  # 合成数据端到端冒烟测试（不联网）
├── sst_cache.npz        # 数据缓存（自动生成，已 gitignore）
├── rf_results.json      # 基线结果
├── unet_results.json    # U-Net 结果（共享字段与 rf_results.json 完全一致，另多一个 model 块存超参）
├── unet_weights.pt      # 权重（含标准化统计量与背景场）
├── compare.md           # compare_models.py 的产出
├── REPORT.md            # 实验报告（含本次实跑数字）
└── figures/
```

**已经跑过一次**（2024-01 全月，CPU 约 1.5 分钟训练）：结论与图见 `REPORT.md` 和 `figures/`。

`lab_eddy_unet.py` 直接 `from lab_eddy_rf import detections, match_detections, prf, MIN_AREA, ...`，
保证两个模型的后处理参数与匹配逻辑**只有一份实现**。

---

## 2. 数据

| 项 | 值 |
|---|---|
| 源 | NOAA OISST v2.1 日平均（AVHRR, 0.25°），经 NOAA CoastWatch ERDDAP `ncdcOisst21Agg` 取子集 |
| 时段 | 2024-01-01 ~ 2024-01-31（31 天） |
| 窗口 | lat 20–45°N / lon 140–165°E（方案 §2.2 选路 1） |
| 网格 | 100 × 100 × 31（0.25°） |
| 缓存 | `sst_cache.npz`（`sst/lat/lon/dates`），首次运行自动下载约 31 次请求，之后只读缓存 |

**为什么扩窗口**：原窗口只有 40×60 格点，放不下 96×96 切片。扩到 100×100 后可以切 96×96，
与 EddyNet 的切片尺度可比；代价是基线也必须在新窗口重跑（本仓库的 `lab_eddy_rf.py` 就是这个基线）。

日期切分（方案 §2.3）：

- **训练**：1–21 日；**测试**：22–31 日；
- 切片可以打乱，但**日期绝不混进两边**；
- 训练期的任何统计量（SST 均值/标准差、距平背景场）都只用 1–21 日。

---

## 3. 标签

```python
anom = sst - background            # background = 训练日逐格点时间平均
label = 0 (|anom| <= 0.4) / 1 (anom > 0.4, 暖涡) / 2 (anom < -0.4, 冷涡)
label[sst is NaN] = -1             # 缺测，loss 里权重置 0
```

- 像素级标签**不做连通域过滤**（过滤只发生在后处理），与基线一致；
- 背景场只用训练日，避免测试日通过距平泄漏进特征。

**缺测处理（硬要求）**：OISST 有陆地格与云遮挡 `NaN`。做法是
① 第 3 通道显式标出有效掩膜；② 归一化前把 `NaN` 置 0；③ loss 对缺测像素 `ignore_index=-1`。

---

## 4. 网络

- 编码器 4 级下采样，每级 `3×3 conv + BN + ReLU` ×2；
- 解码器对称上采样（转置卷积），每级与编码器同级特征 **concat**（不是 add）；
- 输出 `1×1 conv → 3 logits`；
- 输入通道 `[SST 标准化, 距平标准化, 有效掩膜]`。

**关于参数量（对方案的一处必要取舍）**：方案 §3.1 要求通道 `[32,64,128,256]` + 瓶颈 512，
同时要求参数量 < 2M —— 这两个条件在「每级两个 3×3 卷积」下无法同时满足：
该配置约 **7.8M** 参数。本仓库默认：

| 配置 | 通道 | 参数量 | 用途 |
|---|---|---|---|
| `--base 16`（默认） | `[16,32,64,128]` + 瓶颈 256 | **≈1.9M** | 满足 <2M 与 CPU 可训 |
| `--base 32` | `[32,64,128,256]` + 瓶颈 512 | ≈7.8M | 复现方案原表 |

两个配置的网络**结构形状完全一致**（4 级下采样 / 双卷积 / concat skip / 1×1 head），只是宽度不同；
报告里如实说明这一取舍即可。

**不换 SELU**（方案 §3.2）：样本量小，SELU 的收益不成立，且 EddyNet 作者自己也报告直接替换会让损失变噪。
`Dropout2d` 默认关闭，可用 `--dropout` 打开。

---

## 5. 训练与损失

| 项 | 值 |
|---|---|
| 优化器 | Adam, lr 1e-3 |
| 调度 | `ReduceLROnPlateau(factor=0.5, patience=5)`，监控**训练 loss** |
| batch | 8 |
| epoch | 60 |
| 种子 | 0（`np.random.seed` + `torch.manual_seed`） |
| 切片 | 96×96，stride 48（50% 重叠），反射填充 |

```text
L = 0.5 * (1 - Dice_macro) + 1.0 * CE(weighted, ignore_index=-1)
```

- Dice 按类计算后取平均，**只让出现在 batch 里的类参与**（空类不把 loss 拉成 0）；
- CE 用类权重（方案 §4 的估计值是 `1.0 : 3.0 : 1.5`，脚本默认**按训练集实测类频率反比自动算**，
  跑完把实测值写进报告）；缺测像素权重为 0；
- 两个都用的理由：纯 Dice 早期梯度不稳，纯 CE 在小目标上被背景淹没。这是工程选择，不宣称最优。

**无验证集选择偏差**：LR 调度只看训练 loss；权重取最后一个 epoch。
测试日只用于最终报告，不参与任何决策（曲线里的 `val_IoU` 纯粹是画图监控）。

---

## 6. 后处理：与基线逐参数一致（方案 §6）

```python
MIN_AREA = 12          # 与 lab_eddy_rf.py 一致
MAX_RADIUS_KM = 250
CELL_KM = 27.75
MATCH_RADIUS_FACTOR = 1.0   # 预测中心落在参考涡旋半径 1 倍内算命中
```

`argmax 掩膜 → scipy.ndimage.label(8 连通) → 面积过滤 → 半径过滤 → 中心/半径/振幅`。
参数一改，检测级指标就不可比 —— 这一步没有自由度，两个脚本共用同一份实现。

**超限大块的两种处理**（`detections(..., split_large=)`）：

- `False`（方案原始口径）：半径超 `MAX_RADIUS_KM` 的连通域**整块丢弃**。这块窗口里规则法的
  冷区常连成一整片（约 8.4×10³ 像素、等效半径 1.4×10³ km），整块丢下去，检测级指标只覆盖
  **0.67%** 的冷像素 —— 它量的其实是一天 1–2 个孤立小涡。
- `True`（诊断口径，新增）：先用**距平的局部极值**做 marker（暖涡取极大、冷涡取极小，
  种子最小间距 7 像素），按最近种子把连通域切成多块（欧氏 Voronoi，等价极值 marker 分水岭），
  再走同一套 `MIN_AREA / MAX_RADIUS_KM`。覆盖度回到 **99.91%**。

两个口径都会写进结果 json（`detection` / `detection_split`），主表仍以方案原始口径为准。
注意切分用的是**同一个距平场**给两边取种子，所以它是「同一切分下两个掩膜是否吻合」，
不是独立的检测能力（见 `REPORT.md` §4.5）。

---

## 7. 评估

`unet_results.json` 与 `rf_results.json` **schema 完全一致**：

```json
{
  "data": {...}, "split": {...}, "rule": {...},
  "class_counts": {...}, "train_time_s": 0.0,
  "pixel": {"warm": {"iou": 0, "dice": 0, "tp": 0, "fp": 0, "fn": 0}, "cold": {...}},
  "confusion_matrix": [[0,0,0],[0,0,0],[0,0,0]],
  "detection": {"warm": {"micro": {...}, "macro_f1": 0, "radius_mae_km": 0, "amp_mae": 0}, "cold": {...}},
  "detection_split": {"warm": {...}, "cold": {...}, "macro_f1": 0},
  "absorption": {"warm": {"n_ref": 0, "n_absorbed": 0, "rate": 0}, "cold": {...}},
  "coverage": {"warm": {"pixels": 0, "covered": 0, "ratio": 0, "n_comp": 0, "n_over": 0,
                        "covered_split": 0, "ratio_split": 0}, "cold": {...}},
  "threshold_drift": {"thresh_0.3": 0, "thresh_0.5": 0},
  "daily_count_corr": 0.0,
  "timing": {"unet_infer_per_day_s": 0.0, "rule_per_day_s": 0.0}
}
```

阈值漂移的做法：**固定模型预测**，把参考标签的阈值换成 ±0.3 / ±0.5 重算，看 IoU 掉多少 ——
用来检验模型是不是只把 0.4 这个阈值背了下来。

两个诊断字段帮助判断「检测级 F1 还能不能读」：

- `coverage`：检测级指标实际覆盖了多少已标注像素。比例很低时，检测级数字只代表一小撮像素。
- `absorption`：参考里本来能形成检测的涡旋，有多少被预测的**超限**连通域吞掉
  （像素判对也没用，整块会被丢弃）。

---

## 8. 复现

```bash
pip install -r requirements.txt

# 一条命令跑完全流程（首次会下载数据并写 sst_cache.npz）
python lab_eddy_rf.py && \
python lab_eddy_unet.py && \
python compare_models.py --out compare.md && \
python make_figures.py

# 只改了后处理/指标、模型没变时，别重训（会从 unet_weights.pt 复原结构）
python lab_eddy_unet.py --eval-only
```

先验证流水线（方案 §8：先用 5 epoch 跑通）：

```bash
python lab_eddy_unet.py --epochs 5 --out unet_results_5ep.json
```

不联网的冒烟测试：

```bash
python -m unittest discover -s tests -v
```

消融（方案 §7.3）：

```bash
python lab_eddy_unet.py --no-skip            --out unet_noskip.json
python lab_eddy_unet.py --channels sst_only  --out unet_sst.json
python lab_eddy_unet.py --loss ce            --out unet_ce.json
```

环境：Python 3.11/3.12 + `torch==2.4.0+cpu`（CPU 即可，无需 GPU）。
Windows 上若 `import torch` 报 `Error loading fbgemm.dll`，是 torch 2.4 缺
`libomp140.x86_64.dll`（pytorch#133488），换 `torch>=2.5` 或装 VC++ 2022 运行库即可。

---

## 9. 已知局限（写进报告，别藏）

1. **标签循环**：标签由规则法生成，模型再高的像素 IoU 也只是「把阈值学了一遍」。
   基线 RF 已到 IoU 0.99+，所以主结论必须落在检测级与阈值漂移。
2. **没有独立真值**：这只是与规则法的一致度，不是准确度。
3. **样本量小**：100×100 的场切 96×96 + stride 48 只有 **4 片/日**（训练 84 片）。
   这是扩窗方案在下垫面上的硬限制；`--stride 24` 可提到 16 片/日（训练 336 片），
   报告里要写清用的哪个值。
4. **窗口扩到 20–45°N** 后涡旋统计特征与原来 30–40°N 窗口不完全可比，
   基线必须在同一新窗口重跑（本仓库已如此）。