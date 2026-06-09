"""
引导滤波 (Guided Filter)
===========================
基于 He et al. "Guided Image Filtering" (2013) 的纯 NumPy/OpenCV 实现。

数学原理:
    引导滤波假设输出 q 是引导图 I 在局部窗口内的线性变换:
        q_i = a_k * I_i + b_k,  ∀i ∈ ω_k

    最小化代价函数:
        E(a_k, b_k) = Σ((a_k*I_i + b_k - p_i)² + ε*a_k²)

    最优解:
        a_k = cov(I, p) / (var(I) + ε)
        b_k = mean(p) - a_k * mean(I)
        q_i = mean(a)_i * I_i + mean(b)_i

复杂度: O(N) — 使用盒式滤波加速，窗口大小不影响复杂度。

使用场景:
- 边缘感知平滑 (I = p = 输入图像)
- Alpha Matte 边缘优化 (I = 灰度引导图, p = 粗alpha)
- 细节增强 (I = p, ε 取负或小值)
"""

import numpy as np
import cv2


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    引导滤波 — 标准实现。

    :param guide: 引导图 (H, W) float32 或 (H, W, C)
    :param src: 待滤波信号 (H, W) float32
    :param radius: 窗口半径 (像素)
    :param eps: 正则化参数 (防止除零)
    :return: 滤波后信号 (H, W) float32

    算法步骤:
        1. 计算引导图局部统计量: mean_I, var_I
        2. 计算信号的局部统计量: mean_p
        3. 计算互相关: corr_Ip
        4. 计算线性系数: a, b
        5. 取窗口均值: mean_a, mean_b
        6. 重构: q = mean_a * I + mean_b
    """
    # 确保 float32
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    # 对多通道引导图，使用第一通道（灰度）或所有通道均值
    if guide.ndim == 3 and guide.shape[2] > 1:
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY) if guide.shape[2] == 3 else guide.mean(axis=2)
    else:
        guide_gray = guide.squeeze()

    # 归一化到 [0, 1]
    if guide_gray.max() > 1.0:
        guide_gray = guide_gray / 255.0
    if src.max() > 1.0:
        src = src / 255.0

    # 核尺寸
    ksize = (radius, radius)

    # Step 1: 盒式滤波计算局部统计量 (O(N) 复杂度)
    mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ksize, normalize=True)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize, normalize=True)

    # Step 2: 计算引导图方差和互协方差
    corr_I = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ksize, normalize=True)
    corr_Ip = cv2.boxFilter(guide_gray * src, cv2.CV_32F, ksize, normalize=True)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    # Step 3: 计算线性系数
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Step 4: 对系数取窗口均值
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True)

    # Step 5: 重构输出
    q = mean_a * guide_gray + mean_b

    # 裁剪到 [0, 1]
    return np.clip(q, 0.0, 1.0)


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 4,
    eps: float = 1e-4,
    subsample: int = 2,
) -> np.ndarray:
    """
    快速引导滤波 — 通过下采样加速。

    先对引导图和信号下采样，在低分辨率上计算 a/b 系数，
    再上采样回原始分辨率重构输出。

    :param subsample: 下采样比例，=2 时加速约 4 倍
    """
    h, w = src.shape[:2]
    small_h, small_w = h // subsample, w // subsample

    guide_small = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
    src_small = cv2.resize(src, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 在小图上计算系数
    guide_small = guide_small.astype(np.float32)
    src_small = src_small.astype(np.float32)

    if guide_small.ndim == 3:
        guide_gray = cv2.cvtColor(guide_small, cv2.COLOR_BGR2GRAY) if guide_small.shape[2] == 3 else guide_small.mean(axis=2)
    else:
        guide_gray = guide_small.squeeze()

    if guide_gray.max() > 1.0:
        guide_gray /= 255.0
    if src_small.max() > 1.0:
        src_small /= 255.0

    r_small = max(1, radius // subsample)
    ksize = (r_small, r_small)

    mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ksize, normalize=True)
    mean_p = cv2.boxFilter(src_small, cv2.CV_32F, ksize, normalize=True)
    corr_I = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ksize, normalize=True)
    corr_Ip = cv2.boxFilter(guide_gray * src_small, cv2.CV_32F, ksize, normalize=True)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True)

    # 上采样系数
    mean_a_full = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
    mean_b_full = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)

    # 在全分辨率上重构
    if guide.max() > 1.0:
        guide = guide.astype(np.float32) / 255.0
    if guide.ndim == 3:
        guide_gray_full = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY) if guide.shape[2] == 3 else guide.mean(axis=2)
    else:
        guide_gray_full = guide.squeeze().astype(np.float32)

    q = mean_a_full * guide_gray_full + mean_b_full
    return np.clip(q, 0.0, 1.0)
