"""
图像质量检测规则 (Image Quality Checks)
========================================
检测证件照的图像质量是否满足印刷/提交要求。

规则:
- Q01: 分辨率检测 — 输出尺寸是否达到目标
- Q02: 清晰度检测 — Laplacian 方差
- Q03: 摩尔纹检测 — 频域分析（简化）
- Q04: 压缩质量 — 块效应分析
"""

import numpy as np
import cv2
from typing import Tuple

from .models import RuleResult, Verdict


def check_resolution(
    image: np.ndarray, spec_size: Tuple[int, int], standard
) -> RuleResult:
    """Q01: 分辨率检测"""
    h, w = image.shape[:2]
    min_px = standard.min_resolution_px

    if h >= min_px and w >= min_px:
        return RuleResult(
            rule_id="Q01", category="quality", name="分辨率",
            verdict=Verdict.PASS, score=100,
            detail=f"分辨率 {w}×{h} 满足要求 (≥ {min_px}px)",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="Q01", category="quality", name="分辨率",
            verdict=Verdict.FAIL,
            score=max(0, int(min(h, w) / min_px * 100)),
            detail=f"分辨率不足 {w}×{h} (要求 ≥ {min_px}px)",
            hint="请使用更高分辨率的相机拍摄，或减少裁剪",
            critical=False,
        )


def check_sharpness(image: np.ndarray, standard) -> RuleResult:
    """Q02: 清晰度检测 — Laplacian 方差"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    threshold = standard.sharpness_min

    if variance >= threshold:
        return RuleResult(
            rule_id="Q02", category="quality", name="清晰度",
            verdict=Verdict.PASS, score=100,
            detail=f"清晰度良好 (Laplacian方差: {variance:.1f}, 阈值: {threshold})",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="Q02", category="quality", name="清晰度",
            verdict=Verdict.FAIL,
            score=max(0, int(variance / threshold * 100)),
            detail=f"图像可能模糊 (Laplacian方差: {variance:.1f}, 阈值: {threshold})",
            hint="请确保拍摄时对焦准确、手不抖动",
            critical=False,
        )


def check_moire(image: np.ndarray, standard=None) -> RuleResult:
    """Q03: 摩尔纹检测 — 简化频域分析"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # 2D FFT 分析
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    # 排除 DC 中心区域
    center_h, center_w = h // 2, w // 2
    mask_size = 5
    y1 = max(0, center_h - mask_size)
    y2 = min(h, center_h + mask_size)
    x1 = max(0, center_w - mask_size)
    x2 = min(w, center_w + mask_size)

    mag_no_dc = magnitude.copy()
    mag_no_dc[y1:y2, x1:x2] = 0

    # 搜索异常高频峰值（可能的摩尔纹）
    # 计算高频（外部区域）的峰值
    outer_radius = min(h, w) // 3
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
    outer_mask = dist > outer_radius

    outer_mag = mag_no_dc[outer_mask]
    if outer_mag.size == 0:
        return RuleResult(
            rule_id="Q03", category="quality", name="摩尔纹",
            verdict=Verdict.PASS, score=100,
            detail="未检测到异常高频模式",
            hint=None, critical=False,
        )

    mean_mag = np.mean(outer_mag)
    max_mag = np.max(outer_mag) if outer_mag.size > 0 else 0
    # 异常峰值检测：高频区域存在远高于均值的峰值
    peak_ratio = max_mag / mean_mag if mean_mag > 0 else 1.0

    if peak_ratio < 20:
        return RuleResult(
            rule_id="Q03", category="quality", name="摩尔纹",
            verdict=Verdict.PASS, score=100,
            detail=f"频域分析正常 (峰值比: {peak_ratio:.1f})",
            hint=None, critical=False,
        )
    elif peak_ratio < 35:
        return RuleResult(
            rule_id="Q03", category="quality", name="摩尔纹",
            verdict=Verdict.WARN, score=70,
            detail=f"可能存在轻微摩尔纹 (峰值比: {peak_ratio:.1f})",
            hint="建议检查原图是否存在扫描或拍摄引起的条纹",
            critical=False,
        )
    else:
        return RuleResult(
            rule_id="Q03", category="quality", name="摩尔纹",
            verdict=Verdict.FAIL, score=30,
            detail=f"检测到异常高频模式 (峰值比: {peak_ratio:.1f})",
            hint="请使用原始照片而非扫描件，或调整拍摄距离",
            critical=False,
        )


def check_compression_artifacts(image: np.ndarray, standard=None) -> RuleResult:
    """Q04: 压缩质量检测 — 8×8 块边界分析"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape

    if h < 16 or w < 16:
        return RuleResult(
            rule_id="Q04", category="quality", name="压缩质量",
            verdict=Verdict.WARN, score=50,
            detail="图像过小无法检测",
            hint=None, critical=False,
        )

    # 检查 8×8 块边界（JPEG 压缩特征）
    block_size = 8
    boundary_diff = 0
    count = 0

    # 水平边界扫描
    for y in range(block_size - 1, h - block_size, block_size):
        for x in range(w):
            diff = abs(float(gray[y, x]) - float(gray[y + 1, x]))
            boundary_diff += diff
            count += 1

    # 垂直边界扫描
    for y in range(h):
        for x in range(block_size - 1, w - block_size, block_size):
            diff = abs(float(gray[y, x]) - float(gray[y, x + 1]))
            boundary_diff += diff
            count += 1

    avg_boundary_diff = boundary_diff / count if count > 0 else 0

    # 内部差异（非边界）
    inner_diff = 0
    inner_count = 0
    for y in range(h):
        for x in range(w - 1):
            if (x + 1) % block_size != 0:
                diff = abs(float(gray[y, x]) - float(gray[y, x + 1]))
                inner_diff += diff
                inner_count += 1

    avg_inner_diff = inner_diff / inner_count if inner_count > 0 else 0

    # 块效应指标: 边界差异 / 内部差异
    blocking_ratio = avg_boundary_diff / avg_inner_diff if avg_inner_diff > 0 else 1.0

    if blocking_ratio < 1.3:
        return RuleResult(
            rule_id="Q04", category="quality", name="压缩质量",
            verdict=Verdict.PASS, score=100,
            detail=f"未检测到明显块效应 (块比: {blocking_ratio:.2f})",
            hint=None, critical=False,
        )
    elif blocking_ratio < 1.6:
        return RuleResult(
            rule_id="Q04", category="quality", name="压缩质量",
            verdict=Verdict.WARN, score=75,
            detail=f"轻微块效应 (块比: {blocking_ratio:.2f})",
            hint="建议使用较低压缩比的原始照片",
            critical=False,
        )
    else:
        return RuleResult(
            rule_id="Q04", category="quality", name="压缩质量",
            verdict=Verdict.FAIL,
            score=max(0, int((2.0 - blocking_ratio) * 100)),
            detail=f"明显块效应，可能经过过度压缩 (块比: {blocking_ratio:.2f})",
            hint="请使用原始质量的照片，避免多次压缩",
            critical=False,
        )


def run_quality_checks(
    image: np.ndarray,
    spec_size: Tuple[int, int] = (295, 413),
    standard=None,
) -> list:
    """运行所有图像质量检测规则"""
    return [
        check_resolution(image, spec_size, standard),
        check_sharpness(image, standard),
        check_moire(image, standard),
        check_compression_artifacts(image, standard),
    ]
