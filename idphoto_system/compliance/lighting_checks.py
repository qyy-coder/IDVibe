"""
光照与色彩检测规则 (Illumination & Color Checks)
=================================================
检测证件照的光照质量和背景色彩合规性。

规则:
- L01: 面部光照均匀度 — 左右脸亮度差异
- L02: 阴影检测 — 面部/背景网格分析
- L03: 背景均匀度 — Lab空间 a*/b* 通道方差
- L04: 背景色合规 — RGB值与标准色距离
"""

import numpy as np
import cv2
from typing import Dict, Tuple

from .models import RuleResult, Verdict


def _get_face_region(
    image: np.ndarray,
    face_info: Dict,
    alpha_matte: np.ndarray = None,
) -> np.ndarray:
    """提取面部区域像素"""
    h, w = image.shape[:2]
    bbox = face_info.get("bbox")

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # 扩展到包含更多面部区域
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return image[y1:y2, x1:x2], (x1, y1, x2, y2)
    else:
        # 回退: 图像中心区域
        cy, cx = h // 2, w // 2
        size = min(h, w) // 3
        y1 = max(0, cy - size)
        y2 = min(h, cy + size)
        x1 = max(0, cx - size)
        x2 = min(w, cx + size)
        return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def check_face_lighting_uniformity(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """L01: 面部光照均匀度 — 左右脸亮度差异"""
    face_region, _ = _get_face_region(image, face_info)

    if face_region.size == 0:
        return RuleResult(
            rule_id="L01", category="lighting", name="面部光照均匀",
            verdict=Verdict.WARN, score=50,
            detail="无法提取面部区域",
            hint=None, critical=False,
        )

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if face_region.ndim == 3 else face_region
    h, w = gray.shape

    # 分为左右两半
    mid = w // 2
    left_half = gray[:, :mid]
    right_half = gray[:, mid:]

    if left_half.size == 0 or right_half.size == 0:
        return RuleResult(
            rule_id="L01", category="lighting", name="面部光照均匀",
            verdict=Verdict.WARN, score=50,
            detail="面部区域过小",
            hint=None, critical=False,
        )

    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)

    # 计算左右亮度比（取 min/max 避免方向性）
    min_bright = min(left_mean, right_mean)
    max_bright = max(left_mean, right_mean)
    ratio = min_bright / max_bright if max_bright > 0 else 1.0

    threshold = standard.face_uniformity_min

    if ratio >= threshold:
        return RuleResult(
            rule_id="L01", category="lighting", name="面部光照均匀",
            verdict=Verdict.PASS, score=100,
            detail=f"左右脸亮度均匀 (比值: {ratio:.1%}, 阈值: {threshold:.0%})",
            hint=None, critical=False,
        )
    else:
        darker = "左侧" if left_mean < right_mean else "右侧"
        return RuleResult(
            rule_id="L01", category="lighting", name="面部光照均匀",
            verdict=Verdict.FAIL,
            score=max(0, int(ratio / threshold * 100)),
            detail=f"面部{darker}偏暗 (比值: {ratio:.1%}, 要求 ≥ {threshold:.0%})",
            hint="请调整光源位置，避免侧光造成的'阴阳脸'",
            critical=False,
        )


def check_shadow(
    image: np.ndarray,
    face_info: Dict,
    alpha_matte: np.ndarray = None,
    standard=None,
) -> RuleResult:
    """L02: 阴影检测 — 面部网格分析"""
    face_region, _ = _get_face_region(image, face_info)

    if face_region.size == 0:
        return RuleResult(
            rule_id="L02", category="lighting", name="面部阴影",
            verdict=Verdict.WARN, score=50,
            detail="无法提取面部区域",
            hint=None, critical=False,
        )

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if face_region.ndim == 3 else face_region
    h, w = gray.shape

    # 8×8 网格分块分析
    grid_size = 8
    block_h = max(1, h // grid_size)
    block_w = max(1, w // grid_size)

    global_mean = np.mean(gray)
    if global_mean < 1:
        return RuleResult(
            rule_id="L02", category="lighting", name="面部阴影",
            verdict=Verdict.PASS, score=100,
            detail="面部区域过暗，跳过阴影检测",
            hint=None, critical=False,
        )

    shadow_blocks = 0
    total_blocks = 0
    shadow_ratio = standard.shadow_l_ratio if standard else 0.85

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * block_h
            y2 = min(h, (i + 1) * block_h)
            x1 = j * block_w
            x2 = min(w, (j + 1) * block_w)

            if y2 <= y1 or x2 <= x1:
                continue

            block = gray[y1:y2, x1:x2]
            if block.size < 16:
                continue

            block_mean = np.mean(block)
            total_blocks += 1

            if block_mean < global_mean * shadow_ratio:
                shadow_blocks += 1

    shadow_pct = shadow_blocks / total_blocks if total_blocks > 0 else 0

    if shadow_pct < 0.25:
        return RuleResult(
            rule_id="L02", category="lighting", name="面部阴影",
            verdict=Verdict.PASS, score=100,
            detail=f"未检测到明显阴影 (阴影块: {shadow_pct:.0%})",
            hint=None, critical=False,
        )
    elif shadow_pct < 0.40:
        return RuleResult(
            rule_id="L02", category="lighting", name="面部阴影",
            verdict=Verdict.WARN, score=75,
            detail=f"面部有轻微阴影 ({shadow_pct:.0%} 区域)",
            hint="建议调整光源方向以减少面部阴影",
            critical=False,
        )
    else:
        return RuleResult(
            rule_id="L02", category="lighting", name="面部阴影",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - shadow_pct) * 100)),
            detail=f"面部存在明显阴影 ({shadow_pct:.0%} 区域)",
            hint="请调整光源位置，确保面部光照均匀无阴影",
            critical=False,
        )


def check_background_uniformity(
    image: np.ndarray,
    alpha_matte: np.ndarray,
    standard,
) -> RuleResult:
    """L03: 背景均匀度 — Lab 空间 a*/b* 通道方差分析"""
    if alpha_matte is None:
        return RuleResult(
            rule_id="L03", category="lighting", name="背景均匀度",
            verdict=Verdict.WARN, score=50,
            detail="无 alpha 遮罩，无法检测背景",
            hint=None, critical=False,
        )

    # 提取纯背景像素 (alpha < 0.05)
    bg_mask = alpha_matte < 0.05
    bg_pixels = image[bg_mask]

    if len(bg_pixels) < 500:
        return RuleResult(
            rule_id="L03", category="lighting", name="背景均匀度",
            verdict=Verdict.WARN, score=50,
            detail="背景像素不足",
            hint=None, critical=False,
        )

    # 转换到 Lab 空间
    # image 是 BGR，需要先转 RGB 再转 Lab (或用 cv2 直接 BGR2Lab)
    bg_bgr = bg_pixels.reshape(-1, 1, 3).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_bgr.astype(np.uint8), cv2.COLOR_BGR2Lab)
    bg_lab = bg_lab.reshape(-1, 3).astype(np.float32)

    a_channel = bg_lab[:, 1]
    b_channel = bg_lab[:, 2]

    var_a = np.var(a_channel)
    var_b = np.var(b_channel)
    combined_var = var_a + var_b

    # 映射到 0-100 评分
    var_threshold = 25.0
    uniformity_score = max(0, 100 - (combined_var / var_threshold * 100))

    threshold = standard.bg_uniformity_min

    if uniformity_score >= threshold:
        return RuleResult(
            rule_id="L03", category="lighting", name="背景均匀度",
            verdict=Verdict.PASS,
            score=int(uniformity_score),
            detail=f"背景均匀度良好 (评分: {uniformity_score:.0f}/100, 阈值: {threshold})",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="L03", category="lighting", name="背景均匀度",
            verdict=Verdict.FAIL,
            score=int(uniformity_score),
            detail=f"背景不够均匀 (评分: {uniformity_score:.0f}/100, 阈值: {threshold})",
            hint="请确保背景布平整无褶皱，光照均匀",
            critical=False,
        )


def check_background_color(
    image: np.ndarray,
    alpha_matte: np.ndarray,
    expected_color_bgr: Tuple[int, int, int] = None,
    standard=None,
) -> RuleResult:
    """L04: 背景色合规 — 检测背景色是否在标准范围内"""
    if alpha_matte is None:
        return RuleResult(
            rule_id="L04", category="lighting", name="背景色合规",
            verdict=Verdict.WARN, score=50,
            detail="无 alpha 遮罩，无法检测",
            hint=None, critical=False,
        )

    # 提取背景像素（alpha < 0.1 确保纯净背景区域）
    bg_mask = alpha_matte < 0.1
    bg_pixels = image[bg_mask]

    if len(bg_pixels) < 500:
        return RuleResult(
            rule_id="L04", category="lighting", name="背景色合规",
            verdict=Verdict.WARN, score=50,
            detail="背景像素不足",
            hint=None, critical=False,
        )

    # 计算背景平均 BGR
    bg_mean = np.mean(bg_pixels, axis=0)

    max_dist = standard.bg_color_max_distance if standard else 30

    if expected_color_bgr is not None:
        # 计算与目标背景色的欧氏距离
        expected = np.array(expected_color_bgr, dtype=np.float32)
        distance = np.linalg.norm(bg_mean - expected)

        if distance <= max_dist:
            return RuleResult(
                rule_id="L04", category="lighting", name="背景色合规",
                verdict=Verdict.PASS, score=100,
                detail=f"背景色符合标准 (距离: {distance:.1f}, 限值: {max_dist:.0f})",
                hint=None, critical=False,
            )
        else:
            return RuleResult(
                rule_id="L04", category="lighting", name="背景色合规",
                verdict=Verdict.FAIL,
                score=max(0, int((1 - (distance - max_dist) / max_dist) * 100)),
                detail=f"背景色偏差较大 (距离: {distance:.1f})",
                hint="请确认选择了正确的背景颜色",
                critical=False,
            )
    else:
        # 无目标色时，仅报告检测到的背景色
        b, g, r = bg_mean
        return RuleResult(
            rule_id="L04", category="lighting", name="背景色",
            verdict=Verdict.PASS, score=100,
            detail=f"检测到背景色 BGR({b:.0f}, {g:.0f}, {r:.0f})",
            hint=None, critical=False,
        )


def run_lighting_checks(
    image: np.ndarray,
    face_info: Dict,
    alpha_matte: np.ndarray,
    standard,
    expected_bg_color: Tuple[int, int, int] = None,
) -> list:
    """运行所有光照与色彩检测规则"""
    return [
        check_face_lighting_uniformity(image, face_info, standard),
        check_shadow(image, face_info, alpha_matte, standard),
        check_background_uniformity(image, alpha_matte, standard),
        check_background_color(image, alpha_matte, expected_bg_color, standard),
    ]
