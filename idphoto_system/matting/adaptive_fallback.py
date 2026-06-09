"""
自适应回退机制 (Adaptive Fallback)
=====================================
三阶段级联抠图的第三阶段：质量判定与回退决策。

评估抠图质量，当置信度不足时触发:
- 高质量 (confidence ≥ 70): 使用引导滤波结果
- 中等质量 (40 ≤ confidence < 70): 混合原始与细化结果
- 低质量 (confidence < 40): 标记需回退重算

置信度评分基于:
- 不确定区域占比 (过渡区域越小越确定)
- 边缘梯度强度 (边缘越清晰越确定)
- 边缘连续性 (碎片越少越确定)
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import cv2


@dataclass
class MattingQuality:
    """抠图质量评估结果"""
    confidence: float              # 综合置信度 0-100
    tier: str                      # 质量等级: "high" | "medium" | "low"
    uncertain_ratio: float         # 不确定区域占比
    mean_edge_strength: float      # 平均边缘梯度强度
    edge_continuity: float         # 边缘连续性评分 0-100
    method: str                    # 推荐方法
    needs_fallback: bool           # 是否需要回退
    details: dict                  # 额外细节


class AdaptiveFallback:
    """
    自适应回退决策器。

    评估抠图质量，并根据置信度决定使用哪种方法。

    用法:
        fb = AdaptiveFallback()
        quality = fb.evaluate(coarse_alpha, refined_alpha)
        final_alpha = fb.decide(quality, coarse_alpha, refined_alpha, image)
    """

    def __init__(
        self,
        high_confidence_threshold: float = 70.0,
        medium_confidence_threshold: float = 40.0,
        max_uncertain_ratio: float = 0.15,
    ):
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.max_uncertain_ratio = max_uncertain_ratio

    def evaluate(
        self,
        coarse_alpha: np.ndarray,
        refined_alpha: np.ndarray,
    ) -> MattingQuality:
        """
        评估抠图质量。

        :param coarse_alpha: MODNet 粗抠图 alpha (H, W) float32 [0, 1]
        :param refined_alpha: 引导滤波细化后的 alpha (H, W)
        :return: MattingQuality 质量评估结果
        """
        h, w = coarse_alpha.shape

        # ----- 1. 不确定区域分析 -----
        # alpha 在 0.05-0.95 之间的过渡区域
        uncertain_mask = (coarse_alpha > 0.05) & (coarse_alpha < 0.95)
        uncertain_pixels = np.sum(uncertain_mask)
        total_pixels = h * w
        uncertain_ratio = uncertain_pixels / total_pixels

        # ----- 2. 边缘梯度分析 -----
        grad_x = cv2.Sobel(coarse_alpha.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(coarse_alpha.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 仅统计不确定区域内的梯度
        uncertain_grad = gradient_mag[uncertain_mask] if uncertain_pixels > 0 else gradient_mag
        mean_edge_strength = float(np.mean(uncertain_grad)) if uncertain_grad.size > 0 else 0

        # ----- 3. 边缘连续性分析 -----
        # 对细化后的 alpha 进行 Canny 边缘检测
        alpha_uint8 = (refined_alpha * 255).astype(np.uint8)
        edges = cv2.Canny(alpha_uint8, 50, 150)
        edge_pixels = np.sum(edges > 0) / total_pixels

        # 边缘连续性: 理想情况下边缘像素数应在合理范围
        # 太多 → 碎片化, 太少 → 过于平滑
        ideal_edge_ratio = 0.02  # 约 2% 的边缘像素
        edge_continuity = max(0, 100 - abs(edge_pixels - ideal_edge_ratio) / ideal_edge_ratio * 100)

        # ----- 4. 置信度计算 -----
        # 不确定区域越小 → 置信度越高
        area_penalty = uncertain_ratio * 80  # 最大扣 80 分

        # 边缘越强 → 置信度越高（到一定限度）
        edge_bonus = min(mean_edge_strength * 250, 25)  # 最大加 25 分

        # 边缘连续性加分
        continuity_bonus = edge_continuity * 0.15  # 最大加 15 分

        confidence = 100 - area_penalty + edge_bonus + continuity_bonus
        confidence = max(0.0, min(100.0, confidence))

        # ----- 5. 质量等级判定 -----
        if confidence >= self.high_threshold:
            tier = "high"
            method = "guided_filter"
            needs_fallback = False
        elif confidence >= self.medium_threshold:
            tier = "medium"
            method = "blend"
            needs_fallback = False
        else:
            tier = "low"
            method = "fallback"
            needs_fallback = True

        return MattingQuality(
            confidence=round(confidence, 2),
            tier=tier,
            uncertain_ratio=round(uncertain_ratio, 4),
            mean_edge_strength=round(mean_edge_strength, 4),
            edge_continuity=round(edge_continuity, 1),
            method=method,
            needs_fallback=needs_fallback,
            details={
                "image_size": f"{w}×{h}",
                "uncertain_pixels": int(uncertain_pixels),
                "edge_pixel_ratio": round(edge_pixels, 4),
            },
        )

    def decide(
        self,
        quality: MattingQuality,
        coarse_alpha: np.ndarray,
        refined_alpha: np.ndarray,
    ) -> np.ndarray:
        """
        根据质量评估决定最终 alpha。

        :param quality: 质量评估结果
        :param coarse_alpha: 粗抠图 alpha
        :param refined_alpha: 细化后的 alpha
        :return: 最终 alpha (H, W) float32 [0, 1]
        """
        if quality.tier == "high":
            # 高质量: 直接使用引导滤波结果
            return refined_alpha.copy()

        elif quality.tier == "medium":
            # 中等质量: 混合细化与原始结果
            blend_weight = (quality.confidence - self.medium_threshold) / (
                self.high_threshold - self.medium_threshold
            )
            blend_weight = np.clip(blend_weight, 0.0, 1.0)

            final = refined_alpha * blend_weight + coarse_alpha * (1 - blend_weight)
            return np.clip(final, 0.0, 1.0).astype(np.float32)

        else:
            # 低质量: 保留粗结果，标记需回退
            # 注意: 实际回退（如换用 RMBG-2.0）在外部流水线中处理
            return refined_alpha.copy()  # 返回细化结果，但标记了 needs_fallback

    def to_dict(self, quality: MattingQuality) -> dict:
        """序列化质量评估结果"""
        return {
            "confidence": quality.confidence,
            "tier": quality.tier,
            "uncertain_ratio": quality.uncertain_ratio,
            "mean_edge_strength": quality.mean_edge_strength,
            "edge_continuity": quality.edge_continuity,
            "method": quality.method,
            "needs_fallback": quality.needs_fallback,
            "details": quality.details,
        }


# ============================================================
# 便捷函数
# ============================================================

def cascade_refine(
    image: np.ndarray,
    coarse_alpha: np.ndarray,
    radius: int = 8,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, MattingQuality]:
    """
    完整的级联抠图优化流程。

    三阶段:
        1. MODNet 粗抠图 (外部提供 coarse_alpha)
        2. 引导滤波边缘优化
        3. 置信度评估与回退决策

    :param image: BGR 图像 (H, W, 3)
    :param coarse_alpha: MODNet 粗抠图 alpha (H, W)
    :param radius: 引导滤波半径
    :param eps: 引导滤波正则化参数
    :return: (最终 alpha, 质量报告)
    """
    from .edge_refiner import EdgeRefiner
    from .adaptive_fallback import AdaptiveFallback

    # Stage 2: 边缘精细化
    refiner = EdgeRefiner(radius=radius, eps=eps)
    refined_alpha = refiner.refine(image, coarse_alpha)

    # Stage 3: 质量评估与决策
    fallback = AdaptiveFallback()
    quality = fallback.evaluate(coarse_alpha, refined_alpha)
    final_alpha = fallback.decide(quality, coarse_alpha, refined_alpha)

    return final_alpha, quality
