"""
边缘精细化器 (Edge Refiner)
=============================
三阶段级联抠图的第二阶段：基于引导滤波的边缘优化。

流程:
    1. 生成 trimap (确定前景/不确定区域/确定背景)
    2. 对不确定区域应用引导滤波
    3. 混合细化结果与原始 alpha
"""

import numpy as np
import cv2

from .guided_filter import guided_filter, fast_guided_filter


class TrimapGenerator:
    """
    Trimap 生成器 — 从 alpha 遮罩生成三值图。

    Trimap:
        - 0: 确定背景 (alpha < low_threshold)
        - 128: 不确定区域 (low_threshold <= alpha <= high_threshold)
        - 255: 确定前景 (alpha > high_threshold)
    """

    def __init__(
        self,
        low_threshold: float = 0.05,
        high_threshold: float = 0.95,
        dilate_radius: int = 5,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.dilate_radius = dilate_radius

    def generate(self, alpha: np.ndarray) -> np.ndarray:
        """
        从 alpha 遮罩生成 trimap。

        :param alpha: (H, W) float32 [0, 1]
        :return: (H, W) uint8 {0, 128, 255}
        """
        h, w = alpha.shape
        trimap = np.zeros((h, w), dtype=np.uint8)

        # 确定前景 (alpha > high_threshold)
        fg_mask = alpha > self.high_threshold
        trimap[fg_mask] = 255

        # 确定背景 (alpha < low_threshold)
        bg_mask = alpha < self.low_threshold
        # 已默认为 0

        # 不确定区域：对前景-背景边界进行形态学膨胀
        boundary = (alpha > self.low_threshold) & (alpha < self.high_threshold)

        # 也对确定前景的边界进行膨胀，捕获发丝过渡区
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.dilate_radius * 2 + 1, self.dilate_radius * 2 + 1),
        )
        fg_boundary = cv2.dilate(fg_mask.astype(np.uint8), kernel) & (~fg_mask)
        expanded_uncertain = boundary | (fg_boundary > 0)

        trimap[expanded_uncertain] = 128

        return trimap


class EdgeRefiner:
    """
    边缘精细化器 — 三阶段级联抠图的第二阶段。

    对 MODNet 粗抠图结果的边缘区域进行引导滤波优化，
    恢复发丝等细节，消除边缘锯齿。

    用法:
        refiner = EdgeRefiner(radius=8, eps=1e-6)
        refined_alpha = refiner.refine(image, coarse_alpha)
    """

    def __init__(
        self,
        radius: int = 8,
        eps: float = 1e-6,
        use_fast: bool = True,
        fast_subsample: int = 2,
    ):
        """
        :param radius: 引导滤波窗口半径
        :param eps: 正则化参数（越大平滑越强）
        :param use_fast: 是否使用快速引导滤波
        :param fast_subsample: 快速模式的下采样比例
        """
        self.radius = radius
        self.eps = eps
        self.use_fast = use_fast
        self.fast_subsample = fast_subsample
        self.trimap_gen = TrimapGenerator()

    def refine(
        self,
        image: np.ndarray,
        coarse_alpha: np.ndarray,
        trimap: np.ndarray = None,
    ) -> np.ndarray:
        """
        边缘精细化。

        :param image: BGR 图像 (H, W, 3)
        :param coarse_alpha: 粗抠图 alpha (H, W) float32 [0, 1]
        :param trimap: 可选的预生成 trimap
        :return: 细化后的 alpha (H, W) float32 [0, 1]
        """
        h, w = coarse_alpha.shape[:2]

        # 1. 生成引导图（原始图像灰度化）
        if image.ndim == 3:
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            guide = image

        # 归一化引导图
        guide_f = guide.astype(np.float32) / 255.0

        # 2. 生成 trimap
        if trimap is None:
            trimap = self.trimap_gen.generate(coarse_alpha)

        # 3. 对不确定区域 (trimap == 128) 应用引导滤波
        uncertain_mask = trimap == 128
        uncertain_ratio = np.mean(uncertain_mask)

        if uncertain_ratio < 0.001:
            # 不确定区域极小，无需细化
            return coarse_alpha.copy()

        # 4. 创建引导滤波的输入信号
        # 使用粗 alpha 作为信号
        src = coarse_alpha.astype(np.float32)

        # 5. 执行引导滤波
        if self.use_fast and min(h, w) > 200:
            refined = fast_guided_filter(
                guide,
                src,
                radius=self.radius,
                eps=self.eps,
                subsample=self.fast_subsample,
            )
        else:
            refined = guided_filter(
                guide,
                src,
                radius=self.radius,
                eps=self.eps,
            )

        # 6. 混合: 不确定区域用细化结果，确定区域保留原始
        fg_mask = trimap == 255
        bg_mask = trimap == 0

        result = coarse_alpha.copy()

        # 不确定区域: 使用滤波结果
        result[uncertain_mask] = refined[uncertain_mask]

        # 确保确定区域的约束不被打破
        result[fg_mask] = np.maximum(result[fg_mask], 0.95)
        result[bg_mask] = np.minimum(result[bg_mask], 0.05)

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def refine_with_detail_preservation(
        self,
        image: np.ndarray,
        coarse_alpha: np.ndarray,
        detail_strength: float = 0.3,
    ) -> np.ndarray:
        """
        保留细节的边缘优化 — 使用引导滤波的细节增强变体。

        通过调整 eps 参数控制细节保留程度:
        - eps 小 (< 1e-5): 强细节保留 (发丝清晰，但可能引入噪声)
        - eps 中 (1e-4~1e-5): 平衡
        - eps 大 (> 1e-4): 强平滑 (边缘柔和，但可能丢失细节)

        :param detail_strength: 细节保留强度 [0, 1]，越大保留越强
        """
        # 映射 detail_strength 到 eps
        # strength 0 → eps 1e-3 (强平滑)
        # strength 1 → eps 1e-7 (强细节保留)
        eps = 10 ** (-3 - 4 * detail_strength)

        # 临时覆盖 eps 参数
        old_eps = self.eps
        self.eps = eps
        try:
            return self.refine(image, coarse_alpha)
        finally:
            self.eps = old_eps
