"""
人像抠图模块
================================
封装 HivisionIDPhotos 的人像抠图功能，提供统一的抠图接口。

支持多种抠图模型:
- MODNet (默认): 速度快，纯色背景效果好
- RMBG-1.4: 精度更高但速度较慢（复杂场景回退方案）

P0 阶段使用 HivisionIDPhotos 原始模型。
P1 阶段将加入 Guided Filter 边缘优化 + 自适应回退机制。
"""

import sys
import os
from typing import Tuple, Optional

import cv2
import numpy as np

# 将 HivisionIDPhotos 添加到 Python 路径
_HIVISION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "HivisionIDPhotos",
)
if _HIVISION_PATH not in sys.path:
    sys.path.insert(0, _HIVISION_PATH)


class HumanMatting:
    """
    人像抠图器 — 封装 HivisionIDPhotos 的抠图功能。

    用法:
        matting = HumanMatting()
        fg_rgba, alpha = matting.matte(image)  # 返回前景 RGBA + alpha遮罩
    """

    # 可用模型列表
    AVAILABLE_MODELS = [
        "modnet_photographic_portrait_matting",
        "hivision_modnet",
        "rmbg-1.4",
        "birefnet-v1-lite",
    ]

    def __init__(
        self,
        model_name: str = "modnet_photographic_portrait_matting",
        trimap_threshold: float = 0.5,
    ):
        """
        :param model_name: 抠图模型名称
        :param trimap_threshold: 二值化 alpha 遮罩的阈值
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"不支持的抠图模型: {model_name}。"
                f"可用: {self.AVAILABLE_MODELS}"
            )
        self.model_name = model_name
        self.trimap_threshold = trimap_threshold
        self._extractor = None

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._extractor is not None:
            return

        from hivision.creator.human_matting import extract_human
        from hivision.creator.choose_handler import choose_handler
        from hivision import IDCreator

        # 使用 HivisionIDPhotos 的 choose_handler 来选择模型
        creator = IDCreator()
        choose_handler(creator, self.model_name, "mtcnn")
        self._extractor = creator.matting_handler

    def matte(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对输入图像执行人像抠图。

        :param image: BGR 格式 numpy 数组 (H, W, 3)
        :return: (foreground_rgba, alpha_matte)
            - foreground_rgba: BGRA 格式的前景图像 (H, W, 4)
            - alpha_matte: alpha 遮罩 (H, W)，值域 [0, 1]
        """
        self._ensure_loaded()

        # 构造 HivisionIDPhotos 兼容的上下文
        from hivision.creator.context import Context, Params

        ctx = Context(Params(size=(413, 295)))
        ctx.processing_image = image.copy()
        ctx.origin_image = image.copy()

        # 执行抠图
        self._extractor(ctx)

        matting_image = ctx.matting_image  # BGRA, (H, W, 4)

        # 提取 alpha 遮罩
        if matting_image.shape[2] >= 4:
            alpha = matting_image[:, :, 3].astype(np.float32) / 255.0
        else:
            # 无 alpha 通道，生成全 1 遮罩
            alpha = np.ones(matting_image.shape[:2], dtype=np.float32)

        return matting_image, alpha

    def matte_fast(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """快速抠图（同 matte，P0 阶段为别名）。"""
        return self.matte(image)

    @staticmethod
    def create_trimap(alpha: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        从 alpha 遮罩生成 trimap。
        - 0: 确定背景
        - 128: 不确定区域
        - 255: 确定前景

        P1 阶段在 Guided Filter 中使用。
        """
        trimap = np.zeros_like(alpha, dtype=np.uint8)
        trimap[alpha > 1 - threshold] = 255
        trimap[(alpha > threshold) & (alpha < 1 - threshold)] = 128
        return trimap
