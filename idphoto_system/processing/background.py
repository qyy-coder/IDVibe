"""
背景替换模块
================================
基于 HivisionIDPhotos 的背景合成功能，扩展为完整的证件照背景替换。

支持:
- 纯色背景 (白色/蓝色/红色)
- 上下渐变背景
- 中心渐变背景
- 自定义背景图
"""

import sys
import os
from typing import Tuple

import cv2
import numpy as np

_HIVISION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "HivisionIDPhotos",
)
if _HIVISION_PATH not in sys.path:
    sys.path.insert(0, _HIVISION_PATH)


class BackgroundReplacer:
    """
    背景替换器 — 封装 HivisionIDPhotos 的背景合成。

    用法:
        replacer = BackgroundReplacer()
        result = replacer.replace(fg_rgba, color="blue", mode="pure")
    """

    # 渲染模式
    MODE_PURE = "pure_color"
    MODE_GRADIENT_UP = "updown_gradient"
    MODE_GRADIENT_CENTER = "center_gradient"

    def __init__(self):
        pass

    def replace(
        self,
        foreground_rgba: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
        mode: str = "pure_color",
    ) -> np.ndarray:
        """
        替换背景。

        :param foreground_rgba: BGRA 格式前景图像 (H, W, 4)
        :param color: BGR 颜色元组，如 (255, 255, 255) 白色
        :param mode: 渲染模式
        :return: BGR 格式合成图像 (H, W, 3)
        """
        from hivision.utils import add_background

        result = add_background(foreground_rgba, bgr=color, mode=mode)
        return result.astype(np.uint8)

    def replace_with_image(
        self,
        foreground_rgba: np.ndarray,
        background_image: np.ndarray,
    ) -> np.ndarray:
        """
        使用自定义背景图替换。

        :param foreground_rgba: BGRA 格式前景图像
        :param background_image: BGR 格式背景图像（会自动缩放到匹配前景尺寸）
        :return: BGR 格式合成图像
        """
        from hivision.utils import add_background_with_image

        return add_background_with_image(foreground_rgba, background_image)

    @staticmethod
    def composite_manual(
        foreground_bgr: np.ndarray,
        alpha: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        手动合成（不依赖 HivisionIDPhotos）—— 用于 P1+ 自定义光照融合。

        :param foreground_bgr: BGR 格式前景 (H, W, 3)
        :param alpha: alpha 遮罩 (H, W)，值域 [0, 1]
        :param background_color: BGR 颜色
        :return: BGR 格式合成图像
        """
        h, w = foreground_bgr.shape[:2]
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        bg = np.ones((h, w, 3), dtype=np.float32)
        bg[:, :, 0] = background_color[0]
        bg[:, :, 1] = background_color[1]
        bg[:, :, 2] = background_color[2]

        fg = foreground_bgr.astype(np.float32)
        result = fg * alpha_3ch + bg * (1 - alpha_3ch)
        return result.clip(0, 255).astype(np.uint8)
