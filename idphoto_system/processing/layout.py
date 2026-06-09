"""
排版生成模块
================================
生成证件照冲印排版（多张照片排列在一张相纸上）。

标准排版:
- 一寸 × 8 张 (4×2)
- 二寸 × 4 张 (2×2)
- 自定义行列数

基于 HivisionIDPhotos 的 layout_calculator 模块。
"""

import sys
import os
from typing import List, Tuple

import cv2
import numpy as np

_HIVISION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "HivisionIDPhotos",
)
if _HIVISION_PATH not in sys.path:
    sys.path.insert(0, _HIVISION_PATH)

from ..utils.image_utils import PhotoSpec


class LayoutGenerator:
    """
    排版生成器。

    用法:
        gen = LayoutGenerator()
        layout = gen.generate(photo, spec, rows=4, cols=2)
    """

    # 标准排版配置
    STANDARD_LAYOUTS = {
        "一寸×8": {"rows": 4, "cols": 2, "spec": "一寸"},
        "一寸×4": {"rows": 2, "cols": 2, "spec": "一寸"},
        "二寸×4": {"rows": 2, "cols": 2, "spec": "二寸"},
        "二寸×2": {"rows": 1, "cols": 2, "spec": "二寸"},
        "大一寸×2": {"rows": 1, "cols": 2, "spec": "大一寸"},
    }

    def __init__(
        self,
        margin: int = 8,       # 照片间距（像素）
        border: int = 20,      # 页面边框（像素）
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.margin = margin
        self.border = border
        self.bg_color = bg_color

    def generate(
        self,
        photo: np.ndarray,
        spec: PhotoSpec,
        rows: int = 4,
        cols: int = 2,
    ) -> np.ndarray:
        """
        生成排版图像。

        :param photo: 单张证件照图像 (H, W, C)
        :param spec: 证件照规格
        :param rows: 行数
        :param cols: 列数
        :return: 排版图像 (BGR)
        """
        # 确保照片尺寸与规格匹配
        if photo.shape[1] != spec.width or photo.shape[0] != spec.height:
            photo = cv2.resize(photo, (spec.width, spec.height))

        # 计算画布尺寸
        canvas_w = cols * spec.width + (cols - 1) * self.margin + 2 * self.border
        canvas_h = rows * spec.height + (rows - 1) * self.margin + 2 * self.border

        # 创建白色背景
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :, 0] = self.bg_color[0]  # B
        canvas[:, :, 1] = self.bg_color[1]  # G
        canvas[:, :, 2] = self.bg_color[2]  # R

        # 放置照片
        for r in range(rows):
            for c in range(cols):
                x = self.border + c * (spec.width + self.margin)
                y = self.border + r * (spec.height + self.margin)

                # 处理不同通道数
                if photo.shape[2] == 4:
                    # RGBA 照片需要合成到白色背景上
                    b, g, r, a = cv2.split(photo)
                    alpha = a / 255.0
                    alpha_3ch = np.stack([alpha] * 3, axis=-1)
                    fg = cv2.merge([b, g, r]).astype(np.float32)
                    bg_region = canvas[y:y + spec.height, x:x + spec.width].astype(np.float32)
                    blended = fg * alpha_3ch + bg_region * (1 - alpha_3ch)
                    canvas[y:y + spec.height, x:x + spec.width] = blended.astype(np.uint8)
                else:
                    canvas[y:y + spec.height, x:x + spec.width] = photo

        return canvas

    def generate_standard(
        self,
        photo: np.ndarray,
        layout_name: str,
        spec: PhotoSpec,
    ) -> np.ndarray:
        """
        生成标准排版。

        :param photo: 证件照图像
        :param layout_name: 标准排版名称，如 "一寸×8"
        :param spec: 证件照规格
        """
        config = self.STANDARD_LAYOUTS.get(layout_name)
        if config is None:
            raise ValueError(
                f"未知的标准排版: '{layout_name}'。"
                f"可用: {list(self.STANDARD_LAYOUTS.keys())}"
            )

        return self.generate(
            photo,
            spec=spec,
            rows=config["rows"],
            cols=config["cols"],
        )

    @staticmethod
    def get_available_layouts(for_spec: str = None) -> List[str]:
        """获取可用的排版方案。"""
        if for_spec is None:
            return list(LayoutGenerator.STANDARD_LAYOUTS.keys())
        return [
            name for name, cfg in LayoutGenerator.STANDARD_LAYOUTS.items()
            if cfg["spec"] == for_spec
        ]
