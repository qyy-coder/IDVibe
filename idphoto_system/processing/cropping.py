"""
智能裁剪模块
================================
基于人脸检测结果对证件照进行智能裁剪和定位。

P0 功能:
- 根据人脸位置自动裁剪到目标规格
- 确保头部在画面中的比例符合标准
- 人脸居中调整

P1 将加入合规检测引擎的完整规则。
"""

from typing import Tuple, Dict, Optional

import cv2
import numpy as np

from ..utils.image_utils import PhotoSpec


class PhotoCropper:
    """
    证件照裁剪器。

    用法:
        cropper = PhotoCropper()
        result = cropper.crop(image, face_info, spec)
    """

    def __init__(
        self,
        head_height_ratio: float = 0.50,       # 头部中心在画面高度的位置
        head_measure_ratio: float = 0.23,       # 头部面积占画面比例
        head_top_range: Tuple[float, float] = (0.10, 0.08),  # 头顶距离上沿比例
    ):
        self.head_height_ratio = head_height_ratio
        self.head_measure_ratio = head_measure_ratio
        self.head_top_range = head_top_range

    def crop(
        self,
        image: np.ndarray,
        face_info: Dict,
        spec: PhotoSpec,
    ) -> np.ndarray:
        """
        根据人脸位置裁剪图像到目标规格。

        :param image: 输入图像 (H, W, C)，可以是 BGR 或 BGRA
        :param face_info: 人脸检测结果字典，必须包含 "bbox"
        :param spec: 目标证件照规格
        :return: 裁剪并缩放到目标尺寸的图像
        """
        h, w = image.shape[:2]
        bbox = face_info.get("bbox")

        if bbox is None:
            # 无人脸信息时使用中心裁剪
            return self._center_crop(image, spec)

        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2

        # 目标宽高比
        target_ratio = spec.width / spec.height

        # 计算裁剪区域
        # 头部占画面比例约为 face_h / crop_h
        # 目标: face_h / crop_h = head_measure_ratio * (某种关系)
        # 简化: crop_h = face_h / 0.45 (使头部约45%画面高度)
        crop_h = face_h / 0.45
        crop_w = crop_h * target_ratio

        # 确保裁剪区域在图像内
        if crop_w > w:
            crop_w = w
            crop_h = crop_w / target_ratio
        if crop_h > h:
            crop_h = h
            crop_w = crop_h * target_ratio

        # 头部中心位置: 在裁剪区域的上部约 40-45% 处
        head_y_in_crop = crop_h * 0.42
        crop_y = face_cy - head_y_in_crop
        crop_x = face_cx - crop_w / 2

        # 边界约束
        crop_x = max(0, min(crop_x, w - crop_w))
        crop_y = max(0, min(crop_y, h - crop_h))

        # 执行裁剪
        x1_c = int(crop_x)
        y1_c = int(crop_y)
        x2_c = int(crop_x + crop_w)
        y2_c = int(crop_y + crop_h)

        cropped = image[y1_c:y2_c, x1_c:x2_c]

        # 缩放到目标尺寸
        result = cv2.resize(
            cropped,
            (spec.width, spec.height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        return result

    def _center_crop(self, image: np.ndarray, spec: PhotoSpec) -> np.ndarray:
        """无人脸信息时的中心裁剪回退方案。"""
        h, w = image.shape[:2]
        target_ratio = spec.width / spec.height

        if w / h > target_ratio:
            # 图像更宽，以高度为准
            new_w = int(h * target_ratio)
            x1 = (w - new_w) // 2
            cropped = image[:, x1:x1 + new_w]
        else:
            # 图像更高，以宽度为准
            new_h = int(w / target_ratio)
            y1 = (h - new_h) // 2
            cropped = image[y1:y1 + new_h, :]

        return cv2.resize(
            cropped,
            (spec.width, spec.height),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def adjust_head_position(
        self,
        image: np.ndarray,
        face_info: Dict,
    ) -> Dict:
        """
        分析人脸的裁剪参数，返回调整建议。
        P1 阶段将接入合规检测引擎。

        :return: 包含裁剪参数和建议的字典
        """
        h, w = image.shape[:2]
        bbox = face_info.get("bbox", None)

        if bbox is None:
            return {"status": "no_face", "crop_region": (0, 0, w, h)}

        x1, y1, x2, y2 = bbox
        face_h = y2 - y1
        head_ratio = face_h / h

        info = {
            "status": "ok",
            "face_height": face_h,
            "head_ratio": head_ratio,
            "head_ratio_ok": 0.35 <= head_ratio <= 0.55,
            "face_center_x": (x1 + x2) / 2 / w,
            "face_center_y": (y1 + y2) / 2 / h,
            "suggestion": None,
        }

        if head_ratio < 0.30:
            info["status"] = "too_small"
            info["suggestion"] = "人脸偏小，请靠近镜头"
        elif head_ratio > 0.60:
            info["status"] = "too_large"
            info["suggestion"] = "人脸偏大，请远离镜头"

        return info
