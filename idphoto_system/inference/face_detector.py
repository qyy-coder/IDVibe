"""
人脸检测模块
================================
封装 HivisionIDPhotos 的 MTCNN 人脸检测，提供统一的检测接口。
后续可扩展 RetinaFace、SCRFD 等模型。

支持:
- 人脸边界框检测
- 关键点检测（5点: 双眼、鼻尖、嘴角）
- 人脸质量快速检测
"""

import sys
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# 将 HivisionIDPhotos 添加到 Python 路径
_HIVISION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "HivisionIDPhotos",
)
if _HIVISION_PATH not in sys.path:
    sys.path.insert(0, _HIVISION_PATH)


class FaceDetector:
    """
    人脸检测器 — 封装 HivisionIDPhotos 的人脸检测功能。

    用法:
        detector = FaceDetector()
        face = detector.detect(image)  # 返回人脸信息字典
    """

    def __init__(
        self,
        model_type: str = "mtcnn",
        min_face_size: int = 80,
        confidence_threshold: float = 0.9,
    ):
        """
        :param model_type: 检测模型类型 — "mtcnn" | "retinaface"
        :param min_face_size: 最小人脸尺寸（像素）
        :param confidence_threshold: 置信度阈值
        """
        self.model_type = model_type
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self._detector = None

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._detector is not None:
            return

        if self.model_type == "mtcnn":
            from hivision.creator.face_detector import detect_face_mtcnn
            self._detector = detect_face_mtcnn
        elif self.model_type == "retinaface":
            # 使用 RetinaFace（需要下载对应 ONNX 模型）
            from hivision.creator.retinaface.inference import RetinaFaceDetector
            self._detector = RetinaFaceDetector()
        else:
            raise ValueError(f"不支持的检测模型: {self.model_type}")

    def detect(self, image: np.ndarray) -> Dict:
        """
        检测图像中的人脸。

        :param image: BGR 格式 numpy 数组 (H, W, 3)
        :return: 人脸信息字典，包含:
            - bbox: (x1, y1, x2, y2) 边界框
            - landmarks: 5个关键点坐标 [(x,y), ...]
            - confidence: 检测置信度
            - quality: 人脸质量评分 (0-100)
        :raises FaceNotFoundError: 未检测到人脸时
        :raises MultiFaceError: 检测到多张人脸时
        """
        self._ensure_loaded()

        if self.model_type == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.model_type == "retinaface":
            return self._detect_retinaface(image)

    def _detect_mtcnn(self, image: np.ndarray) -> Dict:
        """使用 MTCNN 检测"""
        from hivision.creator.face_detector import detect_face_mtcnn

        result = detect_face_mtcnn(image)

        if result is None or len(result) == 0:
            raise FaceNotFoundError("未检测到人脸，请确保面部完整且光线充足")

        # 取置信度最高的人脸
        if isinstance(result, list):
            best = max(result, key=lambda x: x.get("confidence", 0))
        else:
            best = result

        # 检查是否有多张人脸
        if isinstance(result, list) and len(result) > 1:
            high_conf_faces = [
                f for f in result
                if f.get("confidence", 0) > self.confidence_threshold
            ]
            if len(high_conf_faces) > 1:
                raise MultiFaceError(
                    f"检测到 {len(high_conf_faces)} 张人脸，请确保画面中只有一人"
                )

        face_info = {
            "bbox": best.get("bbox", None),
            "landmarks": best.get("landmarks", None),
            "confidence": best.get("confidence", 0),
            "roll_angle": best.get("roll_angle", 0),
        }

        # 快速质量评估
        face_info["quality"] = self._quick_quality_check(image, face_info)

        return face_info

    def _detect_retinaface(self, image: np.ndarray) -> Dict:
        """使用 RetinaFace 检测"""
        faces = self._detector.detect(image)
        if not faces:
            raise FaceNotFoundError("未检测到人脸")
        if len(faces) > 1:
            raise MultiFaceError(f"检测到 {len(faces)} 张人脸")

        face = faces[0]
        face_info = {
            "bbox": face["bbox"],
            "landmarks": face.get("landmarks", None),
            "confidence": face.get("confidence", 0),
            "roll_angle": face.get("roll_angle", 0),
        }
        face_info["quality"] = self._quick_quality_check(image, face_info)
        return face_info

    def _quick_quality_check(self, image: np.ndarray, face: Dict) -> int:
        """
        快速人脸质量检查（P0 版本：基础检查）。
        P1 将扩展为完整的 25+ 项合规检测引擎。

        :return: 质量评分 0-100
        """
        score = 100
        h, w = image.shape[:2]
        bbox = face.get("bbox")

        if bbox is None:
            return 0

        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1

        # 1. 人脸尺寸检查: 至少占画面 10%
        face_area_ratio = (face_w * face_h) / (w * h)
        if face_area_ratio < 0.05:
            score -= 30
        elif face_area_ratio < 0.10:
            score -= 15

        # 2. 人脸位置检查: 不应太偏边缘
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        horizontal_offset = abs(center_x - w / 2) / w
        if horizontal_offset > 0.15:
            score -= 20
        elif horizontal_offset > 0.10:
            score -= 10

        # 3. 倾斜角度检查
        roll = face.get("roll_angle", 0)
        if abs(roll) > 10:
            score -= 25
        elif abs(roll) > 5:
            score -= 10

        return max(0, score)

    def detect_with_context(self, image: np.ndarray) -> Tuple[Dict, "ContextWrapper"]:
        """
        检测人脸并返回 HivisionIDPhotos 兼容的上下文对象。
        用于与现有 HivisionIDPhotos 流水线对接。
        """
        face = self.detect(image)

        ctx = ContextWrapper()
        ctx.face = face
        ctx.processing_image = image
        ctx.origin_image = image.copy()

        return face, ctx


class ContextWrapper:
    """HivisionIDPhotos Context 对象的轻量替代——避免强依赖其内部结构。"""
    def __init__(self):
        self.params = None
        self.face = None
        self.matting_image = None
        self.processing_image = None
        self.origin_image = None
        self.result = None


class FaceNotFoundError(Exception):
    """未检测到人脸"""
    pass


class MultiFaceError(Exception):
    """检测到多张人脸"""
    pass
