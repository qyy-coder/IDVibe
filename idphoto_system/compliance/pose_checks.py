"""
姿态检测规则 (Pose Checks)
================================
使用 EPnP 算法从 2D 关键点解算 3D 头部姿态。

规则:
- P01: 偏航角 (Yaw) — |角度| < 阈值
- P02: 俯仰角 (Pitch) — |角度| < 阈值
- P03: 翻滚角 (Roll) — |角度| < 阈值

技术: OpenCV solvePnP + 通用 3D 人脸模型 → 欧拉角
"""

import numpy as np
import cv2
from typing import Dict, Tuple

from .models import RuleResult, Verdict


# 通用 3D 人脸模型关键点（归一化世界坐标）
# 参考: MediaPipe Face Mesh 定义的 6 个核心点
FACE_MODEL_3D = np.array([
    [0.0,  0.0,  0.0],       # 鼻尖
    [0.0, -0.35, 0.05],      # 下颚
    [-0.09, 0.05, 0.02],     # 左眼内角
    [0.09,  0.05, 0.02],     # 右眼内角
    [-0.09, -0.18, 0.01],    # 左嘴角
    [0.09, -0.18, 0.01],     # 右嘴角
], dtype=np.float32)


def _extract_model_points(landmarks) -> np.ndarray:
    """
    从 MTCNN 5 点关键点映射到 6 点 3D 模型。

    MTCNN 5点: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]

    我们构造 6 个对应点:
    [鼻尖, 下颚估计, 左眼, 右眼, 左嘴角, 右嘴角]
    其中下颚从 bbox 底部和鼻尖插值估算。
    """
    if landmarks is None or len(landmarks) < 5:
        return None

    pts = np.array(landmarks, dtype=np.float32)

    left_eye = pts[0]
    right_eye = pts[1]
    nose = pts[2]
    left_mouth = pts[3]
    right_mouth = pts[4]

    # 估算下颚: 在鼻子和嘴角下方
    mouth_center = (left_mouth + right_mouth) / 2.0
    eye_center = (left_eye + right_eye) / 2.0
    chin = mouth_center + (mouth_center - eye_center) * 0.5

    return np.array([
        nose,            # 0: 鼻尖
        chin,            # 1: 下颚(估算)
        left_eye,        # 2: 左眼
        right_eye,       # 3: 右眼
        left_mouth,      # 4: 左嘴角
        right_mouth,     # 5: 右嘴角
    ], dtype=np.float32)


def _rotation_vector_to_euler(rvec) -> Tuple[float, float, float]:
    """旋转向量 → 欧拉角 (yaw, pitch, roll) 单位: 度"""
    R, _ = cv2.Rodrigues(rvec)

    # 从旋转矩阵提取欧拉角
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[2, 1], R[2, 2])
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        roll = 0

    return (
        np.degrees(yaw),
        np.degrees(pitch),
        np.degrees(roll),
    )


def estimate_pose(
    face_info: Dict,
    image_shape: Tuple[int, int],
) -> Tuple[float, float, float]:
    """
    估计头部三维姿态。

    :return: (yaw, pitch, roll) 单位: 度
    """
    h, w = image_shape[:2]
    landmarks = face_info.get("landmarks")

    image_points = _extract_model_points(landmarks)
    if image_points is None:
        return (0, 0, 0)

    # 相机内参矩阵（近似，假设无畸变）
    focal = max(w, h)
    camera_matrix = np.array([
        [focal, 0, w / 2.0],
        [0, focal, h / 2.0],
        [0, 0, 1],
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    try:
        success, rvec, _ = cv2.solvePnP(
            FACE_MODEL_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return (0, 0, 0)

        return _rotation_vector_to_euler(rvec)
    except cv2.error:
        return (0, 0, 0)


def check_yaw(yaw: float, standard) -> RuleResult:
    """P01: 偏航角检测"""
    threshold = standard.yaw_max
    abs_yaw = abs(yaw)

    if abs_yaw <= threshold:
        return RuleResult(
            rule_id="P01", category="pose", name="偏航角 (Yaw)",
            verdict=Verdict.PASS, score=100,
            detail=f"偏航角 {yaw:.1f}° (限值: ±{threshold:.0f}°)",
            hint=None, critical=True,
        )
    else:
        direction = "右" if yaw > 0 else "左"
        return RuleResult(
            rule_id="P01", category="pose", name="偏航角 (Yaw)",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - (abs_yaw - threshold) / threshold) * 100)),
            detail=f"头部偏{direction} {abs_yaw:.1f}° (限值: ±{threshold:.0f}°)",
            hint=f"请将头部向{'左' if yaw > 0 else '右'}转约 {abs_yaw - threshold:.0f}°",
            critical=True,
        )


def check_pitch(pitch: float, standard) -> RuleResult:
    """P02: 俯仰角检测"""
    threshold = standard.pitch_max
    abs_pitch = abs(pitch)

    if abs_pitch <= threshold:
        return RuleResult(
            rule_id="P02", category="pose", name="俯仰角 (Pitch)",
            verdict=Verdict.PASS, score=100,
            detail=f"俯仰角 {pitch:.1f}° (限值: ±{threshold:.0f}°)",
            hint=None, critical=True,
        )
    else:
        hint = "请稍微抬起下巴" if pitch > 0 else "请稍微收低下巴"
        return RuleResult(
            rule_id="P02", category="pose", name="俯仰角 (Pitch)",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - (abs_pitch - threshold) / threshold) * 100)),
            detail=f"俯仰角 {pitch:.1f}° (限值: ±{threshold:.0f}°)",
            hint=hint, critical=True,
        )


def check_roll(roll: float, standard) -> RuleResult:
    """P03: 翻滚角检测"""
    threshold = standard.roll_max
    abs_roll = abs(roll)

    if abs_roll <= threshold:
        return RuleResult(
            rule_id="P03", category="pose", name="翻滚角 (Roll)",
            verdict=Verdict.PASS, score=100,
            detail=f"翻滚角 {roll:.1f}° (限值: ±{threshold:.0f}°)",
            hint=None, critical=True,
        )
    else:
        direction = "右" if roll > 0 else "左"
        return RuleResult(
            rule_id="P03", category="pose", name="翻滚角 (Roll)",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - (abs_roll - threshold) / threshold) * 100)),
            detail=f"头部倾斜 {abs_roll:.1f}° (限值: ±{threshold:.0f}°)",
            hint=f"请将头部向{'左' if roll > 0 else '右'}修正",
            critical=True,
        )


def run_pose_checks(
    face_info: Dict,
    image_shape: Tuple[int, int],
    standard,
) -> list:
    """运行所有姿态检测规则"""
    yaw, pitch, roll = estimate_pose(face_info, image_shape)
    return [
        check_yaw(yaw, standard),
        check_pitch(pitch, standard),
        check_roll(roll, standard),
    ]
