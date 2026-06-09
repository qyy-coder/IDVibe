"""
面部状态检测规则 (Facial State Checks)
========================================
检测面部特征状态，确保符合证件照要求。

规则:
- F01: 眼睛开合度 — 眼睛纵横比 (EAR)
- F02: 嘴部开合度 — 嘴巴纵横比 (MAR)
- F03: 表情检测 — 基于 MAR 的微笑检测
- F04: 眼镜检测 — 鼻梁区域边缘密度
- F05: 眼镜反光检测 — 眼部附近亮斑
- F06: 红眼检测 — 眼部红色通道分析
"""

import numpy as np
import cv2
from typing import Dict, Tuple

from .models import RuleResult, Verdict


def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    计算眼睛纵横比 (Eye Aspect Ratio, EAR)。

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    对 MTCNN 只有单点，我们基于眼睛中心和上下估计。
    简化版: 对于只有单眼中心点的情况，使用 bbox 验证
    """
    # 当只有单点时，无法计算精确 EAR
    if len(eye_points) < 4:
        return 1.0  # 假设眼睛睁开
    p = eye_points
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
    return vertical / horizontal if horizontal > 0 else 0


def _estimate_ear_from_mtcnn(
    image: np.ndarray, face_info: Dict
) -> Tuple[float, float]:
    """
    从 MTCNN 5 点 + 图像分析估算双眼 EAR。

    MTCNN 提供左右眼中心点。我们分析眼睛区域的
    垂直梯度来估算开合状态。

    返回: (left_ear, right_ear)
    """
    landmarks = face_info.get("landmarks")
    if landmarks is None or len(landmarks) < 2:
        return (1.0, 1.0)

    left_eye_pt = np.array(landmarks[0], dtype=np.int32)
    right_eye_pt = np.array(landmarks[1], dtype=np.int32)

    # 估算眼睛区域窗口
    h, w = image.shape[:2]
    eye_window = max(15, int(min(h, w) * 0.03))

    results = []
    for pt in [left_eye_pt, right_eye_pt]:
        x1 = max(0, pt[0] - eye_window)
        y1 = max(0, pt[1] - eye_window // 2)
        x2 = min(w, pt[0] + eye_window)
        y2 = min(h, pt[1] + eye_window // 2)

        if x2 <= x1 or y2 <= y1:
            results.append(1.0)
            continue

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            results.append(1.0)
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # 垂直梯度：眼睛睁开时垂直方向梯度大
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vert_activity = np.mean(np.abs(grad_y))

        # 归一化为 0-1 的 EAR 估算值
        # 经验: vert_activity < 5 → 闭眼, > 20 → 睁眼
        ear = min(1.0, max(0.0, (vert_activity - 3) / 20.0))
        results.append(ear)

    return (results[0], results[1])


def check_eyes_open(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """F01: 眼睛开合度检测"""
    left_ear, right_ear = _estimate_ear_from_mtcnn(image, face_info)
    min_ear = min(left_ear, right_ear)
    threshold = standard.ear_min

    if min_ear >= threshold:
        return RuleResult(
            rule_id="F01", category="facial", name="眼睛开合",
            verdict=Verdict.PASS, score=100,
            detail=f"双眼睛开合度正常 (L:{left_ear:.2f}, R:{right_ear:.2f}, 阈值: {threshold:.2f})",
            hint=None, critical=True,
        )
    else:
        closed_eye = "左眼" if left_ear < right_ear else "右眼"
        return RuleResult(
            rule_id="F01", category="facial", name="眼睛开合",
            verdict=Verdict.FAIL,
            score=max(0, int(min_ear / threshold * 100)),
            detail=f"{closed_eye}可能闭合 (L:{left_ear:.2f}, R:{right_ear:.2f})",
            hint="请确保双眼自然睁开，正视镜头",
            critical=True,
        )


def _estimate_mar_from_mtcnn(
    image: np.ndarray, face_info: Dict
) -> float:
    """
    从 MTCNN 关键点估算嘴巴纵横比 (Mouth Aspect Ratio, MAR)。

    MAR = 嘴唇高度 / 嘴唇宽度
    使用左右嘴角 + 鼻尖位置估算嘴部区域。
    """
    landmarks = face_info.get("landmarks")
    if landmarks is None or len(landmarks) < 5:
        return 0.0

    left_mouth = np.array(landmarks[3])
    right_mouth = np.array(landmarks[4])
    mouth_width = np.linalg.norm(right_mouth - left_mouth)

    if mouth_width < 1:
        return 0.0

    # 在嘴角中点周围分析嘴部区域
    mouth_center = (left_mouth + right_mouth) / 2.0
    h, w = image.shape[:2]
    region_h = int(mouth_width * 0.5)
    region_w = int(mouth_width * 0.8)

    x1 = max(0, int(mouth_center[0] - region_w // 2))
    y1 = max(0, int(mouth_center[1] - region_h // 2))
    x2 = min(w, int(mouth_center[0] + region_w // 2))
    y2 = min(h, int(mouth_center[1] + region_h // 2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    # 嘴唇区域通常比周围暗
    # 开口时，口腔内部更暗 → 垂直方向暗区占比大
    mean_val = np.mean(gray)

    # 计算暗像素比例（口腔区域）
    dark_ratio = np.mean(gray < (mean_val * 0.7))

    # MAR 估算: 暗像素越多 → 嘴可能张开越大
    mar = dark_ratio * 3.0  # 缩放因子
    return min(1.0, mar)


def check_mouth_closed(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """F02: 嘴部闭合检测"""
    mar = _estimate_mar_from_mtcnn(image, face_info)
    threshold = standard.mar_max

    if mar <= threshold:
        return RuleResult(
            rule_id="F02", category="facial", name="嘴部闭合",
            verdict=Verdict.PASS, score=100,
            detail=f"嘴部闭合度正常 (MAR: {mar:.2f}, 阈值: {threshold:.2f})",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="F02", category="facial", name="嘴部闭合",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - (mar - threshold)) * 100)),
            detail=f"嘴部可能未闭合 (MAR: {mar:.2f}, 阈值: {threshold:.2f})",
            hint="请闭合嘴唇，保持中性表情",
            critical=False,
        )


def check_expression(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """F03: 表情检测（微笑/大笑）"""
    mar = _estimate_mar_from_mtcnn(image, face_info)
    threshold = standard.mar_smile_max

    if mar <= threshold:
        return RuleResult(
            rule_id="F03", category="facial", name="表情",
            verdict=Verdict.PASS, score=100,
            detail=f"表情正常 (MAR: {mar:.2f}, 微笑阈值: {threshold:.2f})",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="F03", category="facial", name="表情",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - (mar - threshold)) * 100)),
            detail=f"可能检测到微笑/张嘴 (MAR: {mar:.2f})",
            hint="请保持中性表情，不要微笑",
            critical=False,
        )


def check_glasses(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """F04: 眼镜检测"""
    landmarks = face_info.get("landmarks")
    if landmarks is None or len(landmarks) < 3:
        return RuleResult(
            rule_id="F04", category="facial", name="眼镜",
            verdict=Verdict.SKIP, score=100,
            detail="无关键点数据，跳过",
            hint=None, critical=False,
        )

    # 分析鼻梁区域（两眼之间偏下）的边缘密度
    h, w = image.shape[:2]
    left_eye = np.array(landmarks[0], dtype=np.int32)
    right_eye = np.array(landmarks[1], dtype=np.int32)

    bridge_center = (left_eye + right_eye) // 2
    bridge_center[1] += int(np.linalg.norm(right_eye - left_eye) * 0.3)

    box_size = int(np.linalg.norm(right_eye - left_eye) * 0.5)
    x1 = max(0, bridge_center[0] - box_size // 2)
    y1 = max(0, bridge_center[1] - box_size // 4)
    x2 = min(w, bridge_center[0] + box_size // 2)
    y2 = min(h, bridge_center[1] + box_size // 4)

    if x2 <= x1 or y2 <= y1:
        return RuleResult(
            rule_id="F04", category="facial", name="眼镜",
            verdict=Verdict.WARN, score=50, detail="鼻梁区域无效",
            hint=None, critical=False,
        )

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    # Sobel 边缘检测
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_density = np.mean(np.abs(grad_x))

    # 经验阈值: edge_density > 15 可能戴眼镜
    has_glasses = edge_density > 15

    if not has_glasses:
        return RuleResult(
            rule_id="F04", category="facial", name="眼镜",
            verdict=Verdict.PASS, score=100,
            detail="未检测到眼镜",
            hint=None, critical=False,
        )
    elif standard.glasses_allowed:
        return RuleResult(
            rule_id="F04", category="facial", name="眼镜",
            verdict=Verdict.PASS, score=80,
            detail="检测到眼镜，该标准允许佩戴",
            hint="请确保镜片无反光、镜框不遮挡眼睛",
            critical=False,
        )
    else:
        return RuleResult(
            rule_id="F04", category="facial", name="眼镜",
            verdict=Verdict.FAIL, score=30,
            detail="检测到眼镜，该标准要求不佩戴眼镜",
            hint="请摘下眼镜后重新拍摄",
            critical=False,
        )


def check_glare(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """F05: 眼镜反光检测"""
    landmarks = face_info.get("landmarks")
    if landmarks is None or len(landmarks) < 2:
        return RuleResult(
            rule_id="F05", category="facial", name="镜片反光",
            verdict=Verdict.SKIP, score=100,
            detail="无关键点数据，跳过", hint=None, critical=False,
        )

    h, w = image.shape[:2]
    left_eye = np.array(landmarks[0], dtype=np.int32)
    right_eye = np.array(landmarks[1], dtype=np.int32)

    # 分析每只眼睛周围
    max_glare = 0
    for eye_pt in [left_eye, right_eye]:
        radius = max(12, int(min(h, w) * 0.02))
        x1 = max(0, eye_pt[0] - radius)
        y1 = max(0, eye_pt[1] - radius)
        x2 = min(w, eye_pt[0] + radius)
        y2 = min(h, eye_pt[1] + radius)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # 亮斑检测: 局部亮度 > 240 且与周围形成高对比度
        bright_mask = gray > standard.glare_max_brightness
        bright_ratio = np.mean(bright_mask)
        max_glare = max(max_glare, bright_ratio)

    if max_glare < 0.03:  # 少于3%像素过亮
        return RuleResult(
            rule_id="F05", category="facial", name="镜片反光",
            verdict=Verdict.PASS, score=100,
            detail="未检测到明显镜片反光",
            hint=None, critical=False,
        )
    elif max_glare < 0.08:
        return RuleResult(
            rule_id="F05", category="facial", name="镜片反光",
            verdict=Verdict.WARN, score=70,
            detail="眼部附近有轻微反光",
            hint="建议调整光源位置或角度以减少反光",
            critical=False,
        )
    else:
        return RuleResult(
            rule_id="F05", category="facial", name="镜片反光",
            verdict=Verdict.FAIL, score=30,
            detail="眼部附近有较强反光",
            hint="请调整光源位置，确保镜片无反光",
            critical=False,
        )


def check_red_eye(
    image: np.ndarray, face_info: Dict, standard
) -> RuleResult:
    """F06: 红眼检测"""
    landmarks = face_info.get("landmarks")
    if landmarks is None or len(landmarks) < 2:
        return RuleResult(
            rule_id="F06", category="facial", name="红眼",
            verdict=Verdict.SKIP, score=100,
            detail="无关键点数据，跳过", hint=None, critical=False,
        )

    h, w = image.shape[:2]
    max_red_ratio = 0

    for eye_pt in [np.array(landmarks[0], dtype=np.int32),
                    np.array(landmarks[1], dtype=np.int32)]:
        radius = max(8, int(min(h, w) * 0.015))
        x1 = max(0, eye_pt[0] - radius)
        y1 = max(0, eye_pt[1] - radius)
        x2 = min(w, eye_pt[0] + radius)
        y2 = min(h, eye_pt[1] + radius)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = image[y1:y2, x1:x2]
        if roi.ndim < 3:
            continue

        # 红色通道分析
        b, g, r = cv2.split(roi.astype(np.float32))
        # 避免除零
        gb_sum = g + b + 1
        red_ratio = np.mean(r / gb_sum)
        max_red_ratio = max(max_red_ratio, red_ratio)

    threshold = standard.red_eye_max_ratio

    if max_red_ratio < threshold:
        return RuleResult(
            rule_id="F06", category="facial", name="红眼",
            verdict=Verdict.PASS, score=100,
            detail=f"红眼比正常 (R/(G+B): {max_red_ratio:.2f}, 阈值: {threshold:.2f})",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="F06", category="facial", name="红眼",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - (max_red_ratio - threshold)) * 100)),
            detail=f"可能检测到红眼 (R/(G+B): {max_red_ratio:.2f})",
            hint="请避免使用闪光灯直射，或使用红眼修复功能",
            critical=False,
        )


def run_facial_checks(
    image: np.ndarray,
    face_info: Dict,
    standard,
) -> list:
    """运行所有面部状态检测规则"""
    return [
        check_eyes_open(image, face_info, standard),
        check_mouth_closed(image, face_info, standard),
        check_expression(image, face_info, standard),
        check_glasses(image, face_info, standard),
        check_glare(image, face_info, standard),
        check_red_eye(image, face_info, standard),
    ]
