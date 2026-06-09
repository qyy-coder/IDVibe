"""
几何检测规则 (Geometric Checks)
=================================
检测证件照中人脸的几何位置是否符合标准。

规则:
- G01: 头部占比检测 — 头部高度占画面比例
- G02: 眼睛水平线位置 — 距画面上沿比例
- G03: 人脸居中偏离度 — 水平偏移
- G04: 下颚到画面下沿距离

依赖: 人脸 bbox + landmarks
"""

from typing import Dict, Tuple, Optional
import numpy as np

from .models import RuleResult, Verdict


def check_head_ratio(
    face_info: Dict,
    image_shape: Tuple[int, int],
    standard,
) -> RuleResult:
    """
    G01: 头部占比检测

    使用 MTCNN 的 5 点关键点估算头部高度:
    - 额头 y ≈ 左眼 + 右眼中点 - 眼睛到嘴角距离 × 0.8
    - 估算含发际线的头部顶部
    """
    h, w = image_shape[:2]
    bbox = face_info.get("bbox")
    landmarks = face_info.get("landmarks")

    if bbox is None:
        return RuleResult(
            rule_id="G01",
            category="geometric",
            name="头部占比",
            verdict=Verdict.FAIL,
            score=0,
            detail="无法检测人脸边界框",
            hint="请确保面部完整可见",
            critical=True,
        )

    x1, y1, x2, y2 = bbox
    face_h = y2 - y1

    # 使用关键点估算更精确的头部高度
    if landmarks is not None and len(landmarks) >= 5:
        # MTCNN 5点: 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        left_mouth = np.array(landmarks[3])
        right_mouth = np.array(landmarks[4])

        eyes_center = (left_eye + right_eye) / 2.0
        mouth_center = (left_mouth + right_mouth) / 2.0
        eye_mouth_dist = np.linalg.norm(eyes_center - mouth_center)

        # 额头估算: 眼睛上方约 0.9 × 眼嘴距离
        forehead_y = eyes_center[1] - eye_mouth_dist * 0.9
        # 下颚: bbox 底部（MTCNN bbox 通常到下巴）
        chin_y = y2
        # 发际线补偿: MTCNN bbox 不包含头发，真人头部约 12% 在 bbox 上方
        head_raw = chin_y - forehead_y
        top_y = forehead_y - head_raw * 0.12
        head_height = chin_y - top_y
    else:
        # 回退: 使用 bbox 高度
        head_height = face_h

    head_ratio = head_height / h
    min_r, max_r = standard.head_ratio_range

    if min_r <= head_ratio <= max_r:
        return RuleResult(
            rule_id="G01", category="geometric", name="头部占比",
            verdict=Verdict.PASS, score=100,
            detail=f"头部占比 {head_ratio:.1%} (标准: {min_r:.0%}-{max_r:.0%})",
            hint=None, critical=True,
        )
    elif head_ratio < min_r:
        return RuleResult(
            rule_id="G01", category="geometric", name="头部占比",
            verdict=Verdict.FAIL, score=max(0, int(head_ratio / min_r * 100)),
            detail=f"头部占比偏小 {head_ratio:.1%} (要求 ≥ {min_r:.0%})",
            hint="请靠近镜头或重新裁剪画面，使头部占画面 {:.0%}-{:.0%}".format(min_r, max_r),
            critical=True,
        )
    else:
        return RuleResult(
            rule_id="G01", category="geometric", name="头部占比",
            verdict=Verdict.FAIL, score=max(0, int(max_r / head_ratio * 100)),
            detail=f"头部占比偏大 {head_ratio:.1%} (要求 ≤ {max_r:.0%})",
            hint="请远离镜头，使头部占画面 {:.0%}-{:.0%}".format(min_r, max_r),
            critical=True,
        )


def check_eye_line(
    face_info: Dict,
    image_shape: Tuple[int, int],
    standard,
) -> RuleResult:
    """
    G02: 眼睛水平线位置

    检测双眼中心距画面上沿的比例。
    """
    h, w = image_shape[:2]
    landmarks = face_info.get("landmarks")

    if landmarks is None or len(landmarks) < 2:
        return RuleResult(
            rule_id="G02", category="geometric", name="眼睛位置",
            verdict=Verdict.WARN, score=50,
            detail="无关键点数据，无法检测眼睛位置",
            hint=None, critical=False,
        )

    # MTCNN: [左眼, 右眼, ...]
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_ratio = eye_center_y / h

    min_r, max_r = standard.eye_line_range

    if min_r <= eye_ratio <= max_r:
        return RuleResult(
            rule_id="G02", category="geometric", name="眼睛位置",
            verdict=Verdict.PASS, score=100,
            detail=f"眼睛位置 {eye_ratio:.1%} (标准: {min_r:.0%}-{max_r:.0%})",
            hint=None, critical=False,
        )
    elif eye_ratio < min_r:
        return RuleResult(
            rule_id="G02", category="geometric", name="眼睛位置",
            verdict=Verdict.FAIL, score=max(0, int(eye_ratio / min_r * 100)),
            detail=f"眼睛位置偏高 {eye_ratio:.1%} (要求 ≥ {min_r:.0%})",
            hint="请将头部下移，或重新裁剪使眼睛在画面上半部",
            critical=False,
        )
    else:
        return RuleResult(
            rule_id="G02", category="geometric", name="眼睛位置",
            verdict=Verdict.FAIL, score=max(0, int(max_r / eye_ratio * 100)),
            detail=f"眼睛位置偏低 {eye_ratio:.1%} (要求 ≤ {max_r:.0%})",
            hint="请将头部上移，或重新裁剪使眼睛在画面上半部",
            critical=False,
        )


def check_face_centering(
    face_info: Dict,
    image_shape: Tuple[int, int],
    standard,
) -> RuleResult:
    """
    G03: 人脸居中偏离度

    检测人脸中心与画面中轴的水平偏移。
    """
    h, w = image_shape[:2]
    bbox = face_info.get("bbox")

    if bbox is None:
        return RuleResult(
            rule_id="G03", category="geometric", name="人脸居中",
            verdict=Verdict.FAIL, score=0,
            detail="无法检测人脸",
            hint=None, critical=False,
        )

    x1, y1, x2, y2 = bbox
    face_center_x = (x1 + x2) / 2.0
    offset = abs(face_center_x - w / 2.0) / w

    max_offset = standard.face_center_max_offset

    if offset <= max_offset:
        return RuleResult(
            rule_id="G03", category="geometric", name="人脸居中",
            verdict=Verdict.PASS, score=100,
            detail=f"水平偏移 {offset:.1%} (限值: {max_offset:.0%})",
            hint=None, critical=False,
        )
    else:
        direction = "左" if face_center_x < w / 2 else "右"
        return RuleResult(
            rule_id="G03", category="geometric", name="人脸居中",
            verdict=Verdict.FAIL,
            score=max(0, int((1 - offset / max_offset) * 100)),
            detail=f"人脸偏{direction} {offset:.1%} (限值: {max_offset:.0%})",
            hint=f"请将头部向{'右' if direction == '左' else '左'}移动",
            critical=False,
        )


def check_chin_margin(
    face_info: Dict,
    image_shape: Tuple[int, int],
    standard,
) -> RuleResult:
    """
    G04: 下颚到画面下沿距离

    确保下颚距离画面底部有足够空间。
    """
    h, w = image_shape[:2]
    bbox = face_info.get("bbox")

    if bbox is None:
        return RuleResult(
            rule_id="G04", category="geometric", name="下颚位置",
            verdict=Verdict.FAIL, score=0,
            detail="无法检测人脸",
            hint=None, critical=False,
        )

    x1, y1, x2, y2 = bbox
    chin_to_bottom = (h - y2) / h

    min_margin = standard.chin_margin_min

    if chin_to_bottom >= min_margin:
        return RuleResult(
            rule_id="G04", category="geometric", name="下颚位置",
            verdict=Verdict.PASS, score=100,
            detail=f"下颚距下沿 {chin_to_bottom:.1%} (要求 ≥ {min_margin:.0%})",
            hint=None, critical=False,
        )
    else:
        return RuleResult(
            rule_id="G04", category="geometric", name="下颚位置",
            verdict=Verdict.FAIL,
            score=max(0, int(chin_to_bottom / min_margin * 100)),
            detail=f"下颚距下沿不足 {chin_to_bottom:.1%} (要求 ≥ {min_margin:.0%})",
            hint="请将头部上移，给下颚留出更多空间",
            critical=False,
        )


# ============================================================
# 批量运行
# ============================================================

def run_geometric_checks(
    face_info: Dict,
    image_shape: Tuple[int, int],
    standard,
) -> list:
    """运行所有几何检测规则，返回 RuleResult 列表"""
    return [
        check_head_ratio(face_info, image_shape, standard),
        check_eye_line(face_info, image_shape, standard),
        check_face_centering(face_info, image_shape, standard),
        check_chin_margin(face_info, image_shape, standard),
    ]
