"""
智能合规检测引擎 (Smart Compliance Engine) — 轻量版
====================================================
复用 ComplianceEngine 的检测函数，用宽松阈值 + 精简规则集运行。
只对真正严重的问题报警。面向前端 API 响应设计。

重构要点: 不再内联重写检测逻辑，改为组合调用已有检测模块。
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict

from .models import RuleResult, Verdict
from .standards import get_standard

# ── 导入全部检测函数 ──
from .geometric_checks import check_head_ratio, check_face_centering
from .pose_checks import check_roll
from .lighting_checks import check_face_lighting_uniformity, check_background_uniformity
from .quality_checks import check_sharpness


# ============================================================
# 数据模型（保持原有对外接口不变）
# ============================================================

@dataclass
class SmartCheck:
    """智能检测结果"""
    name: str
    icon: str
    status: str        # ok / warn / issue
    detail: str
    suggestion: str = ""


@dataclass
class SmartReport:
    """智能合规报告"""
    checks: List[SmartCheck] = field(default_factory=list)
    total_time: float = 0

    @property
    def ok_count(self): return sum(1 for c in self.checks if c.status == 'ok')
    @property
    def warn_count(self): return sum(1 for c in self.checks if c.status == 'warn')
    @property
    def issue_count(self): return sum(1 for c in self.checks if c.status == 'issue')
    @property
    def is_good(self): return self.issue_count == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok_count, "warn": self.warn_count, "issue": self.issue_count,
            "is_good": self.is_good, "total_time": round(self.total_time, 3),
            "checks": [{"name": c.name, "icon": c.icon, "status": c.status,
                        "detail": c.detail, "suggestion": c.suggestion} for c in self.checks]
        }


# ============================================================
# RuleResult → SmartCheck 转换
# ============================================================

# 图标映射
_ICON_MAP = {
    Verdict.PASS: "✅",
    Verdict.WARN: "💡",
    Verdict.FAIL: "⚠️",
    Verdict.SKIP: "⏭️",
}

# 状态映射
_STATUS_MAP = {
    Verdict.PASS: "ok",
    Verdict.WARN: "warn",
    Verdict.FAIL: "issue",
    Verdict.SKIP: "ok",
}


def _to_smart_check(rule: RuleResult, name_override: str = None) -> SmartCheck:
    """将 RuleResult 转为 SmartCheck"""
    status = _STATUS_MAP.get(rule.verdict, "ok")
    # 仅失败且 critical 的标为 issue
    if rule.verdict == Verdict.FAIL and not rule.critical:
        status = "warn"
    return SmartCheck(
        name=name_override or rule.name,
        icon=_ICON_MAP.get(rule.verdict, "✅"),
        status=status,
        detail=rule.detail if rule.detail else "",
        suggestion=rule.hint if rule.hint and status != "ok" else "",
    )


# ============================================================
# 次要检查: 不依赖 ComplianceEngine 检查函数的辅助项
# ============================================================

def _check_brightness(gray: np.ndarray) -> SmartCheck:
    """照片过曝/过暗（纯 OpenCV 辅助检测）"""
    mean_bright = gray.mean()
    if mean_bright < 50:
        return SmartCheck("照片亮度", "⚠️", "issue",
                          f"照片过暗 ({mean_bright:.0f})",
                          "请增加环境光线或使用闪光灯")
    elif mean_bright > 230:
        return SmartCheck("照片亮度", "⚠️", "issue",
                          f"照片过曝 ({mean_bright:.0f})",
                          "请避免直射强光，降低曝光")
    elif mean_bright < 80:
        return SmartCheck("照片亮度", "💡", "warn",
                          f"照片偏暗 ({mean_bright:.0f})",
                          "建议增加环境光线")
    else:
        return SmartCheck("照片亮度", "✅", "ok",
                          f"亮度正常 ({mean_bright:.0f})")


def _check_face_presence(face_info: dict) -> SmartCheck | None:
    """人脸存在检测 — 返回 None 表示通过，返回 SmartCheck 表示失败"""
    rect = face_info.get("rectangle")
    bbox = face_info.get("bbox")
    if (not rect or len(rect) != 4) and not bbox:
        return SmartCheck("人脸检测", "❌", "issue",
                          "未检测到人脸", "请确保面部完整、光线充足")
    return None


def _hivision_to_compliance_format(face_info: dict, h: int, w: int) -> dict:
    """将 HivisionIDPhotos 格式的 face_info 转为 ComplianceEngine 格式"""
    bbox = face_info.get("bbox")
    rect = face_info.get("rectangle")

    if bbox is None and rect and len(rect) == 4:
        cx, cy, fw, fh = rect
        bbox = (int(cx - fw/2), int(cy - fh/2), int(cx + fw/2), int(cy + fh/2))

    result = {"bbox": bbox, "landmarks": face_info.get("landmarks"), "confidence": 1.0}
    # 传递 roll_angle (HivisionIDPhotos 可能提供)
    result["roll_angle"] = face_info.get("roll_angle", 0)
    return result


# ============================================================
# SmartComplianceEngine — 重构版
# ============================================================

class SmartComplianceEngine:
    """智能合规检测——复用 ComplianceEngine 检测函数，仅跑 8 项核心规则"""

    def __init__(self):
        self._profile = get_standard("quick_check")

    def check(self,
              image: np.ndarray,
              face_info: dict,
              alpha: np.ndarray = None) -> SmartReport:
        """
        执行智能合规检测。

        :param image: BGR 图像 (H, W, 3)
        :param face_info: HivisionIDPhotos 格式的人脸信息
        :param alpha: alpha 遮罩 (H, W) float32 [0, 1]
        """
        t0 = time.time()
        checks: List[SmartCheck] = []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # ── 0. 人脸存在 ──
        presence_fail = _check_face_presence(face_info)
        if presence_fail:
            checks.append(presence_fail)
            return SmartReport(checks=checks, total_time=time.time() - t0)

        # ── 格式转换 ──
        cf = _hivision_to_compliance_format(face_info, h, w)

        # ── 1. 头部大小 (G01) ──
        r = check_head_ratio(cf, (h, w), self._profile)
        checks.append(_to_smart_check(r, "头部大小"))

        # ── 2. 人脸居中 (G03) ──
        r = check_face_centering(cf, (h, w), self._profile)
        checks.append(_to_smart_check(r, "人脸居中"))

        # ── 3. 头部倾斜 (P03 — 用 HivisionIDPhotos roll_angle) ──
        roll = face_info.get("roll_angle", 0)
        r = check_roll(roll, self._profile)
        checks.append(_to_smart_check(r, "头部倾斜"))

        # ── 4. 面部光照均匀度 (L01) ──
        r = check_face_lighting_uniformity(image, cf, self._profile)
        checks.append(_to_smart_check(r, "面部光照"))

        # ── 5. 背景纯净度 (L03) ──
        if alpha is not None:
            r = check_background_uniformity(image, alpha, self._profile)
            checks.append(_to_smart_check(r, "背景纯净度"))
        else:
            checks.append(SmartCheck("背景纯净度", "⏭️", "ok", "无遮罩数据，跳过"))

        # ── 6. 清晰度 (Q02) ──
        r = check_sharpness(image, self._profile)
        checks.append(_to_smart_check(r, "照片清晰度"))

        # ── 7. 亮度（辅助检测） ──
        checks.append(_check_brightness(gray))

        # ── 8. 面部大小比例（辅助检测 — 区分近距离/正常/远距离） ──
        bbox = cf.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            face_h_px = y2 - y1
            face_ratio = face_h_px / h
            if face_ratio > 0.40:
                checks.append(SmartCheck("拍摄距离", "💡", "warn",
                                         f"头部偏大 ({face_ratio:.0%})", "建议稍微远离镜头"))
            elif face_ratio < 0.20:
                checks.append(SmartCheck("拍摄距离", "💡", "warn",
                                         f"头部偏小 ({face_ratio:.0%})", "建议靠近镜头"))
            else:
                checks.append(SmartCheck("拍摄距离", "✅", "ok",
                                         f"头部占比适中 ({face_ratio:.0%})"))

        return SmartReport(checks=checks, total_time=time.time() - t0)
