"""
智能合规检测引擎 (Smart Compliance Engine)
只对真正严重的问题报警，阈值自适应照片质量
"""

import time, datetime
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

@dataclass
class SmartCheck:
    """智能检测结果"""
    name: str          # 检测项名称
    icon: str          # 图标 emoji
    status: str        # ok / warn / issue
    detail: str        # 详细描述
    suggestion: str    # 改进建议（仅warn/issue时有）

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

    def to_dict(self):
        return {
            "ok": self.ok_count, "warn": self.warn_count, "issue": self.issue_count,
            "is_good": self.is_good, "total_time": round(self.total_time, 3),
            "checks": [{"name": c.name, "icon": c.icon, "status": c.status,
                        "detail": c.detail, "suggestion": c.suggestion} for c in self.checks]
        }

class SmartComplianceEngine:
    """智能合规检测——只检测真正重要的问题"""

    def __init__(self):
        pass

    def check(self, image: np.ndarray, face_info: dict, alpha: np.ndarray = None) -> SmartReport:
        t0 = time.time()
        checks = []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # 获取人脸信息
        bbox = face_info.get("bbox")
        rect = face_info.get("rectangle")  # HivisionIDPhotos format

        # --- 1. 人脸检测 ---
        face_detected = False
        if rect and len(rect) == 4:
            cx, cy, fw, fh = rect
            bbox = (int(cx-fw/2), int(cy-fh/2), int(cx+fw/2), int(cy+fh/2))
            face_detected = True
        elif bbox:
            face_detected = True

        if not face_detected:
            checks.append(SmartCheck("人脸检测", "❌", "issue",
                "未检测到人脸", "请确保面部完整、光线充足"))
            return SmartReport(checks=checks, total_time=time.time()-t0)

        x1, y1, x2, y2 = bbox
        face_h = y2 - y1
        face_w = x2 - x1

        # 根据人脸大小判断照片类型并自适应阈值
        face_ratio = face_h / h
        is_closeup = face_ratio > 0.40     # 近距离自拍
        is_normal = 0.20 <= face_ratio <= 0.40  # 正常距离
        is_far = face_ratio < 0.20         # 远距离

        # --- 2. 头部大小 ---
        if is_far:
            checks.append(SmartCheck("头部大小", "⚠️", "warn",
                f"头部偏小 ({face_ratio:.0%})，建议靠近镜头",
                "靠近镜头拍摄或裁剪后效果更好"))
        elif is_closeup:
            checks.append(SmartCheck("头部大小", "⚠️", "warn",
                f"头部偏大 ({face_ratio:.0%})，建议稍微远离",
                "稍微后退一点，留出肩部空间"))
        else:
            checks.append(SmartCheck("头部大小", "✅", "ok",
                f"头部占比适中 ({face_ratio:.0%})", ""))

        # --- 3. 人脸居中 ---
        face_cx = (x1 + x2) / 2
        offset = abs(face_cx - w/2) / w
        if offset > 0.15:
            direction = "左" if face_cx < w/2 else "右"
            checks.append(SmartCheck("人脸居中", "⚠️", "warn",
                f"人脸偏{direction} ({offset:.0%})",
                f"请将头部向{'右' if direction=='左' else '左'}移动"))
        else:
            checks.append(SmartCheck("人脸居中", "✅", "ok",
                f"人脸位置端正 ({offset:.0%})", ""))

        # --- 4. 头部倾斜 ---
        roll = face_info.get("roll_angle", 0)
        if abs(roll) > 15:
            checks.append(SmartCheck("头部倾斜", "⚠️", "warn",
                f"头部倾斜 {abs(roll):.0f}°", "请将头部摆正"))
        elif abs(roll) > 8:
            checks.append(SmartCheck("头部倾斜", "💡", "warn",
                f"轻微倾斜 {abs(roll):.0f}°", "建议稍微摆正"))
        else:
            checks.append(SmartCheck("头部倾斜", "✅", "ok",
                f"头部端正 ({abs(roll):.0f}°)", ""))

        # --- 5. 光照均匀度 ---
        if face_detected:
            face_roi = gray[y1:y2, x1:x2]
            mid = face_roi.shape[1] // 2
            left = face_roi[:, :mid].mean()
            right = face_roi[:, mid:].mean()
            lr_ratio = min(left, right) / max(left, right) if max(left, right) > 0 else 1
            if lr_ratio < 0.65:
                darker = "左" if left < right else "右"
                checks.append(SmartCheck("面部光照", "⚠️", "warn",
                    f"{darker}侧明显偏暗 ({lr_ratio:.0%})",
                    f"请调整光源方向，避免侧光"))
            elif lr_ratio < 0.80:
                checks.append(SmartCheck("面部光照", "💡", "warn",
                    f"左右略有不均 ({lr_ratio:.0%})",
                    "建议面向光源拍摄"))
            else:
                checks.append(SmartCheck("面部光照", "✅", "ok",
                    f"光照均匀 ({lr_ratio:.0%})", ""))

        # --- 6. 清晰度 ---
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 30:
            checks.append(SmartCheck("照片清晰度", "⚠️", "warn",
                f"照片较模糊 (清晰度: {laplacian_var:.0f})",
                "请确保对焦准确，手持稳定"))
        elif laplacian_var < 80:
            checks.append(SmartCheck("照片清晰度", "💡", "warn",
                f"清晰度一般 ({laplacian_var:.0f})",
                "建议在光线充足环境下拍摄"))
        else:
            checks.append(SmartCheck("照片清晰度", "✅", "ok",
                f"清晰度良好 ({laplacian_var:.0f})", ""))

        # --- 7. 背景纯净度 ---
        if alpha is not None:
            bg = image[alpha < 0.1]
            if len(bg) > 500:
                bg_lab = cv2.cvtColor(bg.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2Lab).reshape(-1,3).astype(np.float32)
                var_ab = np.var(bg_lab[:,1]) + np.var(bg_lab[:,2])
                if var_ab > 50:
                    checks.append(SmartCheck("背景纯净度", "💡", "warn",
                        "原图背景较杂乱",
                        "建议在纯色背景前拍摄，或使用三色同出功能换底"))
                else:
                    checks.append(SmartCheck("背景纯净度", "✅", "ok",
                        "背景较纯净", ""))

        # --- 8. 照片过曝/过暗 ---
        mean_bright = gray.mean()
        if mean_bright < 50:
            checks.append(SmartCheck("照片亮度", "⚠️", "warn",
                f"照片过暗 ({mean_bright:.0f})",
                "请增加环境光线或使用闪光灯"))
        elif mean_bright > 230:
            checks.append(SmartCheck("照片亮度", "⚠️", "warn",
                f"照片过曝 ({mean_bright:.0f})",
                "请避免直射强光，降低曝光"))
        else:
            checks.append(SmartCheck("照片亮度", "✅", "ok",
                f"亮度正常 ({mean_bright:.0f})", ""))

        return SmartReport(checks=checks, total_time=time.time()-t0)
