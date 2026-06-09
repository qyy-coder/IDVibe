"""
合规检测引擎 (Compliance Engine)
===================================
编排 22+ 项检测规则的执行和结果汇总。

数据模型在 models.py 中定义以避免循环导入。
"""

import time
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Verdict, RuleResult, ComplianceReport
from .standards import StandardProfile, get_standard


class ComplianceEngine:
    """
    合规检测引擎 — 编排所有检测规则的执行。

    用法:
        engine = ComplianceEngine(standard="ISO")
        report = engine.check(image, face_info, alpha_matte)
    """

    def __init__(self, standard: str = "ISO"):
        self.standard_name = standard
        self.profile = get_standard(standard)

    def check(
        self,
        image: np.ndarray,
        face_info: Dict,
        alpha_matte: Optional[np.ndarray] = None,
        spec_size: Tuple[int, int] = (295, 413),
        expected_bg_color: Tuple[int, int, int] = None,
    ) -> ComplianceReport:
        """
        执行全面的合规检测。

        :param image: BGR 格式输入图像 (H, W, 3)
        :param face_info: 人脸检测结果 (bbox, landmarks, etc.)
        :param alpha_matte: alpha 遮罩 (H, W)，值域 [0, 1]
        :param spec_size: 目标证件照尺寸 (width, height)
        :param expected_bg_color: 期望背景色 BGR 元组
        :return: ComplianceReport
        """
        # 延迟导入检测模块（避免循环依赖）
        from . import geometric_checks
        from . import pose_checks
        from . import facial_checks
        from . import lighting_checks
        from . import quality_checks

        total_start = time.time()
        all_rules = []

        # ----- 1. 几何检测 -----
        all_rules.extend(
            geometric_checks.run_geometric_checks(
                face_info, image.shape[:2], self.profile
            )
        )

        # ----- 2. 姿态检测 -----
        all_rules.extend(
            pose_checks.run_pose_checks(
                face_info, image.shape[:2], self.profile
            )
        )

        # ----- 3. 面部状态检测 -----
        all_rules.extend(
            facial_checks.run_facial_checks(
                image, face_info, self.profile
            )
        )

        # ----- 4. 光照与色彩检测 -----
        all_rules.extend(
            lighting_checks.run_lighting_checks(
                image, face_info, alpha_matte, self.profile, expected_bg_color
            )
        )

        # ----- 5. 图像质量检测 -----
        all_rules.extend(
            quality_checks.run_quality_checks(
                image, spec_size, self.profile
            )
        )

        total_time = time.time() - total_start

        return ComplianceReport(
            standard=self.profile.label,
            rules=all_rules,
            total_time=total_time,
            timestamp=datetime.datetime.now().isoformat(),
        )

    def quick_check(
        self,
        image: np.ndarray,
        face_info: Dict,
        alpha_matte: Optional[np.ndarray] = None,
    ) -> Tuple[bool, List[str]]:
        """
        快速检查 — 返回是否通过 + 关键问题列表。
        """
        report = self.check(image, face_info, alpha_matte)
        issues = [
            f"{r.rule_id}: {r.hint}"
            for r in report.critical_failures
        ]
        return report.is_compliant, issues
