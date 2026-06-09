"""
合规检测引擎 (Compliance Detection Engine)
===========================================
★ 本系统最重要的自研创新模块。

提供 22+ 项自动检测规则，覆盖 ISO/IEC 19794-5、ICAO 9303
及中国国家标准的核心要求。

检测维度:
- 几何检测 (Geometric)
- 姿态检测 (Pose)
- 面部状态检测 (Facial State)
- 光照与色彩检测 (Illumination & Color)
- 图像质量检测 (Image Quality)

用法:
    from idphoto_system.compliance import ComplianceEngine

    engine = ComplianceEngine(standard="ISO")
    report = engine.check(image, face_info, alpha_matte)
    print(report.summary())
"""

from .models import Verdict, RuleResult, ComplianceReport
from .engine import ComplianceEngine
from .standards import STANDARDS, get_standard

__all__ = [
    "ComplianceEngine",
    "ComplianceReport",
    "RuleResult",
    "Verdict",
    "STANDARDS",
    "get_standard",
]
