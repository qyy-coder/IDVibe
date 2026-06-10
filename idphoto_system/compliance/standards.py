"""
多国标准规则预设
==================
定义不同国家/地区的证件照合规检测阈值。

标准来源:
- ISO: ISO/IEC 19794-5:2005
- ICAO: ICAO 9303
- 中国: GA/T 公安机关标准
- 美国: US Department of State
- 申根: EU Schengen
- 日本: Japan MOFA
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class StandardProfile:
    """证件照合规标准配置"""
    name: str
    label: str
    description: str = ""

    # 几何检测阈值
    head_ratio_range: Tuple[float, float] = (0.55, 0.75)     # 头部占画面比例
    eye_line_range: Tuple[float, float] = (0.45, 0.65)       # 眼睛水平线位置
    face_center_max_offset: float = 0.05                     # 人脸居中最大偏移
    chin_margin_min: float = 0.05                            # 下颚到画面下沿最小距离

    # 姿态检测阈值
    yaw_max: float = 5.0           # 偏航角最大值 (度)
    pitch_max: float = 5.0         # 俯仰角最大值 (度)
    roll_max: float = 5.0          # 翻滚角最大值 (度)

    # 面部状态检测阈值
    ear_min: float = 0.20          # 眼睛纵横比最小值
    mar_max: float = 0.30          # 嘴巴纵横比最大值（闭合）
    mar_smile_max: float = 0.45    # 微笑嘴巴纵横比最大值
    glasses_allowed: bool = True   # 是否允许戴眼镜
    glare_max_brightness: int = 240  # 反光最大亮度
    red_eye_max_ratio: float = 1.5   # 红眼红色通道比值

    # 光照检测阈值
    face_uniformity_min: float = 0.80  # 面部光照均匀度最小值（左右比）
    shadow_l_ratio: float = 0.85       # 阴影L通道比例
    bg_uniformity_min: float = 85      # 背景均匀度最低评分
    bg_color_max_distance: float = 30  # 背景色最大欧氏距离

    # 图像质量检测阈值
    min_resolution_px: int = 295       # 最小分辨率（像素）
    sharpness_min: float = 100         # 清晰度 Laplacian 方差最小值
    recommend_dpi: int = 300           # 推荐 DPI

    # 规则权重（影响overall_score）
    rule_weights: Dict[str, float] = field(default_factory=lambda: {
        "geometric": 0.25,
        "pose": 0.25,
        "facial": 0.20,
        "lighting": 0.15,
        "quality": 0.15,
    })

    # 关键规则（FAIL 则整体不合格）
    critical_rules: List[str] = field(default_factory=lambda: [
        "G01", "P01", "P02", "P03", "F01",
    ])


# ============================================================
# 各标准定义
# ============================================================

STANDARDS: Dict[str, StandardProfile] = {
    "ISO": StandardProfile(
        name="ISO",
        label="ISO/IEC 19794-5 (国际标准)",
        description="国际证件照标准，适用于大多数国家和组织",
        head_ratio_range=(0.55, 0.75),
        eye_line_range=(0.45, 0.65),
        face_center_max_offset=0.05,
        chin_margin_min=0.05,
        yaw_max=5.0,
        pitch_max=5.0,
        roll_max=5.0,
        ear_min=0.20,
        mar_max=0.30,
        mar_smile_max=0.45,
        glasses_allowed=True,
        glare_max_brightness=240,
        red_eye_max_ratio=1.5,
        face_uniformity_min=0.80,
        shadow_l_ratio=0.80,
        bg_uniformity_min=85,
        bg_color_max_distance=150,
        min_resolution_px=295,
        sharpness_min=100,
    ),

    "中国": StandardProfile(
        name="中国",
        label="中国标准 (GA/T)",
        description="中国公安机关证件照标准，适用于身份证、护照、驾驶证",
        head_ratio_range=(0.55, 0.75),
        eye_line_range=(0.48, 0.62),
        face_center_max_offset=0.04,
        chin_margin_min=0.06,
        yaw_max=5.0,
        pitch_max=5.0,
        roll_max=5.0,
        ear_min=0.20,
        mar_max=0.28,
        mar_smile_max=0.40,
        glasses_allowed=False,  # 中国标准通常不允许眼镜
        glare_max_brightness=230,
        red_eye_max_ratio=1.4,
        face_uniformity_min=0.80,
        shadow_l_ratio=0.80,
        bg_uniformity_min=85,
        bg_color_max_distance=130,
        min_resolution_px=295,
        sharpness_min=120,
    ),

    "US_VISA": StandardProfile(
        name="US_VISA",
        label="美国签证标准",
        description="美国国务院签证照片要求",
        head_ratio_range=(0.58, 0.72),
        eye_line_range=(0.50, 0.62),
        face_center_max_offset=0.03,
        chin_margin_min=0.05,
        yaw_max=4.0,
        pitch_max=4.0,
        roll_max=4.0,
        ear_min=0.22,
        mar_max=0.25,
        mar_smile_max=0.35,
        glasses_allowed=False,
        glare_max_brightness=235,
        red_eye_max_ratio=1.3,
        face_uniformity_min=0.80,
        shadow_l_ratio=0.80,
        bg_uniformity_min=85,
        bg_color_max_distance=130,
        min_resolution_px=295,
        sharpness_min=100,
    ),

    "SCHENGEN": StandardProfile(
        name="SCHENGEN",
        label="申根签证标准",
        description="欧盟申根区签证照片要求 (35×45mm)",
        head_ratio_range=(0.60, 0.75),
        eye_line_range=(0.48, 0.60),
        face_center_max_offset=0.05,
        chin_margin_min=0.05,
        yaw_max=5.0,
        pitch_max=5.0,
        roll_max=5.0,
        ear_min=0.20,
        mar_max=0.30,
        mar_smile_max=0.40,
        glasses_allowed=True,
        glare_max_brightness=240,
        red_eye_max_ratio=1.5,
        face_uniformity_min=0.80,
        shadow_l_ratio=0.80,
        bg_uniformity_min=85,
        bg_color_max_distance=130,
        min_resolution_px=295,
        sharpness_min=100,
    ),

    "JAPAN": StandardProfile(
        name="JAPAN",
        label="日本签证标准",
        description="日本入国管理局照片要求 (35×45mm)",
        head_ratio_range=(0.58, 0.73),
        eye_line_range=(0.48, 0.60),
        face_center_max_offset=0.05,
        chin_margin_min=0.05,
        yaw_max=5.0,
        pitch_max=5.0,
        roll_max=5.0,
        ear_min=0.20,
        mar_max=0.28,
        mar_smile_max=0.40,
        glasses_allowed=True,
        glare_max_brightness=240,
        red_eye_max_ratio=1.5,
        face_uniformity_min=0.82,
        shadow_l_ratio=0.80,
        bg_uniformity_min=85,
        bg_color_max_distance=130,
        min_resolution_px=295,
        sharpness_min=100,
    ),
    # 快速检测专用标准（阈值较宽松，用于前端 SmartComplianceEngine）
    "quick_check": StandardProfile(
        name="quick_check",
        label="快速检测 (Smart)",
        description="前端智能检测使用的宽松标准，仅对严重问题报警",
        head_ratio_range=(0.20, 0.90),       # 更宽范围
        eye_line_range=(0.30, 0.75),
        face_center_max_offset=0.15,          # 允许更大偏移
        chin_margin_min=0.01,
        yaw_max=8.0,                          # 更宽松角度阈值
        pitch_max=8.0,
        roll_max=8.0,
        ear_min=0.12,                         # 更低眼睛开合阈值
        mar_max=0.40,
        mar_smile_max=0.55,
        glasses_allowed=True,
        glare_max_brightness=245,
        red_eye_max_ratio=1.8,
        face_uniformity_min=0.65,
        shadow_l_ratio=0.75,
        bg_uniformity_min=70,
        bg_color_max_distance=180,
        min_resolution_px=200,
        sharpness_min=30,
    ),
}


def get_standard(name: str) -> StandardProfile:
    """获取标准配置，支持别名模糊匹配"""
    name_lower = name.lower().strip()

    # 精确匹配
    if name in STANDARDS:
        return STANDARDS[name]

    # 别名映射
    aliases = {
        "iso": "ISO",
        "icao": "ISO",
        "iec": "ISO",
        "cn": "中国",
        "china": "中国",
        "chinese": "中国",
        "中国": "中国",
        "us": "US_VISA",
        "usa": "US_VISA",
        "american": "US_VISA",
        "美国": "US_VISA",
        "eu": "SCHENGEN",
        "schengen": "SCHENGEN",
        "europe": "SCHENGEN",
        "申根": "SCHENGEN",
        "jp": "JAPAN",
        "japan": "JAPAN",
        "日本": "JAPAN",
        "quick": "quick_check",
        "smart": "quick_check",
    }

    key = aliases.get(name_lower)
    if key and key in STANDARDS:
        return STANDARDS[key]

    # 默认回退到 ISO
    return STANDARDS["ISO"]
