"""
级联精细化人像抠图 (Cascade Matting)
=====================================
★ 自研创新点二：三阶段级联抠图流水线。

基于 HivisionIDPhotos MODNet 粗抠图结果，通过:
1. 引导滤波 (Guided Filter) 边缘精细化
2. 置信度评分与自适应回退
实现发丝级别的高精度人像抠图。

用法:
    from idphoto_system.matting import (
        guided_filter,
        EdgeRefiner,
        AdaptiveFallback,
        cascade_refine,
    )

    # 完整级联优化
    refined, quality = cascade_refine(image, coarse_alpha)
"""

from .guided_filter import guided_filter, fast_guided_filter
from .edge_refiner import EdgeRefiner, TrimapGenerator
from .adaptive_fallback import AdaptiveFallback, MattingQuality, cascade_refine

__all__ = [
    "guided_filter",
    "fast_guided_filter",
    "EdgeRefiner",
    "TrimapGenerator",
    "AdaptiveFallback",
    "MattingQuality",
    "cascade_refine",
]
