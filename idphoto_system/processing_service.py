"""
证件照处理核心服务 (PhotoProcessingService)
=============================================
统一 api_server.py、run_demo.py、cli.py 三条路径的核心处理逻辑。
封装 HivisionIDPhotos IDCreator 调用 + 前后处理 + 合规检测。

用法:
    from idphoto_system.processing_service import PhotoProcessingService, ProcessRequest

    service = PhotoProcessingService()
    result = service.process(ProcessRequest(image=img, spec="一寸", color="blue"))
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

from .utils.image_utils import SPECS, COLORS, resolve_spec, resolve_color
from .matting import cascade_refine
from .compliance.smart_engine import SmartComplianceEngine
from .compliance.engine import ComplianceEngine


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ProcessRequest:
    """处理请求"""
    image: np.ndarray                          # BGR 图像
    spec: str = "一寸"
    color: str = "blue"
    bg_mode: str = "pure_color"                # pure_color | updown_gradient | center_gradient
    beauty_strength: float = 0
    brightness_strength: float = 0
    enable_edge_refine: bool = True
    enable_compliance: bool = True
    compliance_standard: str = "ISO"           # ISO | 中国 | US_VISA | SCHENGEN | JAPAN
    compliance_mode: str = "smart"             # smart (8项) | full (22+项)


@dataclass
class ProcessResult:
    """处理结果"""
    standard: np.ndarray                       # 标准尺寸 BGR 证件照
    matting_rgba: Optional[np.ndarray] = None  # BGRA 抠图结果
    alpha: Optional[np.ndarray] = None         # alpha 遮罩 (H, W) float32 [0,1]
    face_info: Optional[Dict] = None           # 人脸检测信息
    checks: Optional[Dict] = None              # 合规检测结果 (dict)
    timing: Dict[str, float] = field(default_factory=dict)
    spec_name: str = ""
    spec_width: int = 0
    spec_height: int = 0
    is_compliant: Optional[bool] = None
    matting_quality: Optional[Dict] = None     # 抠图质量报告


# ============================================================
# 服务
# ============================================================

class PhotoProcessingService:
    """证件照处理核心服务。

    封装 HivisionIDPhotos IDCreator 调用 + P1 增强模块。
    api_server / run_demo / cli 的共有逻辑收拢于此。

    用法:
        service = PhotoProcessingService()
        req = ProcessRequest(image=img, spec="一寸", color="blue")
        result = service.process(req)
    """

    def __init__(self, hivision_path: str = None):
        """
        :param hivision_path: HivisionIDPhotos 库路径。若为 None 则从 config 读取。
        """
        if hivision_path is None:
            from .config import HIVISION_PATH
            hivision_path = HIVISION_PATH

        import sys
        if hivision_path not in sys.path:
            sys.path.insert(0, hivision_path)

        self._hivision_path = hivision_path
        self._creator = None        # IDCreator 单例
        self._compliance_cache: Dict[str, Any] = {}  # 合规引擎缓存 (keyed by standard)

    # ── 公开 API ──

    def process(self, req: ProcessRequest) -> ProcessResult:
        """统一处理入口 — 单次调用完成全部流程。"""
        timing: Dict[str, float] = {}
        total_start = time.time()

        img = self._preprocess(req.image, timing)
        spec = resolve_spec(req.spec)
        rgb = resolve_color(req.color)
        bgr = (rgb[2], rgb[1], rgb[0])

        # [1] IDCreator: 人脸检测 + 抠图
        creator_result = self._run_creator(img, spec, req, crop_only=False, timing=timing)
        matting_rgba = creator_result["matting"]
        alpha = self._extract_alpha(matting_rgba)
        face_info = creator_result.get("face") or {}

        # [2] 可选: 边缘优化 (P1)
        if req.enable_edge_refine:
            alpha, mq = self._edge_refine(img, alpha, timing)
        else:
            mq = None

        # [3] 背景替换
        composited = self._composite(matting_rgba, alpha, bgr, req.bg_mode, timing)

        # [4] 精确裁剪 (使用 HivisionIDPhotos crop_only 模式)
        standard = self._precise_crop(composited, spec, timing)

        # [5] 可选: 合规检测
        checks = None
        is_compliant = None
        if req.enable_compliance:
            checks, is_compliant = self._run_compliance(
                img, face_info, alpha, spec, bgr, req, timing
            )

        timing["total"] = time.time() - total_start

        return ProcessResult(
            standard=standard,
            matting_rgba=matting_rgba,
            alpha=alpha,
            face_info=face_info,
            checks=checks,
            timing=timing,
            spec_name=spec.name,
            spec_width=spec.width,
            spec_height=spec.height,
            is_compliant=is_compliant,
            matting_quality=mq,
        )

    # ── 内部步骤 ──

    def _preprocess(self, image: np.ndarray, timing: dict) -> np.ndarray:
        t0 = time.time()

        # 统一缩放
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > 2000:
            scale = 2000.0 / max_side
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        # 确保 3 通道 BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        timing["preprocess"] = time.time() - t0
        return image

    def _get_creator(self):
        """延迟初始化 HivisionIDPhotos IDCreator"""
        if self._creator is None:
            from hivision import IDCreator
            from hivision.creator.choose_handler import choose_handler
            from .config import DEFAULT_MATTING_MODEL, DEFAULT_FACE_MODEL
            self._creator = IDCreator()
            choose_handler(self._creator, DEFAULT_MATTING_MODEL, DEFAULT_FACE_MODEL)
        return self._creator

    def _run_creator(self, image, spec, req, crop_only=False, timing=None):
        """调用 HivisionIDPhotos IDCreator"""
        from .config import HEAD_MEASURE_RATIO, HEAD_HEIGHT_RATIO, HEAD_TOP_RANGE

        t0 = time.time()
        creator = self._get_creator()

        contrast_ratio = int(req.beauty_strength) // 3 if req.beauty_strength else 0

        result = creator(
            image,
            size=spec.size,
            change_bg_only=False,
            crop_only=crop_only,
            head_measure_ratio=HEAD_MEASURE_RATIO,
            head_height_ratio=HEAD_HEIGHT_RATIO,
            head_top_range=HEAD_TOP_RANGE,
            whitening_strength=int(req.beauty_strength),
            brightness_strength=int(req.brightness_strength),
            contrast_strength=contrast_ratio,
        )

        key = "creator_crop" if crop_only else "creator_main"
        if timing is not None:
            timing[key] = time.time() - t0
        return {"matting": result.matting, "face": result.face, "standard": result.standard}

    def _extract_alpha(self, matting_rgba):
        """从 BGRA 抠图结果提取 alpha"""
        if matting_rgba.shape[2] >= 4:
            return matting_rgba[:, :, 3].astype(np.float32) / 255.0
        return np.ones(matting_rgba.shape[:2], dtype=np.float32)

    def _edge_refine(self, img, alpha, timing):
        """P1 级联边缘优化"""
        from .config import MAX_IMAGE_SIDE

        t0 = time.time()
        try:
            refined, mq = cascade_refine(img, alpha, radius=6, eps=1e-6)
            timing["edge_refine"] = time.time() - t0
            return refined, {"confidence": mq.confidence, "tier": mq.tier,
                             "method": mq.method, "needs_fallback": mq.needs_fallback}
        except Exception:
            timing["edge_refine"] = 0
            return alpha, None

    def _composite(self, matting_rgba, alpha, bgr, bg_mode, timing):
        """背景替换"""
        from hivision.utils import add_background

        t0 = time.time()
        alpha_uint8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([matting_rgba[:, :, :3], alpha_uint8])
        composited = add_background(rgba, bgr=bgr, mode=bg_mode).astype(np.uint8)
        if timing is not None:
            timing["composite"] = time.time() - t0
        return composited

    def _precise_crop(self, composited, spec, timing):
        """crop_only 模式精确裁剪"""
        from .config import HEAD_MEASURE_RATIO, HEAD_HEIGHT_RATIO, HEAD_TOP_RANGE

        t0 = time.time()

        composited_bgra = cv2.cvtColor(composited, cv2.COLOR_BGR2BGRA)
        creator = self._get_creator()
        result = creator(
            composited_bgra,
            size=spec.size,
            crop_only=True,
            head_measure_ratio=HEAD_MEASURE_RATIO,
            head_height_ratio=HEAD_HEIGHT_RATIO,
            head_top_range=HEAD_TOP_RANGE,
        )

        standard_bgr = cv2.cvtColor(result.standard, cv2.COLOR_BGRA2BGR)
        if timing is not None:
            timing["precise_crop"] = time.time() - t0
        return standard_bgr

    def _run_compliance(self, image, face_info, alpha, spec, bgr, req, timing):
        """合规检测 — 支持 smart/full 两种模式"""
        t0 = time.time()

        try:
            if req.compliance_mode == "smart":
                # SmartComplianceEngine 接受 HivisionIDPhotos 格式
                cf = face_info
                cf["roll_angle"] = face_info.get("roll_angle", 0)
                # 如果 face_info 中有 rectangle 但无 bbox，SmartComplianceEngine 内部会处理
                engine = SmartComplianceEngine()
                report = engine.check(image, face_info, alpha)
                result = report.to_dict()
                is_compliant = report.is_good
            else:
                # 完整 ComplianceEngine — 需要转换格式
                cf = self._convert_face_for_compliance(face_info)
                cache_key = req.compliance_standard
                if cache_key not in self._compliance_cache:
                    self._compliance_cache[cache_key] = ComplianceEngine(
                        standard=req.compliance_standard
                    )
                engine = self._compliance_cache[cache_key]
                report = engine.check(
                    image, cf, alpha,
                    spec_size=(spec.width, spec.height),
                    expected_bg_color=bgr,
                )
                result = report.to_dict()
                is_compliant = report.is_compliant
        except Exception:
            result = {"error": "compliance check failed"}
            is_compliant = None

        if timing is not None:
            timing["compliance"] = time.time() - t0
        return result, is_compliant

    @staticmethod
    def _convert_face_for_compliance(face_info: dict) -> dict:
        """HivisionIDPhotos face_info → ComplianceEngine 格式"""
        bbox = face_info.get("bbox")
        rect = face_info.get("rectangle")

        if bbox is None and rect and len(rect) == 4:
            cx, cy, fw, fh = rect
            bbox = (int(cx - fw/2), int(cy - fh/2), int(cx + fw/2), int(cy + fh/2))

        return {
            "bbox": bbox,
            "landmarks": face_info.get("landmarks"),
            "confidence": face_info.get("confidence", 1.0),
            "roll_angle": face_info.get("roll_angle", 0),
        }
