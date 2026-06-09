"""
证件照主流水线 (ID Photo Pipeline)
====================================
完整的证件照生成流水线编排器。

处理流程:
    输入图像 → 人脸检测 → 人像抠图 → 背景替换 → 智能裁剪 → 规格调整 → 输出

P0 阶段流程（当前实现）:
    [1] 加载图像 & 预处理
    [2] 人脸检测 (MTCNN)
    [3] 人像抠图 (MODNet)
    [4] 背景替换 (纯色/渐变)
    [5] 智能裁剪 & 规格化
    [6] 输出 (单张 + 可选排版)

P1 扩展:
    - 合规检测引擎 (22+ 规则，5个维度)
    - 引导滤波边缘优化 + 自适应回退
"""

import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..inference.face_detector import FaceDetector, FaceNotFoundError, MultiFaceError
from ..inference.human_matting import HumanMatting
from ..processing.background import BackgroundReplacer
from ..processing.cropping import PhotoCropper
from ..processing.layout import LayoutGenerator
from ..utils.image_utils import (
    PhotoSpec,
    SPECS,
    COLORS,
    load_image,
    save_image,
    resolve_color,
    resolve_spec,
    array_to_base64,
)

# P1 新增模块
from ..compliance import ComplianceEngine, get_standard as get_compliance_standard
from ..matting import cascade_refine


class IDPhotoPipeline:
    """
    证件照生成流水线。

    用法:
        pipeline = IDPhotoPipeline()
        result = pipeline.process("input.jpg", spec="一寸", color="blue")

        # 或使用 numpy 数组
        img = cv2.imread("input.jpg")
        result = pipeline.process(img, spec="一寸", color="white")
    """

    def __init__(
        self,
        matting_model: str = "modnet_photographic_portrait_matting",
        face_model: str = "mtcnn",
        enable_layout: bool = False,
        enable_compliance: bool = True,
        enable_edge_refine: bool = True,
        head_height_ratio: float = 0.50,
        head_measure_ratio: float = 0.23,
    ):
        """
        初始化流水线。

        :param matting_model: 抠图模型名称
        :param face_model: 人脸检测模型名称
        :param enable_layout: 是否启用排版生成
        :param enable_compliance: 是否启用合规检测 (P1)
        :param enable_edge_refine: 是否启用边缘优化 (P1)
        :param head_height_ratio: 头部中心在画面高度位置的比例
        :param head_measure_ratio: 头部面积占画面比例
        """
        self.face_detector = FaceDetector(model_type=face_model)
        self.matting = HumanMatting(model_name=matting_model)
        self.background_replacer = BackgroundReplacer()
        self.cropper = PhotoCropper(
            head_height_ratio=head_height_ratio,
            head_measure_ratio=head_measure_ratio,
        )
        self.layout_generator = LayoutGenerator()
        self.enable_layout = enable_layout
        self.enable_compliance = enable_compliance
        self.enable_edge_refine = enable_edge_refine

        # P1 模块（延迟初始化）
        self._compliance_engine = None
        self._matting_quality = None

        # 性能统计
        self.last_timing: Dict[str, float] = {}

        print(f"[Pipeline] 初始化完成")
        print(f"  人脸检测: {face_model}")
        print(f"  抠图模型: {matting_model}")
        print(f"  P1 合规检测: {'启用' if enable_compliance else '禁用'}")
        print(f"  P1 边缘优化: {'启用' if enable_edge_refine else '禁用'}")

    def process(
        self,
        input_image,
        spec: str = "一寸",
        color: str = "white",
        bg_mode: str = "pure_color",
        layout: Optional[str] = None,
        output_path: Optional[str] = None,
        return_base64: bool = False,
        dpi: int = 300,
        standard: str = "ISO",
    ) -> Dict:
        """
        处理单张证件照的完整流水线。

        :param input_image: 输入图像路径 (str) 或 numpy 数组 (np.ndarray)
        :param spec: 证件照规格名称，如 "一寸"、"二寸"
        :param color: 背景颜色，如 "white"、"blue"、"red" 或 hex 码
        :param bg_mode: 背景模式 — "pure_color" | "updown_gradient" | "center_gradient"
        :param layout: 可选的排版方案，如 "一寸×8"、"二寸×4"
        :param output_path: 输出文件路径 (可选)
        :param return_base64: 是否返回 Base64 编码
        :param dpi: 输出图像的 DPI
        :param standard: 合规检测标准 — "ISO", "中国", "US_VISA", "SCHENGEN", "JAPAN"
        :return: 处理结果字典，包含:
            - "standard": 标准尺寸证件照 (numpy array)
            - "hd": 高清原尺寸证件照 (numpy array)
            - "matting": 抠图结果 (numpy array, BGRA)
            - "alpha": alpha 遮罩 (numpy array)
            - "face": 人脸检测信息 (dict)
            - "compliance": 合规检测报告 (dict, P1)
            - "matting_quality": 抠图质量报告 (dict, P1)
            - "layout": 排版图 (numpy array, 如果启用)
            - "timing": 各阶段耗时 (dict)
            - "base64": Base64 编码 (如果 return_base64=True)
        """
        total_start = time.time()
        timing = {}

        # 解析参数
        photo_spec = resolve_spec(spec)
        bg_color = resolve_color(color)

        # HivisionIDPhotos 使用 BGR 顺序
        bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])

        # ---------- [1] 加载图像 ----------
        t1 = time.time()
        if isinstance(input_image, str):
            image = load_image(input_image)
            input_path = input_image
        elif isinstance(input_image, np.ndarray):
            image = input_image.copy()
            input_path = "memory"
        else:
            raise TypeError(f"不支持的输入类型: {type(input_image)}")

        # 统一缩放到最大边长2000px
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > 2000:
            scale = 2000 / max_side
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        timing["load"] = time.time() - t1
        print(f"[1] 图像加载: {timing['load']:.3f}s | 尺寸: {image.shape}")

        # ---------- [2] 人脸检测 ----------
        t2 = time.time()
        try:
            face_info = self.face_detector.detect(image)
        except FaceNotFoundError as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": "no_face",
                "timing": timing,
            }
        except MultiFaceError as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": "multi_face",
                "timing": timing,
            }
        timing["face_detect"] = time.time() - t2
        print(f"[2] 人脸检测: {timing['face_detect']:.3f}s | "
              f"质量: {face_info.get('quality', 'N/A')}")

        # ---------- [3] 人像抠图 ----------
        t3 = time.time()
        matting_rgba, alpha = self.matting.matte(image)
        timing["matting"] = time.time() - t3
        print(f"[3] 人像抠图: {timing['matting']:.3f}s")

        # ---------- [3.5] 级联抠图边缘优化 (P1) ----------
        matting_quality = None
        if self.enable_edge_refine:
            t35 = time.time()
            try:
                refined_alpha, matting_quality = cascade_refine(
                    image, alpha, radius=8, eps=1e-6
                )
                # 用细化后的 alpha 更新抠图结果
                refined_alpha_4ch = np.stack([refined_alpha] * 4, axis=-1)
                matting_rgba = (matting_rgba.astype(np.float32) * refined_alpha_4ch).astype(np.uint8)
                matting_rgba[:, :, 3] = (refined_alpha * 255).astype(np.uint8)
                alpha = refined_alpha
                timing["edge_refine"] = time.time() - t35
                print(f"[3.5] 边缘优化: {timing['edge_refine']:.3f}s | "
                      f"置信度: {matting_quality.confidence:.0f} ({matting_quality.tier})")
            except Exception as e:
                print(f"[3.5] 边缘优化: 跳过 ({e})")
                timing["edge_refine"] = 0

        # ---------- [4] 背景替换 ----------
        t4 = time.time()
        composited = self.background_replacer.replace(
            matting_rgba, color=bg_color_bgr, mode=bg_mode,
        )
        timing["background"] = time.time() - t4
        print(f"[4] 背景替换: {timing['background']:.3f}s | "
              f"颜色: {color} | 模式: {bg_mode}")

        # ---------- [5] 智能裁剪 & 规格化 ----------
        t5 = time.time()
        # 对合成图执行裁剪和规格化
        standard = self.cropper.crop(composited, face_info, photo_spec)

        # 生成高清版（较大的图像，用于打印）
        hd_scale = 2.0  # 高清版2倍分辨率
        hd_size = (int(photo_spec.width * hd_scale), int(photo_spec.height * hd_scale))
        hd = cv2.resize(standard, hd_size, interpolation=cv2.INTER_LANCZOS4)

        timing["crop_resize"] = time.time() - t5
        print(f"[5] 裁剪规格化: {timing['crop_resize']:.3f}s | "
              f"规格: {photo_spec.label}")

        # ---------- [5.5] 合规检测 (P1) ----------
        compliance_report = None
        if self.enable_compliance:
            t55 = time.time()
            try:
                if self._compliance_engine is None:
                    self._compliance_engine = ComplianceEngine(standard=standard)
                elif self._compliance_engine.standard_name != standard:
                    self._compliance_engine = ComplianceEngine(standard=standard)

                compliance_report = self._compliance_engine.check(
                    image, face_info, alpha,
                    spec_size=(photo_spec.width, photo_spec.height),
                    expected_bg_color=bg_color_bgr,
                )
                timing["compliance"] = time.time() - t55
                status = "[通过]" if compliance_report.is_compliant else "[未通过]"
                print(f"[5.5] 合规检测: {timing['compliance']:.3f}s | "
                      f"评分: {compliance_report.overall_score:.0f}/100 {status}")
                if compliance_report.critical_failures:
                    for cf in compliance_report.critical_failures:
                        print(f"      关键问题: {cf.rule_id} {cf.name} → {cf.hint}")
            except Exception as e:
                print(f"[5.5] 合规检测: 跳过 ({e})")
                timing["compliance"] = 0

        # ---------- [6] 排版（可选） ----------
        layout_image = None
        if layout or self.enable_layout:
            t6 = time.time()
            if layout is None:
                # 自动选择排版
                layout = f"{spec}×{'8' if '一寸' in spec else '4'}"
            try:
                layout_image = self.layout_generator.generate_standard(
                    standard, layout, photo_spec
                )
                timing["layout"] = time.time() - t6
                print(f"[6] 排版生成: {timing['layout']:.3f}s | 方案: {layout}")
            except ValueError:
                print(f"[6] 排版跳过: 不支持的方案 '{layout}'")

        # ---------- 保存输出 ----------
        if output_path:
            save_image(standard, output_path, dpi=dpi)

            # 保存高清版
            base, ext = os.path.splitext(output_path)
            hd_path = f"{base}_hd{ext}"
            save_image(hd, hd_path, dpi=dpi)

            if layout_image is not None:
                layout_path = f"{base}_layout{ext}"
                save_image(layout_image, layout_path, dpi=dpi)

            print(f"  输出已保存: {output_path}")

        # ---------- 总结 ----------
        total_time = time.time() - total_start
        timing["total"] = total_time
        self.last_timing = timing

        print(f"[总耗时] {total_time:.3f}s")
        print(f"  加载:     {timing.get('load', 0):.3f}s")
        print(f"  人脸检测: {timing.get('face_detect', 0):.3f}s")
        print(f"  抠图:     {timing.get('matting', 0):.3f}s")
        print(f"  背景替换: {timing.get('background', 0):.3f}s")
        print(f"  裁剪:     {timing.get('crop_resize', 0):.3f}s")

        result = {
            "status": "ok",
            "standard": standard,
            "hd": hd,
            "matting": matting_rgba,
            "alpha": alpha,
            "face": face_info,
            "spec": photo_spec,
            "timing": timing,
        }

        if layout_image is not None:
            result["layout"] = layout_image

        if return_base64:
            result["base64"] = array_to_base64(standard)

        # P1: 合规检测报告
        if compliance_report is not None:
            result["compliance"] = compliance_report.to_dict()
            result["is_compliant"] = compliance_report.is_compliant

        # P1: 抠图质量报告
        if matting_quality is not None:
            from ..matting.adaptive_fallback import AdaptiveFallback
            fb = AdaptiveFallback()
            result["matting_quality"] = fb.to_dict(matting_quality)

        return result

    def quick_process(
        self,
        image: np.ndarray,
        spec_name: str = "一寸",
        color_name: str = "white",
    ) -> np.ndarray:
        """
        快速处理 — 返回标准尺寸证件照 numpy 数组。
        简化接口，仅返回最终图像。
        """
        result = self.process(image, spec=spec_name, color=color_name)
        if result["status"] != "ok":
            raise RuntimeError(result["error"])
        return result["standard"]

    def batch_process(
        self,
        input_paths: list,
        spec: str = "一寸",
        color: str = "white",
        output_dir: str = "outputs",
    ) -> list:
        """
        批量处理多张照片。

        :param input_paths: 输入文件路径列表
        :param spec: 统一规格
        :param color: 统一背景色
        :param output_dir: 输出目录
        :return: 处理结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, path in enumerate(input_paths):
            print(f"\n{'='*50}")
            print(f"处理 [{i+1}/{len(input_paths)}]: {path}")
            print(f"{'='*50}")

            base_name = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{spec}.png")

            result = self.process(
                path,
                spec=spec,
                color=color,
                output_path=output_path,
            )
            results.append(result)

        # 统计
        success = sum(1 for r in results if r["status"] == "ok")
        failed = len(results) - success
        print(f"\n批量处理完成: 成功 {success}, 失败 {failed}")

        return results

    def get_performance_report(self) -> Dict:
        """获取性能报告。"""
        if not self.last_timing:
            return {"message": "尚未运行流水线"}

        total = self.last_timing.get("total", 0)
        return {
            "total_time": f"{total:.3f}s",
            "breakdown": {
                k: f"{v:.3f}s ({v/total*100:.1f}%)"
                for k, v in self.last_timing.items()
                if k != "total"
            },
            "fps": f"{1/total:.1f}" if total > 0 else "N/A",
        }
