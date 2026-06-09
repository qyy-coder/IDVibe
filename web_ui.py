#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI一照成证 — Web 交互界面 (Gradio)
====================================
类似 HivisionIDPhotos 的用户友好界面
"""

import sys, os, time

HIVISION_PATH = r"C:\Users\24817\HivisionIDPhotos"
sys.path.insert(0, HIVISION_PATH)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_PATH)

import numpy as np
import cv2
import gradio as gr

from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from hivision.utils import add_background, resize_image_to_kb

from idphoto_system.compliance import ComplianceEngine
from idphoto_system.matting import cascade_refine
from idphoto_system.processing.layout import LayoutGenerator
from idphoto_system.utils.image_utils import SPECS, COLORS, resolve_color

# ============================================================
# 核心处理函数
# ============================================================

creator = None

def get_creator():
    global creator
    if creator is None:
        creator = IDCreator()
        choose_handler(creator, "hivision_modnet", "mtcnn")
    return creator


def process_idphoto(
    input_image,
    spec_name: str,
    color_name: str,
    standard_name: str,
    enable_refine: bool = True,
    enable_compliance: bool = True,
):
    """完整的证件照处理流水线"""
    if input_image is None:
        return None, None, "请先上传照片", None

    total_start = time.time()
    logs = []

    try:
        # 解析参数
        spec = SPECS.get(spec_name)
        if spec is None:
            return None, None, "未知规格", None

        rgb = resolve_color(color_name)
        bgr = (rgb[2], rgb[1], rgb[0])

        img = input_image.copy()
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        if max(h, w) > 2000:
            s = 2000 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))

        # --- 抠图 + 人脸检测 ---
        c = get_creator()
        result = c(
            img,
            size=spec.size,
            change_bg_only=False,
            head_measure_ratio=0.23,
            head_height_ratio=0.45,
            head_top_range=(0.12, 0.1),
        )
        matting = result.matting
        alpha_raw = matting[:, :, 3].astype(np.float32) / 255.0
        logs.append(f"抠图完成 ({time.time() - total_start:.1f}s)")

        # --- P1 边缘优化 ---
        if enable_refine:
            try:
                refined, mq = cascade_refine(img, alpha_raw, radius=6, eps=1e-6)
                alpha = refined
                logs.append(f"边缘优化: {mq.confidence:.0f}/100 ({mq.tier})")
            except Exception:
                alpha = alpha_raw
                mq = None
        else:
            alpha = alpha_raw
            mq = None

        # --- 背景替换 ---
        rgba = np.dstack([matting[:, :, :3], (alpha * 255).astype(np.uint8)])
        composited = add_background(rgba, bgr=bgr, mode="pure_color").astype(np.uint8)

        # --- 精确裁剪 ---
        composited_bgra = cv2.cvtColor(composited, cv2.COLOR_BGR2BGRA)
        c2 = IDCreator()
        choose_handler(c2, "hivision_modnet", "mtcnn")
        result2 = c2(
            composited_bgra,
            size=spec.size,
            crop_only=True,
            head_measure_ratio=0.23,
            head_height_ratio=0.45,
            head_top_range=(0.12, 0.1),
        )
        standard = cv2.cvtColor(result2.standard, cv2.COLOR_BGRA2BGR)
        hd = cv2.cvtColor(result2.hd, cv2.COLOR_BGRA2BGR)
        logs.append(f"裁剪完成 ({time.time() - total_start:.1f}s)")

        # --- 排版 ---
        try:
            gen = LayoutGenerator()
            lname = f"{spec_name}×{'8' if '一寸' in spec_name else '4'}"
            layout = gen.generate_standard(standard, lname, spec)
        except Exception:
            layout = None

        # --- P1 合规检测 ---
        compliance_text = ""
        if enable_compliance:
            raw_face = result2.face or {}
            fc = {"bbox": None, "landmarks": None, "confidence": 1.0}
            rect = raw_face.get("rectangle")
            if rect and len(rect) == 4:
                cx, cy, fw, fh = rect
                fc["bbox"] = (int(cx - fw/2), int(cy - fh/2), int(cx + fw/2), int(cy + fh/2))
            fc["roll_angle"] = raw_face.get("roll_angle", 0)

            engine = ComplianceEngine(standard=standard_name)
            report = engine.check(
                img, fc, alpha,
                spec_size=(spec.width, spec.height),
                expected_bg_color=bgr,
            )

            # 格式化合规报告
            status = "PASS" if report.is_compliant else "FAIL"
            lines = [
                f"综合评分: {report.overall_score:.0f}/100 [{status}]",
                f"通过 {report.passed_count} | 失败 {report.failed_count} | 警告 {report.warn_count} | 共 {report.total_count} 项",
                f"检测耗时: {report.total_time:.3f}s",
                "",
            ]
            if report.critical_failures:
                lines.append("--- 关键问题 (会导致照片被拒) ---")
                for cf in report.critical_failures:
                    lines.append(f"  {cf.rule_id} {cf.name}: {cf.hint}")
                lines.append("")

            cat_labels = {"geometric": "几何", "pose": "姿态", "facial": "面部",
                          "lighting": "光照", "quality": "质量"}
            lines.append("--- 详细结果 ---")
            for cat, rules in report.by_category.items():
                for r in rules:
                    v = "PASS" if r.is_pass else ("WARN" if r.verdict.value == "warn" else "FAIL")
                    lines.append(f"  [{v}] {r.rule_id} {r.name}: {r.detail}")
                    if r.hint and not r.is_pass:
                        lines.append(f"       -> {r.hint}")

            compliance_text = "\n".join(lines)
            logs.append(f"合规检测: {report.overall_score:.0f}/100 ({time.time() - total_start:.1f}s)")

        total_time = time.time() - total_start
        log_text = "\n".join(logs + [f"总耗时: {total_time:.1f}s"])

        # 抠图预览
        matting_preview = cv2.cvtColor(matting, cv2.COLOR_BGRA2RGBA)

        return standard, hd, log_text, {
            "standard": standard,
            "hd": hd,
            "layout": layout,
            "matting": matting_preview,
            "compliance": compliance_text,
        }

    except Exception as e:
        import traceback
        return None, None, f"处理失败:\n{traceback.format_exc()}", None


# ============================================================
# Gradio UI
# ============================================================

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
)

with gr.Blocks(theme=THEME, title="AI一照成证") as demo:
    gr.Markdown("""
    # AI一照成证
    ### 智能证件照生成与优化系统 — 隐私保护 · 全本地处理
    > 基于 HivisionIDPhotos，自研P1合规检测引擎 + 级联边缘优化
    """)

    with gr.Row():
        # ---- 左侧：输入区 ----
        with gr.Column(scale=1):
            input_img = gr.Image(label="上传照片", type="numpy", height=350)

            with gr.Group():
                spec_dropdown = gr.Dropdown(
                    choices=list(SPECS.keys()),
                    value="一寸",
                    label="证件照规格",
                )
                color_radio = gr.Radio(
                    choices=["白色", "蓝色", "红色"],
                    value="蓝色",
                    label="背景颜色",
                )
                standard_dropdown = gr.Dropdown(
                    choices=["ISO", "中国", "US_VISA", "SCHENGEN", "JAPAN"],
                    value="ISO",
                    label="合规检测标准",
                )

            with gr.Row():
                refine_check = gr.Checkbox(value=True, label="P1 边缘优化")
                compliance_check = gr.Checkbox(value=True, label="P1 合规检测")

            generate_btn = gr.Button("生成证件照", variant="primary", size="lg")

            # 处理日志
            log_output = gr.Textbox(label="处理日志", lines=6, interactive=False)

        # ---- 右侧：输出区 ----
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("标准照"):
                    output_standard = gr.Image(label="标准证件照", height=350)
                with gr.TabItem("高清版"):
                    output_hd = gr.Image(label="高清版", height=350)
                with gr.TabItem("抠图预览"):
                    output_matting = gr.Image(label="抠图结果", height=350, visible=False)
                with gr.TabItem("排版"):
                    output_layout = gr.Image(label="冲印排版", height=350, visible=False)

            compliance_output = gr.Textbox(
                label="合规检测报告",
                lines=15,
                interactive=False,
                placeholder="上传照片后自动生成合规检测报告...",
            )

    # ---- 页脚 ----
    gr.Markdown("""
    ---
    **AI一照成证** | 隐私保护 · 全本地推理 · 无需上传云端
    """)

    # ---- 处理函数绑定 ----
    def on_generate(img, spec, color, standard, refine, compliance_check_flag):
        if img is None:
            return None, None, None, None, "请先上传照片", ""

        # 颜色名称映射
        color_map = {"白色": "white", "蓝色": "blue", "红色": "red"}
        color_en = color_map.get(color, "blue")

        std, hd, log_text, extras = process_idphoto(
            img, spec, color_en, standard, refine, compliance_check_flag
        )

        matting_img = extras.get("matting") if extras else None
        layout_img = extras.get("layout") if extras else None
        compliance = extras.get("compliance", "") if extras else ""

        return std, hd, matting_img, layout_img, log_text, compliance

    generate_btn.click(
        fn=on_generate,
        inputs=[input_img, spec_dropdown, color_radio, standard_dropdown,
                refine_check, compliance_check],
        outputs=[output_standard, output_hd, output_matting, output_layout,
                 log_output, compliance_output],
    )


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════╗
    ║   AI一照成证 — Web 交互界面              ║
    ╠══════════════════════════════════════════╣
    ║  启动后打开: http://127.0.0.1:7860       ║
    ║  停止: Ctrl+C                            ║
    ╚══════════════════════════════════════════╝
    """)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
