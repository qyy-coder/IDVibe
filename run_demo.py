#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI一照成证 — 完整产品演示
使用 HivisionIDPhotos 原生流水线 (精确居中) + P1 增强
"""

import os, sys, time, argparse, datetime

HIVISION_PATH = r"C:\Users\24817\HivisionIDPhotos"
sys.path.insert(0, HIVISION_PATH)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_PATH)

import numpy as np
import cv2

from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from hivision.creator.photo_adjuster import adjust_photo
from hivision.creator.context import Context, Params
from hivision.utils import hex_to_rgb, add_background

from idphoto_system.compliance import ComplianceEngine
from idphoto_system.matting import cascade_refine
from idphoto_system.utils.image_utils import resolve_spec, resolve_color
from idphoto_system.processing.layout import LayoutGenerator


def run(photo_path, spec_name="一寸", color_name="white", standard_name="ISO"):
    total_start = time.time()
    timing = {}

    # 解析参数
    spec = resolve_spec(spec_name)
    rgb = resolve_color(color_name)
    bgr = (rgb[2], rgb[1], rgb[0])

    # --- [1] 加载 ---
    t0 = time.time()
    img = cv2.imread(photo_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {photo_path}")
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        s = 2000 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
    timing["load"] = time.time() - t0
    print(f"[1] 加载: {img.shape[1]}x{img.shape[0]} ({timing['load']*1000:.0f}ms)")

    # --- [2] HivisionIDPhotos 原生流水线: 人脸检测 + 抠图 + 精确居中裁剪 ---
    t0 = time.time()
    creator = IDCreator()
    choose_handler(creator, "hivision_modnet", "mtcnn")

    # 使用完整的 HivisionIDPhotos 流程 (精确的头部定位算法)
    result = creator(
        img,
        size=spec.size,
        change_bg_only=False,
        head_measure_ratio=0.23,
        head_height_ratio=0.45,
        head_top_range=(0.12, 0.1),
    )
    timing["hivision"] = time.time() - t0

    matting_rgba = result.matting
    alpha = matting_rgba[:, :, 3].astype(np.float32) / 255.0
    face_info = result.face or {}

    print(f"[2] HivisionIDPhotos: {timing['hivision']*1000:.0f}ms | "
          f"抠图+检测+精确居中")

    # --- [3] P1 级联边缘优化 ---
    t0 = time.time()
    try:
        refined_alpha, mq = cascade_refine(img, alpha, radius=6, eps=1e-6)
        alpha = refined_alpha
        timing["edge_refine"] = time.time() - t0
        print(f"[3] P1边缘优化: {mq.confidence:.0f}/100 ({mq.tier}) | "
              f"{timing['edge_refine']*1000:.0f}ms")
    except Exception as e:
        mq = None
        timing["edge_refine"] = time.time() - t0
        print(f"[3] P1边缘优化: 跳过 ({e})")

    # --- [4] 背景替换 + 重新精确裁剪 ---
    t0 = time.time()
    rgba = np.dstack([matting_rgba[:,:,:3], (alpha*255).astype(np.uint8)])
    composited = add_background(rgba, bgr=bgr, mode="pure_color").astype(np.uint8)

    # 用合成图替换抠图结果，重新执行精确居中裁剪
    # crop_only 模式需要 4 通道 BGRA 输入
    composited_bgra = cv2.cvtColor(composited, cv2.COLOR_BGR2BGRA)

    creator2 = IDCreator()
    choose_handler(creator2, "hivision_modnet", "mtcnn")
    result2 = creator2(
        composited_bgra,
        size=spec.size,
        crop_only=True,
        head_measure_ratio=0.23,
        head_height_ratio=0.45,
        head_top_range=(0.12, 0.1),
    )
    result_standard = result2.standard  # BGRA
    result_hd = result2.hd              # BGRA
    timing["bg_crop"] = time.time() - t0
    print(f"[4] 背景+裁剪: {color_name} {spec.label} ({timing['bg_crop']*1000:.0f}ms)")

    # --- [6] P1 合规检测 ---
    t0 = time.time()
    raw_face = result2.face or {}
    face_for_compliance = {"bbox": None, "landmarks": None, "confidence": 1.0}
    rect = raw_face.get("rectangle")
    if rect and len(rect) == 4:
        cx, cy, fw, fh = rect
        face_for_compliance["bbox"] = (int(cx-fw/2), int(cy-fh/2), int(cx+fw/2), int(cy+fh/2))
    face_for_compliance["roll_angle"] = raw_face.get("roll_angle", 0)

    engine = ComplianceEngine(standard=standard_name)
    report = engine.check(
        img, face_for_compliance, alpha,
        spec_size=(spec.width, spec.height),
        expected_bg_color=bgr,
    )
    timing["compliance"] = time.time() - t0
    status = "[PASS]" if report.is_compliant else "[FAIL]"
    print(f"[6] P1合规检测: {report.overall_score:.0f}/100 {status} | "
          f"{report.passed_count}P/{report.failed_count}F/{report.warn_count}W "
          f"({timing['compliance']*1000:.0f}ms)")
    if report.critical_failures:
        for cf in report.critical_failures:
            print(f"    ! {cf.rule_id} {cf.name} -> {cf.hint}")

    for cat, rules in report.by_category.items():
        labels = {"geometric":"几何","pose":"姿态","facial":"面部","lighting":"光照","quality":"质量"}
        fails = sum(1 for r in rules if not r.is_pass)
        icon = "[!!]" if fails else "[OK]"
        parts = [f"{r.rule_id}={r.verdict.value.upper()}" for r in rules]
        print(f"    {icon} {labels.get(cat,cat)}: {', '.join(parts)}")

    # --- [7] 保存 ---
    t0 = time.time()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(output_dir, f"idphoto_{spec_name}_{color_name}_{standard_name}_{ts}")

    # HivisionIDPhotos 输出为 BGRA，转 BGR 保存
    std_bgr = cv2.cvtColor(result_standard, cv2.COLOR_BGRA2BGR)
    hd_bgr = cv2.cvtColor(result_hd, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(f"{prefix}.png", std_bgr)
    cv2.imwrite(f"{prefix}_hd.png", hd_bgr)

    try:
        gen = LayoutGenerator()
        lname = f"{spec_name}×{'8' if '一寸' in spec_name else '4'}"
        layout = gen.generate_standard(result_standard, lname, spec)
        cv2.imwrite(f"{prefix}_layout.png", layout)
    except: pass

    with open(f"{prefix}_compliance.txt", "w", encoding="utf-8") as f:
        f.write(report.summary())

    timing["save"] = time.time() - t0
    total = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  完成! 总耗时: {total:.1f}s")
    print(f"  标准照: {prefix}.png ({spec.width}x{spec.height})")
    print(f"  高清版: {prefix}_hd.png ({spec.width*2}x{spec.height*2})")
    print(f"  排版:   {prefix}_layout.png")
    print(f"  报告:   {prefix}_compliance.txt")
    mq_str = f"{mq.confidence:.0f}/100" if mq else "N/A"
    print(f"  评分: {report.overall_score:.0f}/100 | 抠图: {mq_str}")
    print(f"{'='*60}\n")
    print(report.summary())
    return True


def main():
    parser = argparse.ArgumentParser(description="AI一照成证 — 成品演示")
    parser.add_argument("--image", required=True)
    parser.add_argument("--spec", default="一寸")
    parser.add_argument("--color", default="blue")
    parser.add_argument("--standard", default="ISO")
    args = parser.parse_args()
    run(args.image, args.spec, args.color, args.standard)


if __name__ == "__main__":
    main()
