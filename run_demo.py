#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI一照成证 — 完整产品演示
使用 HivisionIDPhotos 原生流水线 + P1 增强模块
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np
import cv2

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_PATH)

from idphoto_system.config import HIVISION_PATH
from idphoto_system.processing_service import PhotoProcessingService, ProcessRequest
from idphoto_system.utils.image_utils import resolve_spec, resolve_color
from idphoto_system.processing.layout import LayoutGenerator


def run(photo_path, spec_name="一寸", color_name="white", standard_name="ISO"):
    total_start = time.time()

    # ── [1] 加载 ──
    t0 = time.time()
    img = cv2.imread(photo_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {photo_path}")
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        s = 2000 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
    print(f"[1] 加载: {img.shape[1]}x{img.shape[0]} ({time.time()-t0:.3f}s)")

    # ── [2-6] 核心处理 (统一使用 PhotoProcessingService) ──
    service = PhotoProcessingService(hivision_path=HIVISION_PATH)

    req = ProcessRequest(
        image=img,
        spec=spec_name,
        color=color_name,
        bg_mode="pure_color",
        enable_edge_refine=True,
        enable_compliance=True,
        compliance_standard=standard_name,
        compliance_mode="full",      # 演示用完整版 22+ 规则
    )

    result = service.process(req)

    # 打印阶段耗时
    for stage, elapsed in result.timing.items():
        if stage != "total":
            print(f"    [{stage}] {elapsed:.3f}s")

    # ── 打印抠图质量 ──
    mq = result.matting_quality
    if mq:
        print(f"[P1边缘优化]: 置信度 {mq['confidence']:.0f}/100 ({mq['tier']})")
    else:
        print(f"[P1边缘优化]: 跳过")

    # ── 打印合规报告 ──
    if result.checks:
        report = result.checks
        print(f"[合规检测]: {report['overall_score']:.0f}/100 "
              f"{'[PASS]' if report['is_compliant'] else '[FAIL]'} | "
              f"{report['passed_count']}P/{report['failed_count']}F/{report['warn_count']}W "
              f"({result.timing.get('compliance', 0):.3f}s)")
        if report.get("critical_failures"):
            for cf in report["critical_failures"]:
                print(f"    ! {cf['rule_id']} {cf['name']} -> {cf.get('hint', '')}")

        for cat, stats in report.get("by_category", {}).items():
            labels = {"geometric":"几何", "pose":"姿态", "facial":"面部",
                      "lighting":"光照", "quality":"质量"}
            icon = "[!!]" if stats["failed"] else "[OK]"
            print(f"    {icon} {labels.get(cat, cat)}: {stats['passed']}P/{stats['failed']}F")

    # ── [7] 保存 ──
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(output_dir, f"idphoto_{spec_name}_{color_name}_{standard_name}_{ts}")

    cv2.imwrite(f"{prefix}.png", result.standard)

    # 高清版
    spec = resolve_spec(spec_name)
    hd = cv2.resize(result.standard,
                    (spec.width * 2, spec.height * 2),
                    interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f"{prefix}_hd.png", hd)

    # 排版
    try:
        gen = LayoutGenerator()
        lname = f"{spec_name}×{'8' if '一寸' in spec_name else '4'}"
        standard_bgra = cv2.cvtColor(result.standard, cv2.COLOR_BGR2BGRA)
        layout = gen.generate_standard(standard_bgra, lname, spec)
        cv2.imwrite(f"{prefix}_layout.png", layout)
    except Exception:
        pass

    # 合规报告文本
    from idphoto_system.compliance.engine import ComplianceEngine
    from idphoto_system.compliance import get_standard
    cf = PhotoProcessingService._convert_face_for_compliance(result.face_info or {})
    engine = ComplianceEngine(standard=standard_name)
    report_obj = engine.check(
        img, cf, result.alpha,
        spec_size=(spec.width, spec.height),
        expected_bg_color=(resolve_color(color_name)[2], resolve_color(color_name)[1], resolve_color(color_name)[0]),
    )
    with open(f"{prefix}_compliance.txt", "w", encoding="utf-8") as f:
        f.write(report_obj.summary())

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  完成! 总耗时: {total:.1f}s")
    print(f"  标准照: {prefix}.png ({result.spec_width}x{result.spec_height})")
    print(f"  高清版: {prefix}_hd.png")
    print(f"  排版:   {prefix}_layout.png")
    print(f"  报告:   {prefix}_compliance.txt")
    mq_str = f"{mq['confidence']:.0f}/100" if mq else "N/A"
    print(f"  评分: {report_obj.overall_score:.0f}/100 | 抠图: {mq_str}")
    print(f"{'='*60}\n")
    print(report_obj.summary())
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
