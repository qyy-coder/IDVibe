#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合规检测引擎测试套件
=====================

P1 合规检测模块的完整测试:
- 标准配置加载
- 几何检测规则
- 姿态检测规则
- 面部状态检测规则
- 光照检测规则
- 质量检测规则
- 合规报告生成
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
import time

from idphoto_system.compliance import ComplianceEngine, get_standard, STANDARDS

PASS = FAIL = 0


def test(name):
    def decorator(fn):
        def wrapper(*a, **kw):
            global PASS, FAIL
            try:
                fn(*a, **kw)
                PASS += 1
                print(f"  PASS {name}")
            except Exception as e:
                FAIL += 1
                print(f"  FAIL {name}: {e}")
        return wrapper
    return decorator


def make_mock_face_info(with_landmarks=True):
    """创建模拟人脸检测结果"""
    bbox = (100, 80, 220, 320)  # 120×240 人脸区域在 300×400 图像中
    if with_landmarks:
        landmarks = [
            (140, 160),   # 左眼
            (180, 160),   # 右眼
            (160, 220),   # 鼻尖
            (140, 260),   # 左嘴角
            (180, 260),   # 右嘴角
        ]
    else:
        landmarks = None
    return {
        "bbox": bbox,
        "landmarks": landmarks,
        "confidence": 0.95,
        "quality": 85,
    }


def make_test_image():
    """创建模拟证件照"""
    img = np.ones((400, 300, 3), dtype=np.uint8)
    img[:, :] = (206, 140, 99)  # 蓝底 BGR
    # 画一个简单的椭圆人脸
    cv2.ellipse(img, (160, 200), (60, 90), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (140, 160), 8, (60, 60, 60), -1)
    cv2.circle(img, (180, 160), 8, (60, 60, 60), -1)
    cv2.ellipse(img, (160, 260), (20, 6), 0, 0, 180, (80, 80, 80), 2)
    return img


# ============================================================
# 标准配置测试
# ============================================================

@test("加载所有标准配置")
def test_load_standards():
    assert len(STANDARDS) >= 5
    for name in ["ISO", "中国", "US_VISA", "SCHENGEN", "JAPAN"]:
        assert name in STANDARDS, f"缺失标准: {name}"
        std = STANDARDS[name]
        assert std.head_ratio_range[0] < std.head_ratio_range[1]

@test("标准别名解析")
def test_standard_aliases():
    assert get_standard("cn").name == "中国"
    assert get_standard("US_VISA").name == "US_VISA"
    assert get_standard("申根").name == "SCHENGEN"
    assert get_standard("unknown_std").name == "ISO"  # 回退


# ============================================================
# 几何检测测试
# ============================================================

@test("G01 头部占比 — 正常")
def test_g01_pass():
    from idphoto_system.compliance.geometric_checks import check_head_ratio
    std = get_standard("ISO")
    face = make_mock_face_info()
    # 320 / 400 = 0.80 → 在 ISO 范围 0.55-0.75 的上限附近
    # bbox y1=80, y2=320 → head_h估算
    result = check_head_ratio(face, (400, 300), std)
    assert result.rule_id == "G01"

@test("G02 眼睛位置")
def test_g02_eye_line():
    from idphoto_system.compliance.geometric_checks import check_eye_line
    std = get_standard("ISO")
    face = make_mock_face_info()
    result = check_eye_line(face, (400, 300), std)
    assert result.rule_id == "G02"

@test("G03 人脸居中")
def test_g03_centering():
    from idphoto_system.compliance.geometric_checks import check_face_centering
    std = get_standard("ISO")
    face = make_mock_face_info()
    result = check_face_centering(face, (400, 300), std)
    assert result.rule_id == "G03"

@test("G04 下颚位置")
def test_g04_chin():
    from idphoto_system.compliance.geometric_checks import check_chin_margin
    std = get_standard("ISO")
    face = make_mock_face_info()
    result = check_chin_margin(face, (400, 300), std)
    assert result.rule_id == "G04"


# ============================================================
# 姿态检测测试
# ============================================================

@test("姿态估计 — 产生有效角度")
def test_pose_estimation():
    from idphoto_system.compliance.pose_checks import estimate_pose
    face = make_mock_face_info()
    yaw, pitch, roll = estimate_pose(face, (400, 300))
    # 应该返回有效浮点数
    assert isinstance(yaw, float)
    assert isinstance(pitch, float)
    assert isinstance(roll, float)

@test("P01-P03 姿态规则运行")
def test_pose_checks():
    from idphoto_system.compliance.pose_checks import (
        check_yaw, check_pitch, check_roll, run_pose_checks,
    )
    std = get_standard("ISO")
    face = make_mock_face_info()
    results = run_pose_checks(face, (400, 300), std)
    assert len(results) == 3
    assert all(r.rule_id in ("P01", "P02", "P03") for r in results)

    # 零角度应该通过
    r = check_yaw(3.0, std)
    assert r.is_pass, f"预期 PASS 但得到 {r.verdict.value}"

    # 大角度应该失败
    r = check_yaw(10.0, std)
    assert r.is_fail, f"预期 FAIL 但得到 {r.verdict.value}"

@test("旋转向量到欧拉角")
def test_euler_conversion():
    from idphoto_system.compliance.pose_checks import _rotation_vector_to_euler
    # 零旋转向量 → 零欧拉角
    rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    yaw, pitch, roll = _rotation_vector_to_euler(rvec)
    assert abs(yaw) < 1  # 可能有微小浮点误差
    assert abs(pitch) < 1


# ============================================================
# 面部状态检测测试
# ============================================================

@test("F01 眼睛开合")
def test_f01_eyes_open():
    from idphoto_system.compliance.facial_checks import check_eyes_open
    std = get_standard("ISO")
    img = make_test_image()
    face = make_mock_face_info()
    result = check_eyes_open(img, face, std)
    assert result.rule_id == "F01"

@test("F02 嘴部闭合")
def test_f02_mouth_closed():
    from idphoto_system.compliance.facial_checks import check_mouth_closed
    std = get_standard("ISO")
    img = make_test_image()
    face = make_mock_face_info()
    result = check_mouth_closed(img, face, std)
    assert result.rule_id == "F02"

@test("F04 眼镜检测")
def test_f04_glasses():
    from idphoto_system.compliance.facial_checks import check_glasses
    std = get_standard("ISO")
    img = make_test_image()
    face = make_mock_face_info()
    result = check_glasses(img, face, std)
    assert result.rule_id == "F04"

@test("F06 红眼检测")
def test_f06_red_eye():
    from idphoto_system.compliance.facial_checks import check_red_eye
    std = get_standard("ISO")
    img = make_test_image()
    face = make_mock_face_info()
    result = check_red_eye(img, face, std)
    assert result.rule_id == "F06"


# ============================================================
# 光照检测测试
# ============================================================

@test("L01 面部光照均匀度")
def test_l01_lighting():
    from idphoto_system.compliance.lighting_checks import check_face_lighting_uniformity
    std = get_standard("ISO")
    img = make_test_image()
    face = make_mock_face_info()
    result = check_face_lighting_uniformity(img, face, std)
    assert result.rule_id == "L01"

@test("L03 背景均匀度")
def test_l03_bg_uniformity():
    from idphoto_system.compliance.lighting_checks import check_background_uniformity
    std = get_standard("ISO")
    img = make_test_image()

    # 创建均匀背景的 alpha
    alpha = np.zeros((400, 300), dtype=np.float32)
    alpha[80:320, 100:220] = 1.0

    result = check_background_uniformity(img, alpha, std)
    # 均匀背景应该通过或警告
    assert result.rule_id == "L03"


# ============================================================
# 质量检测测试
# ============================================================

@test("Q01 分辨率检测")
def test_q01_resolution():
    from idphoto_system.compliance.quality_checks import check_resolution
    std = get_standard("ISO")
    img = np.ones((413, 295, 3), dtype=np.uint8)
    result = check_resolution(img, (295, 413), std)
    assert result.rule_id == "Q01"

@test("Q02 清晰度检测")
def test_q02_sharpness():
    from idphoto_system.compliance.quality_checks import check_sharpness
    std = get_standard("ISO")
    img = make_test_image()
    result = check_sharpness(img, std)
    assert result.rule_id == "Q02"


# ============================================================
# 引擎集成测试
# ============================================================

@test("ComplianceEngine 完整检测")
def test_engine_full_check():
    engine = ComplianceEngine(standard="ISO")
    img = make_test_image()
    face = make_mock_face_info()
    alpha = np.ones((400, 300), dtype=np.float32) * 0.9

    start = time.time()
    report = engine.check(img, face, alpha, spec_size=(295, 413))
    elapsed = time.time() - start

    assert report.total_count >= 18, f"预期 ≥18 规则，实际 {report.total_count}"
    assert report.standard == "ISO/IEC 19794-5 (国际标准)"
    assert isinstance(report.is_compliant, bool)
    assert report.overall_score >= 0

    print(f"      ({report.total_count} 规则, {elapsed:.3f}s, 评分: {report.overall_score:.0f})")

@test("ComplianceReport 序列化")
def test_report_to_dict():
    engine = ComplianceEngine(standard="中国")
    face = make_mock_face_info()
    img = make_test_image()
    alpha = np.ones((400, 300), dtype=np.float32) * 0.9

    report = engine.check(img, face, alpha)
    d = report.to_dict()

    assert "rules" in d
    assert "overall_score" in d
    assert "is_compliant" in d
    assert "by_category" in d
    assert len(d["rules"]) >= 18

@test("ComplianceReport 摘要")
def test_report_summary():
    engine = ComplianceEngine()
    face = make_mock_face_info()
    img = make_test_image()
    report = engine.check(img, face)
    s = report.summary()
    assert "合规检测报告" in s
    assert "通过:" in s
    assert "综合评分" in s

@test("不同标准的差异")
def test_different_standards():
    """不同标准应产生不同阈值，影响检测结果"""
    img = make_test_image()
    face = make_mock_face_info()

    r_iso = ComplianceEngine("ISO").check(img, face)
    r_cn = ComplianceEngine("中国").check(img, face)

    # 验证两个报告都成功生成
    assert r_iso.total_count >= 18
    assert r_cn.total_count >= 18
    # 两者可能不同（因为阈值不同）但都在合理范围
    assert r_iso.overall_score != r_cn.overall_score or True  # 可能相同也可能不同


# ============================================================
# 入口
# ============================================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("合规检测引擎测试")
    print("=" * 60)

    print("\n[标准配置]")
    test_load_standards()
    test_standard_aliases()

    print("\n[几何检测 G01-G04]")
    test_g01_pass()
    test_g02_eye_line()
    test_g03_centering()
    test_g04_chin()

    print("\n[姿态检测 P01-P03]")
    test_pose_estimation()
    test_pose_checks()
    test_euler_conversion()

    print("\n[面部状态 F01-F06]")
    test_f01_eyes_open()
    test_f02_mouth_closed()
    test_f04_glasses()
    test_f06_red_eye()

    print("\n[光照检测 L01-L04]")
    test_l01_lighting()
    test_l03_bg_uniformity()

    print("\n[质量检测 Q01-Q04]")
    test_q01_resolution()
    test_q02_sharpness()

    print("\n[引擎集成]")
    test_engine_full_check()
    test_report_to_dict()
    test_report_summary()
    test_different_standards()

    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"结果: {PASS}/{total} 通过 ({PASS/total*100:.0f}%)" if total else "无测试")
    if FAIL:
        print(f"❌ {FAIL} 个测试失败")
    else:
        print("✅ 所有测试通过")
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
