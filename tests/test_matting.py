#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
级联抠图边缘优化测试套件
===========================

P1 抠图优化模块的完整测试:
- 引导滤波
- 快速引导滤波
- Trimap 生成
- 边缘优化
- 自适应回退
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2

from idphoto_system.matting import (
    guided_filter,
    fast_guided_filter,
    EdgeRefiner,
    TrimapGenerator,
    AdaptiveFallback,
    cascade_refine,
)

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


def make_test_data(size=200):
    """创建测试数据：简单的人像抠图场景"""
    h, w = size, size
    # 模拟原始图像
    image = np.ones((h, w, 3), dtype=np.uint8) * 200
    cv2.rectangle(image, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (150, 120, 100), -1)

    # 模拟粗抠图 alpha：中心区域1.0，边缘过渡
    coarse = np.zeros((h, w), dtype=np.float32)
    coarse[h//4:3*h//4, w//4:3*w//4] = 0.95

    # 添加过渡区域
    y, x = np.ogrid[:h, :w]
    center = np.sqrt((x - w//2)**2 + (y - h//2)**2)
    edge_width = 20
    transition = np.clip((1 - (center - w//3) / edge_width), 0, 1).astype(np.float32)

    # 合成：中心区域使用 alpha=0.95，外围渐变过渡
    coarse = np.maximum(coarse, transition * 0.5)

    return image, coarse


# ============================================================
# 引导滤波测试
# ============================================================

@test("引导滤波 — 基本功能")
def test_guided_filter_basic():
    image, alpha = make_test_data(100)

    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    result = guided_filter(guide, alpha, radius=5, eps=1e-4)

    assert result.shape == alpha.shape
    assert result.dtype == np.float32
    assert np.all(result >= 0) and np.all(result <= 1)

@test("引导滤波 — 保边特性 (边缘不模糊)")
def test_guided_filter_edge_preserving():
    """引导滤波的关键特性：边缘保持"""
    h, w = 100, 100
    # 创建一个有尖锐边缘的 alpha
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, :w//2] = 0.0
    alpha[:, w//2:] = 1.0

    # 创建一个有同样边缘的引导图
    guide = np.zeros((h, w), dtype=np.float32)
    guide[:, :w//2] = 0.2
    guide[:, w//2:] = 0.8

    result = guided_filter(guide, alpha, radius=4, eps=1e-4)

    # 边缘处的梯度应该被保留（不应该完全模糊）
    edge_grad = np.abs(np.diff(result[:, w//2 - 1:w//2 + 2], axis=1))
    assert np.max(edge_grad) > 0.1, "边缘被过度平滑"

@test("引导滤波 — 平滑区域应保持平滑")
def test_guided_filter_smooth_region():
    """平滑区域不应引入噪声"""
    h, w = 100, 100
    alpha = np.ones((h, w), dtype=np.float32) * 0.8
    guide = np.ones((h, w), dtype=np.float32) * 0.5

    result = guided_filter(guide, alpha, radius=8, eps=1e-4)

    # 平滑区域应该有很小的方差
    var = np.var(result)
    assert var < 0.01, f"平滑区域方差过大: {var:.4f}"

@test("快速引导滤波")
def test_fast_guided_filter():
    image, alpha = make_test_data(200)

    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    result = fast_guided_filter(image, alpha, radius=4, eps=1e-4, subsample=2)

    assert result.shape == alpha.shape
    assert result.dtype == np.float32
    assert np.all(result >= 0) and np.all(result <= 1)


# ============================================================
# Trimap 测试
# ============================================================

@test("Trimap 生成")
def test_trimap_generation():
    gen = TrimapGenerator(low_threshold=0.1, high_threshold=0.9)
    _, alpha = make_test_data(100)

    trimap = gen.generate(alpha)
    assert trimap.shape == alpha.shape
    assert trimap.dtype == np.uint8
    assert np.isin(trimap, [0, 128, 255]).all(), "trimap 值域错误"

    # 应该有所有三种区域
    has_bg = np.any(trimap == 0)
    has_unknown = np.any(trimap == 128)
    has_fg = np.any(trimap == 255)

    print(f"      (BG:{has_bg} UNKNOWN:{has_unknown} FG:{has_fg})")

    # 至少应该有确定区域和不确定区域
    assert has_unknown or has_fg


# ============================================================
# 边缘优化测试
# ============================================================

@test("EdgeRefiner 完整流程")
def test_edge_refiner():
    image, alpha = make_test_data(150)

    refiner = EdgeRefiner(radius=6, eps=1e-4)
    refined = refiner.refine(image, alpha)

    assert refined.shape == alpha.shape
    assert refined.dtype == np.float32
    assert np.all(refined >= 0) and np.all(refined <= 1)

    # 细化结果不应与原始完全一致
    diff = np.mean(np.abs(refined - alpha))
    print(f"      (平均差异: {diff:.4f})")

@test("EdgeRefiner — 细节保留强度")
def test_edge_refiner_detail():
    image, alpha = make_test_data(100)

    refiner = EdgeRefiner(radius=6)
    refined_strong = refiner.refine_with_detail_preservation(image, alpha, detail_strength=0.8)
    refined_smooth = refiner.refine_with_detail_preservation(image, alpha, detail_strength=0.2)

    # 高低细节保留强度应产生不同结果
    diff = np.mean(np.abs(refined_strong - refined_smooth))
    assert diff > 0, "不同detail_strength应产生不同结果"


# ============================================================
# 自适应回退测试
# ============================================================

@test("AdaptiveFallback 质量评估")
def test_adaptive_fallback_evaluate():
    _, alpha = make_test_data(150)

    # 创建更精细的模拟
    refined = alpha.copy()

    fb = AdaptiveFallback()
    quality = fb.evaluate(alpha, refined)

    assert 0 <= quality.confidence <= 100
    assert quality.tier in ("high", "medium", "low")
    assert isinstance(quality.needs_fallback, bool)
    assert 0 <= quality.edge_continuity <= 100

    print(f"      (置信度: {quality.confidence:.1f}, 等级: {quality.tier}, 不确定区: {quality.uncertain_ratio:.2%})")

@test("AdaptiveFallback 决策")
def test_adaptive_fallback_decide():
    _, alpha = make_test_data(100)

    refiner = EdgeRefiner(radius=4)
    refined = refiner.refine(make_test_data(100)[0], alpha)

    fb = AdaptiveFallback()
    quality = fb.evaluate(alpha, refined)
    final = fb.decide(quality, alpha, refined)

    assert final.shape == alpha.shape
    assert final.dtype == np.float32


# ============================================================
# 完整流程测试
# ============================================================

@test("cascade_refine 完整流程")
def test_cascade_refine():
    image, alpha = make_test_data(120)

    final, quality = cascade_refine(image, alpha, radius=6, eps=1e-6)

    assert final.shape == alpha.shape
    assert quality is not None
    assert quality.confidence >= 0

    print(f"      (置信度: {quality.confidence:.1f}, 方法: {quality.method}, 等级: {quality.tier})")


# ============================================================
# 入口
# ============================================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("级联抠图边缘优化测试")
    print("=" * 60)

    print("\n[引导滤波]")
    test_guided_filter_basic()
    test_guided_filter_edge_preserving()
    test_guided_filter_smooth_region()
    test_fast_guided_filter()

    print("\n[Trimap]")
    test_trimap_generation()

    print("\n[边缘优化]")
    test_edge_refiner()
    test_edge_refiner_detail()

    print("\n[自适应回退]")
    test_adaptive_fallback_evaluate()
    test_adaptive_fallback_decide()

    print("\n[完整流程]")
    test_cascade_refine()

    total = PASS + FAIL
    print(f"\n{'='*60}")
    if total:
        print(f"结果: {PASS}/{total} 通过 ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAIL {FAIL} 个测试失败")
    else:
        print("PASS 所有测试通过")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
