#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI一照成证 — 流水线测试
=========================

P0 基础测试套件:
- 模块导入测试
- 图像工具测试
- 流水线端到端测试 (需要模型)
- 性能基准测试

用法:
    # 运行所有测试
    python tests/test_pipeline.py

    # 跳过需要模型的测试
    python tests/test_pipeline.py --skip-model

    # 性能基准测试
    python tests/test_pipeline.py --benchmark
"""

import os
import sys
import argparse
import time
import tempfile

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np


# ============================================================
# 测试工具
# ============================================================

def create_test_image(width=800, height=600, bg_color=(200, 200, 200)):
    """
    生成测试图像：灰色背景 + 简单的椭圆"人脸"。

    注意：此图像不含真实人脸，仅用于模块功能测试。
    如需端到端测试，请提供真实照片。
    """
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = bg_color[0]  # B
    img[:, :, 1] = bg_color[1]  # G
    img[:, :, 2] = bg_color[2]  # R

    # 画一个椭圆模拟人脸
    center = (width // 2, height // 2 - 40)
    axes = (120, 160)
    cv2.ellipse(img, center, axes, 0, 0, 360, (210, 180, 140), -1)

    # 画两个眼睛
    cv2.circle(img, (center[0] - 40, center[1] - 30), 12, (50, 50, 50), -1)
    cv2.circle(img, (center[0] + 40, center[1] - 30), 12, (50, 50, 50), -1)

    # 画嘴巴
    cv2.ellipse(img, (center[0], center[1] + 50), (30, 10), 0, 0, 180, (80, 80, 80), 2)

    return img


PASS = 0
FAIL = 0

def test(name):
    """测试装饰器"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global PASS, FAIL
            try:
                fn(*args, **kwargs)
                PASS += 1
                print(f"  ✅ {name}")
            except Exception as e:
                FAIL += 1
                print(f"  ❌ {name}: {e}")
        return wrapper
    return decorator


# ============================================================
# 模块导入测试
# ============================================================

@test("导入 idphoto_system 包")
def test_import_package():
    import idphoto_system
    assert idphoto_system.__version__ == "0.2.0"

@test("导入工具模块")
def test_import_utils():
    from idphoto_system.utils.image_utils import (
        PhotoSpec, SPECS, COLORS, resolve_color, resolve_spec,
    )
    assert len(SPECS) >= 5
    assert len(COLORS) >= 3

@test("导入流水线模块（不加载模型）")
def test_import_pipeline():
    from idphoto_system.pipeline.idphoto_pipeline import IDPhotoPipeline
    assert IDPhotoPipeline is not None

@test("导入CLI模块")
def test_import_cli():
    # 验证 cli.py 存在且语法正确
    cli_path = os.path.join(PROJECT_ROOT, "cli.py")
    assert os.path.exists(cli_path), "cli.py not found"

@test("导入API模块")
def test_import_api():
    api_path = os.path.join(PROJECT_ROOT, "api_server.py")
    assert os.path.exists(api_path), "api_server.py not found"


# ============================================================
# 图像工具测试
# ============================================================

@test("PhotoSpec 数据结构")
def test_photo_spec():
    from idphoto_system.utils.image_utils import PhotoSpec, SPECS

    spec = SPECS["一寸"]
    assert spec.width == 295
    assert spec.height == 413
    assert spec.size == (413, 295)
    assert spec.name == "一寸"

@test("背景色解析")
def test_resolve_color():
    from idphoto_system.utils.image_utils import resolve_color

    # 英文名
    assert resolve_color("white") == (255, 255, 255)
    assert resolve_color("blue") == (99, 140, 206)
    assert resolve_color("red") == (206, 53, 53)

    # 中文别名
    assert resolve_color("白") == (255, 255, 255)
    assert resolve_color("蓝色") == (99, 140, 206)

    # Hex 码
    result = resolve_color("#638cce")
    assert result == (99, 140, 206)

@test("规格解析")
def test_resolve_spec():
    from idphoto_system.utils.image_utils import resolve_spec

    # 精确匹配
    spec = resolve_spec("一寸")
    assert spec.name == "一寸"
    assert spec.width == 295

    # 模糊匹配
    spec = resolve_spec("二寸")
    assert spec.name == "二寸"

@test("Base64 编解码")
def test_base64():
    from idphoto_system.utils.image_utils import array_to_base64, base64_to_array

    img = create_test_image(200, 150)
    b64 = array_to_base64(img, fmt=".png")
    assert b64.startswith("data:image/png;base64,")

    decoded = base64_to_array(b64)
    assert decoded.shape == img.shape

@test("图像加载与保存")
def test_image_load_save():
    from idphoto_system.utils.image_utils import load_image, save_image

    # 创建临时文件
    img = create_test_image(300, 200)
    import uuid
    tmp_name = os.path.join(tempfile.gettempdir(), f"idphoto_test_{uuid.uuid4().hex}.png")
    save_name = os.path.join(tempfile.gettempdir(), f"idphoto_test_{uuid.uuid4().hex}_saved.png")

    try:
        cv2.imwrite(tmp_name, img)

        # 加载
        loaded = load_image(tmp_name)
        assert loaded.shape == img.shape

        # 保存
        save_image(loaded, save_name, dpi=300)
        assert os.path.exists(save_name)
    finally:
        # 清理
        for p in [tmp_name, save_name]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


# ============================================================
# 处理模块测试 (无需模型)
# ============================================================

@test("BackgroundReplacer 手动合成")
def test_background_replacer():
    from idphoto_system.processing.background import BackgroundReplacer

    # 创建前景和alpha
    fg = np.ones((100, 100, 3), dtype=np.uint8) * 128
    alpha = np.ones((100, 100), dtype=np.float32) * 0.5

    result = BackgroundReplacer.composite_manual(
        fg, alpha, background_color=(255, 255, 255)
    )
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8

@test("PhotoCropper 中心裁剪")
def test_center_crop():
    from idphoto_system.processing.cropping import PhotoCropper
    from idphoto_system.utils.image_utils import SPECS

    cropper = PhotoCropper()
    img = create_test_image(800, 800)
    spec = SPECS["一寸"]

    # 无人脸信息时应使用中心裁剪（不抛异常）
    result = cropper.crop(img, {"bbox": None}, spec)
    assert result.shape[0] == spec.height
    assert result.shape[1] == spec.width

@test("LayoutGenerator 排版生成")
def test_layout_generator():
    from idphoto_system.processing.layout import LayoutGenerator
    from idphoto_system.utils.image_utils import SPECS

    gen = LayoutGenerator(margin=4, border=10)
    spec = SPECS["一寸"]

    # 创建模拟证件照
    photo = create_test_image(spec.width, spec.height)

    # 生成 4×2 排版
    layout = gen.generate(photo, spec, rows=4, cols=2)
    assert layout.shape[0] > spec.height
    assert layout.shape[1] > spec.width

    # 验证标准排版方案
    layouts = LayoutGenerator.get_available_layouts(for_spec="一寸")
    assert "一寸×8" in layouts


# ============================================================
# 性能基准测试
# ============================================================

@test("性能基准 — 图像预处理")
def test_benchmark_preprocess():
    from idphoto_system.utils.image_utils import load_image

    img = create_test_image(4000, 3000)  # 12MP 模拟照片
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)

        start = time.time()
        loaded = load_image(f.name, max_size=2000)
        elapsed = time.time() - start

        # 验证缩放
        h, w = loaded.shape[:2]
        assert max(h, w) <= 2000

        print(f"      ({elapsed*1000:.1f}ms, {loaded.shape[1]}×{loaded.shape[0]})")
        os.unlink(f.name)


# ============================================================
# 主入口
# ============================================================

def run_model_tests():
    """需要模型的端到端测试（仅在有真实照片时运行）"""
    from idphoto_system.pipeline import IDPhotoPipeline

    # 查找测试照片
    test_photos = []
    test_dir = os.path.join(PROJECT_ROOT, "tests", "test_photos")
    if os.path.exists(test_dir):
        test_photos = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if not test_photos:
        print("\n  ⚠️  未找到测试照片，跳过端到端测试")
        print(f"  请将测试照片放入: {test_dir}")
        return

    print(f"\n  📸 找到 {len(test_photos)} 张测试照片")

    pipeline = IDPhotoPipeline()

    for photo_path in test_photos[:3]:  # 最多测试3张
        print(f"\n  处理: {os.path.basename(photo_path)}")
        try:
            result = pipeline.process(
                photo_path,
                spec="一寸",
                color="white",
                output_path=photo_path.replace(".", "_output."),
            )
            if result["status"] == "ok":
                print(f"    ✅ 成功 | 总耗时: {result['timing']['total']:.2f}s")
            else:
                print(f"    ⚠️  {result['error']}")
        except Exception as e:
            print(f"    ❌ 异常: {e}")


def main():
    global PASS, FAIL

    parser = argparse.ArgumentParser(description="AI一照成证 测试套件")
    parser.add_argument("--skip-model", action="store_true",
                        help="跳过需要模型的测试")
    parser.add_argument("--benchmark", action="store_true",
                        help="运行性能基准测试")
    args = parser.parse_args()

    # 确保在项目环境中
    print("""
╔══════════════════════════════════════════╗
║   AI一照成证 — 测试套件 v0.1.0          ║
╠══════════════════════════════════════════╣
║  项目路径: {0}
╚══════════════════════════════════════════╝
""".format(PROJECT_ROOT.strip()[:50]))

    print("📦 模块导入测试")
    test_import_package()
    test_import_utils()
    test_import_pipeline()
    test_import_cli()
    test_import_api()

    print("\n🛠️  图像工具测试")
    test_photo_spec()
    test_resolve_color()
    test_resolve_spec()
    test_base64()
    test_image_load_save()

    print("\n⚙️  处理模块测试")
    test_background_replacer()
    test_center_crop()
    test_layout_generator()

    if args.benchmark:
        print("\n⏱️  性能基准测试")
        test_benchmark_preprocess()

    if not args.skip_model:
        print("\n🤖 端到端测试 (需模型 + 真实照片)")
        run_model_tests()

    # 摘要
    print("\n" + "=" * 50)
    total = PASS + FAIL
    if total > 0:
        pass_rate = PASS / total * 100
        print(f"📊 测试结果: {PASS} 通过 / {FAIL} 失败 ({pass_rate:.0f}%)")
    else:
        print("📊 没有测试运行")

    if FAIL > 0:
        print(f"\n❌ {FAIL} 个测试失败!")
        sys.exit(1)
    else:
        print("\n✅ 所有测试通过!")


if __name__ == "__main__":
    main()
