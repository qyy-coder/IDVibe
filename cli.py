#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI一照成证 — 命令行工具
========================
用法:
    # 生成一寸蓝底证件照
    python cli.py --input photo.jpg --spec 一寸 --color blue

    # 生成二寸白底证件照 + 排版
    python cli.py --input photo.jpg --spec 二寸 --color white --layout 二寸×4

    # 查看可用规格和颜色
    python cli.py --list-specs
    python cli.py --list-colors
    python cli.py --list-layouts

    # 批量处理
    python cli.py --batch photos/*.jpg --spec 一寸 --color blue --output-dir results/
"""

import argparse
import sys
import os

# 确保项目根目录在 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from idphoto_system.pipeline import IDPhotoPipeline
from idphoto_system.utils.image_utils import SPECS, COLORS, COLOR_ALIASES
from idphoto_system.processing.layout import LayoutGenerator


def cmd_list_specs():
    """列出所有可用的证件照规格"""
    print("\n📐 可用证件照规格:\n")
    print(f"{'名称':<10} {'尺寸(像素)':<18} {'物理尺寸':<16} {'用途'}")
    print("-" * 70)
    for name, spec in SPECS.items():
        print(
            f"{spec.label:<10} "
            f"{spec.width}×{spec.height:<12} "
            f"{spec.mm_width}×{spec.mm_height}mm{'':<6} "
            f"{spec.usage}"
        )
    print()


def cmd_list_colors():
    """列出所有可用背景色"""
    print("\n🎨 可用背景色:\n")
    print(f"{'名称':<15} {'RGB 值':<20} {'Hex 码'}")
    print("-" * 50)
    shown = set()
    for alias, key in COLOR_ALIASES.items():
        if key not in shown:
            color = COLORS[key]
            print(f"{alias:<15} ({color[0]}, {color[1]}, {color[2]}){'':<8} #{color[0]:02x}{color[1]:02x}{color[2]:02x}")
            shown.add(key)
    print()


def cmd_list_layouts():
    """列出所有可用排版"""
    print("\n🖼️  可用排版方案:\n")
    layouts = LayoutGenerator.STANDARD_LAYOUTS
    for name, cfg in layouts.items():
        print(f"  {name:<12} → {cfg['rows']}行 × {cfg['cols']}列 | 规格: {cfg['spec']}")
    print()


def cmd_process(args):
    """处理单张照片"""
    # 验证输入
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在 — {args.input}")
        sys.exit(1)

    # 初始化流水线
    print("\n🔧 初始化证件照生成流水线...")
    pipeline = IDPhotoPipeline(
        matting_model=args.matting_model,
        face_model=args.face_model,
        enable_layout=bool(args.layout),
    )

    # 处理
    print(f"\n📸 开始处理: {args.input}")
    print(f"   规格: {args.spec} | 背景: {args.color}")

    result = pipeline.process(
        args.input,
        spec=args.spec,
        color=args.color,
        bg_mode=args.bg_mode,
        layout=args.layout,
        output_path=args.output,
        dpi=args.dpi,
    )

    if result["status"] != "ok":
        print(f"\n❌ 处理失败: {result['error']}")
        sys.exit(1)

    print(f"\n✅ 处理完成!")
    print(f"   总耗时: {result['timing']['total']:.2f}s")
    if args.output:
        print(f"   输出文件: {args.output}")
    else:
        print(f"   (未指定输出路径，结果在内存中)")


def cmd_batch(args):
    """批量处理"""
    import glob

    # 查找文件
    files = glob.glob(args.batch)
    if not files:
        print(f"❌ 未找到匹配的文件: {args.batch}")
        sys.exit(1)

    print(f"\n📂 找到 {len(files)} 个文件")

    # 确保输出目录存在
    output_dir = args.output_dir or "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化流水线
    pipeline = IDPhotoPipeline(
        matting_model=args.matting_model,
        face_model=args.face_model,
    )

    # 批量处理
    results = pipeline.batch_process(
        files,
        spec=args.spec,
        color=args.color,
        output_dir=output_dir,
    )

    # 统计
    success = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - success
    total_time = sum(r.get("timing", {}).get("total", 0) for r in results)

    print(f"\n{'='*50}")
    print(f"📊 批量处理统计")
    print(f"   总数: {len(files)}")
    print(f"   成功: {success}")
    print(f"   失败: {failed}")
    print(f"   总耗时: {total_time:.2f}s")
    print(f"   输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="AI一照成证 — 智能证件照生成系统 (P0 v0.1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py --input photo.jpg --spec 一寸 --color blue
  python cli.py --input photo.jpg --spec 二寸 --color white --layout 二寸×4
  python cli.py --batch "photos/*.jpg" --spec 一寸 --color blue
        """,
    )

    # 操作子命令
    parser.add_argument("--list-specs", action="store_true",
                        help="列出所有可用的证件照规格")
    parser.add_argument("--list-colors", action="store_true",
                        help="列出所有可用的背景色")
    parser.add_argument("--list-layouts", action="store_true",
                        help="列出所有可用的排版方案")

    # 主要参数
    parser.add_argument("-i", "--input", help="输入图像路径")
    parser.add_argument("-o", "--output", help="输出图像路径")
    parser.add_argument("--spec", default="一寸",
                        help="证件照规格 (默认: 一寸)")
    parser.add_argument("--color", default="white",
                        help="背景色 (默认: white)")
    parser.add_argument("--bg-mode", default="pure_color",
                        choices=["pure_color", "updown_gradient", "center_gradient"],
                        help="背景模式 (默认: pure_color)")
    parser.add_argument("--layout", default=None,
                        help="排版方案，如 '一寸×8'")
    parser.add_argument("--dpi", type=int, default=300,
                        help="输出 DPI (默认: 300)")

    # 模型选择
    parser.add_argument("--matting-model", default="modnet_photographic_portrait_matting",
                        help="抠图模型")
    parser.add_argument("--face-model", default="mtcnn",
                        choices=["mtcnn", "retinaface"],
                        help="人脸检测模型 (默认: mtcnn)")

    # 批量处理
    parser.add_argument("--batch", help="批量处理通配符，如 'photos/*.jpg'")
    parser.add_argument("--output-dir", default="outputs",
                        help="批量处理输出目录 (默认: outputs)")

    args = parser.parse_args()

    # 信息查询
    if args.list_specs:
        cmd_list_specs()
        return
    if args.list_colors:
        cmd_list_colors()
        return
    if args.list_layouts:
        cmd_list_layouts()
        return

    # 批量处理
    if args.batch:
        cmd_batch(args)
        return

    # 单张处理
    if args.input:
        cmd_process(args)
        return

    # 无操作时显示帮助
    parser.print_help()
    print("\n💡 使用 --list-specs / --list-colors / --list-layouts 查看更多信息")


if __name__ == "__main__":
    main()
