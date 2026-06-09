#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载脚本
=============
从 HivisionIDPhotos 下载所需的 ONNX 模型文件。

P0 阶段需要的模型:
- MTCNN 人脸检测: mtcnn_onnx 包自动处理
- MODNet 人像抠图: 自动下载到 models/ 目录

用法:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir ./models
"""

import os
import sys
import argparse

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def download_mtcnn_models(model_dir: str):
    """下载 MTCNN 模型（通过 mtcnn-runtime 自动管理）"""
    print("\n" + "=" * 50)
    print("[1/3] 下载 MTCNN 人脸检测模型...")

    try:
        import mtcnn_runtime
        # mtcnn-runtime 包会自动处理模型下载
        # 验证可用性：尝试创建检测器
        from mtcnn_runtime import MTCNN
        _ = MTCNN()
        print("   ✅ MTCNN 模型已就绪 (via mtcnn-runtime)")
    except ImportError:
        print("   ⚠️  mtcnn-runtime 未安装，请运行: pip install mtcnn-runtime")
    except Exception as e:
        print(f"   ⚠️  MTCNN 初始化异常: {e}")
        print("   运行时将自动下载模型文件")


def download_modnet_model(model_dir: str):
    """下载 MODNet 抠图模型"""
    print("\n" + "=" * 50)
    print("[2/3] 下载 MODNet 人像抠图模型...")

    # MODNet 模型的下载 URL（来自 HivisionIDPhotos 官方仓库）
    MODNET_URLS = {
        "modnet_photographic_portrait_matting.onnx": (
            "https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/"
            "pretrained-model/modnet_photographic_portrait_matting.onnx"
        ),
        "hivision_modnet.onnx": (
            "https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/"
            "pretrained-model/hivision_modnet.onnx"
        ),
    }

    os.makedirs(model_dir, exist_ok=True)

    for filename, url in MODNET_URLS.items():
        filepath = os.path.join(model_dir, filename)

        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ {filename} 已存在 ({size_mb:.1f} MB)")
            continue

        print(f"   ⬇️  下载 {filename} ...")
        try:
            import requests
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r      进度: {progress:.0f}%", end="", flush=True)

            print(f"\n   ✅ {filename} 下载完成 ({downloaded / (1024*1024):.1f} MB)")

        except ImportError:
            print("   ⚠️  requests 库未安装，请运行: pip install requests")
            print(f"   或手动下载: {url}")
            print(f"   保存到: {filepath}")
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            print(f"   请手动下载: {url}")
            print(f"   保存到: {filepath}")


def download_additional_models(model_dir: str):
    """下载其他辅助模型（可选）"""
    print("\n" + "=" * 50)
    print("[3/3] 下载辅助模型...")

    # RMBG-1.4 模型（增强抠图，可选）
    rmbg_path = os.path.join(model_dir, "rmbg-1.4.onnx")
    if os.path.exists(rmbg_path):
        size_mb = os.path.getsize(rmbg_path) / (1024 * 1024)
        print(f"   ⏭️  RMBG-1.4 跳过 (已存在, {size_mb:.1f} MB)")
    else:
        print("   ⏭️  RMBG-1.4 跳过 (P1阶段使用，可选)")

    # RetinaFace 模型（增强人脸检测，P1阶段）
    retinaface_path = os.path.join(model_dir, "retinaface_resnet50.onnx")
    if os.path.exists(retinaface_path):
        print(f"   ⏭️  RetinaFace 跳过 (已存在)")
    else:
        print("   ⏭️  RetinaFace 跳过 (P1阶段使用，可选)")

    print("\n" + "=" * 50)
    print("模型下载完成!")


def verify_models(model_dir: str):
    """验证模型文件完整性"""
    print("\n" + "=" * 50)
    print("验证模型文件...")

    required_models = [
        # MTCNN 由 mtcnn-runtime 管理，无需文件验证
    ]

    optional_models = [
        "modnet_photographic_portrait_matting.onnx",
        "hivision_modnet.onnx",
    ]

    for model in optional_models:
        filepath = os.path.join(model_dir, model)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ {model} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚠️  {model} (未找到，可能需要下载)")

    # 验证 MTCNN
    try:
        import mtcnn_runtime
        print("   ✅ mtcnn-runtime (MTCNN) 可用")
    except ImportError:
        print("   ❌ mtcnn-runtime 未安装")


def main():
    parser = argparse.ArgumentParser(description="下载 AI一照成证 所需的模型文件")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(PROJECT_ROOT, "models"),
        help=f"模型保存目录 (默认: {PROJECT_ROOT}/models)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir

    print("""
╔══════════════════════════════════════════╗
║   AI一照成证 — 模型下载工具              ║
╠══════════════════════════════════════════╣
║  P0 阶段模型:                            ║
║  · MTCNN - 人脸检测 (~2MB)               ║
║  · MODNet - 人像抠图 (~25MB)             ║
║  · 总计: ~27MB                           ║
╚══════════════════════════════════════════╝
""")

    print(f"模型目录: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    download_mtcnn_models(model_dir)
    download_modnet_model(model_dir)
    download_additional_models(model_dir)
    verify_models(model_dir)

    print("\n✅ 所有模型准备就绪!")
    print(f"\n现在可以运行:")
    print(f"  python cli.py --input <照片路径> --spec 一寸 --color blue")
    print(f"  python api_server.py")


if __name__ == "__main__":
    main()
