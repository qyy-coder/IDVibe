"""
集中配置模块
=============
所有环境相关的路径、模型参数、处理阈值都在此处定义。
支持环境变量覆盖，消除硬编码。

用法:
    from idphoto_system.config import HIVISION_PATH, HEAD_MEASURE_RATIO
"""

import os

# ============================================================
# 路径配置 — 换机器时只需修改此处或设置环境变量
# ============================================================

HIVISION_PATH = os.environ.get(
    "HIVISION_PATH",
    r"C:\Users\24817\HivisionIDPhotos"
)
"""HivisionIDPhotos 库的安装路径"""

DEMO_IMAGES_PATH = os.environ.get(
    "DEMO_IMAGES_PATH",
    os.path.join(HIVISION_PATH, "demo", "images")
)
"""示例图片目录"""

MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
)
"""ONNX 模型文件目录"""

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
"""生成结果输出目录"""

# ============================================================
# 模型选择
# ============================================================

DEFAULT_MATTING_MODEL = "hivision_modnet"
"""默认抠图模型名称（传递给 HivisionIDPhotos choose_handler）"""

DEFAULT_FACE_MODEL = "mtcnn"
"""默认人脸检测模型名称"""

# ============================================================
# 图像处理参数
# ============================================================

MAX_IMAGE_SIDE = 2000
"""输入图像最大边长（像素），超出则等比缩放"""

HD_SCALE_FACTOR = 2.0
"""高清版相对于标准尺寸的放大倍数"""

# HivisionIDPhotos IDCreator 头部定位参数
HEAD_MEASURE_RATIO = 0.38
"""头部面积占画面比例 (api_server / run_demo 共用)"""

HEAD_HEIGHT_RATIO = 0.50
"""头部中心在画面高度的位置比例"""

HEAD_TOP_RANGE = (0.10, 0.08)
"""头部顶部范围"""

# ============================================================
# 美颜参数
# ============================================================

DEFAULT_WHITENING_STRENGTH = 15
DEFAULT_BRIGHTNESS_STRENGTH = 3
DEFAULT_CONTRAST_STRENGTH = 5

# ============================================================
# 输出
# ============================================================

DEFAULT_DPI = 300

# ============================================================
# 服务器
# ============================================================

DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
