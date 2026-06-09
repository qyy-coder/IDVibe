"""
图像工具模块 — DPI处理、Base64编解码、证件照规格定义

本模块定义了中国标准证件照的所有规格参数，
以及图像加载/保存/格式转换的通用工具函数。
"""

import io
import base64
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 证件照规格定义
# ============================================================

@dataclass
class PhotoSpec:
    """证件照规格"""
    name: str           # 规格名称（中文）
    width: int          # 宽度（像素 @300 DPI）
    height: int         # 高度（像素 @300 DPI）
    label: str          # 简短标签
    usage: str = ""     # 常见用途
    mm_width: int = 0   # 物理宽度（mm）
    mm_height: int = 0  # 物理高度（mm）

    @property
    def size(self) -> Tuple[int, int]:
        """返回 (height, width) 元组，兼容 HivisionIDPhotos"""
        return (self.height, self.width)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


# 中国标准证件照规格（像素 @300 DPI）
SPECS: Dict[str, PhotoSpec] = {
    "一寸": PhotoSpec(
        name="一寸",
        width=295, height=413,
        mm_width=25, mm_height=35,
        label="一寸 (25×35mm)",
        usage="常见考试报名、简历",
    ),
    "小一寸": PhotoSpec(
        name="小一寸",
        width=259, height=377,
        mm_width=22, mm_height=32,
        label="小一寸 (22×32mm)",
        usage="驾驶证、部分考试",
    ),
    "大一寸": PhotoSpec(
        name="大一寸",
        width=390, height=567,
        mm_width=33, mm_height=48,
        label="大一寸 (33×48mm)",
        usage="护照、签证",
    ),
    "二寸": PhotoSpec(
        name="二寸",
        width=413, height=579,
        mm_width=35, mm_height=49,
        label="二寸 (35×49mm)",
        usage="毕业证、学位证、部分签证",
    ),
    "小二寸": PhotoSpec(
        name="小二寸",
        width=413, height=531,
        mm_width=35, mm_height=45,
        label="小二寸 (35×45mm)",
        usage="港澳通行证、部分签证",
    ),
    "美国签证": PhotoSpec(
        name="美国签证",
        width=600, height=600,
        mm_width=51, mm_height=51,
        label="美国签证 (51×51mm)",
        usage="美国签证",
    ),
    "日本签证": PhotoSpec(
        name="日本签证",
        width=413, height=531,
        mm_width=35, mm_height=45,
        label="日本签证 (35×45mm)",
        usage="日本签证",
    ),
    "申根签证": PhotoSpec(
        name="申根签证",
        width=413, height=531,
        mm_width=35, mm_height=45,
        label="申根签证 (35×45mm)",
        usage="申根国家签证",
    ),
    "大二寸": PhotoSpec(
        name="大二寸",
        width=413, height=626,
        mm_width=35, mm_height=53,
        label="大二寸 (35×53mm)",
        usage="部分签证、毕业证",
    ),
    "三寸": PhotoSpec(
        name="三寸",
        width=649, height=991,
        mm_width=55, mm_height=84,
        label="三寸 (55×84mm)",
        usage="结婚登记照",
    ),
}

# 标准证件照背景色
COLORS: Dict[str, Tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "blue": (99, 140, 206),       # 标准证件照蓝底
    "red": (206, 53, 53),         # 标准证件照红底
    "light_blue": (153, 204, 255),
    "dark_blue": (51, 51, 153),
    "浅蓝": (153, 204, 255),
    "白色": (255, 255, 255),
    "蓝色": (99, 140, 206),
    "红色": (206, 53, 53),
}

# 常用颜色名映射
COLOR_ALIASES = {
    "白": "white", "白底": "white", "白色": "white",
    "蓝": "blue", "蓝底": "blue", "蓝色": "blue",
    "红": "red", "红底": "red", "红色": "red",
}


# ============================================================
# 图像加载/保存工具
# ============================================================

def load_image(path: str, max_size: int = 2000) -> np.ndarray:
    """
    加载图像，若最大边超过 max_size 则等比缩放。
    返回 BGR 格式的 numpy 数组 (OpenCV 格式)。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_size:
        scale = max_size / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def save_image(image: np.ndarray, path: str, dpi: int = 300):
    """
    保存图像到文件，设置 DPI。
    支持 BGR(3通道)/BGRA(4通道) numpy 数组。
    """
    if image.shape[2] == 4:
        # BGRA → RGBA
        rgb_img = np.concatenate(
            (np.flip(image[:, :, :3], axis=-1), image[:, :, 3:]),
            axis=-1,
        ).astype(np.uint8)
        pil_img = Image.fromarray(rgb_img, mode="RGBA")
    else:
        # BGR → RGB
        rgb_img = np.flip(image, axis=-1).astype(np.uint8)
        pil_img = Image.fromarray(rgb_img, mode="RGB")

    pil_img.save(path, dpi=(dpi, dpi))


def array_to_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """将 numpy 图像数组转为 Base64 字符串（含 data URL 前缀）。"""
    if fmt == ".jpg" or fmt == ".jpeg":
        # JPEG 不支持透明通道
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        success, buffer = cv2.imencode(fmt, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("图像编码失败")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/{fmt.lstrip('.')};base64,{b64}"


def base64_to_array(b64_str: str) -> np.ndarray:
    """将 Base64 字符串（含或不含 data URL 前缀）转为 numpy 数组。"""
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img


def resolve_color(color_name: str) -> Tuple[int, int, int]:
    """
    将颜色名解析为 RGB 元组。
    支持中文名、英文名、hex 码。
    返回 (R, G, B) 元组。
    """
    # 先检查别名
    key = COLOR_ALIASES.get(color_name, color_name.lower())
    if key in COLORS:
        return COLORS[key]
    # 尝试解析 hex 码
    if color_name.startswith("#"):
        color_name = color_name.lstrip("#")
        if len(color_name) == 6:
            return (
                int(color_name[0:2], 16),
                int(color_name[2:4], 16),
                int(color_name[4:6], 16),
            )
    raise ValueError(
        f"未知的颜色: '{color_name}'。"
        f"可用颜色: {list(COLORS.keys())}"
    )


def resolve_spec(spec_name: str) -> PhotoSpec:
    """将规格名解析为 PhotoSpec 对象。"""
    # 支持模糊匹配
    for key, spec in SPECS.items():
        if spec_name in key or key in spec_name:
            return spec
    raise ValueError(
        f"未知的规格: '{spec_name}'。"
        f"可用规格: {list(SPECS.keys())}"
    )


def resize_to_kb(image: np.ndarray, target_kb: int, dpi: int = 300) -> bytes:
    """
    将图像压缩到目标 KB 大小。
    返回 JPEG 字节流。
    """
    if image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        img = image

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    quality = 95

    while True:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, dpi=(dpi, dpi))
        size_kb = len(buf.getvalue()) / 1024

        if size_kb <= target_kb or quality <= 5:
            if size_kb < target_kb:
                # 填充到目标大小
                padding = b"\x00" * int((target_kb * 1024) - len(buf.getvalue()))
                buf.write(padding)
            return buf.getvalue()

        quality -= 5
