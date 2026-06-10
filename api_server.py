#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI一照成证 — REST API 服务器"""

import os
import sys
import time
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from idphoto_system.config import HIVISION_PATH, DEMO_IMAGES_PATH, DEFAULT_PORT, DEFAULT_HOST
from idphoto_system.processing_service import PhotoProcessingService, ProcessRequest
from idphoto_system.utils.image_utils import SPECS, COLORS, resolve_spec, resolve_color, array_to_base64

# 确保 HivisionIDPhotos 在路径中
if HIVISION_PATH not in sys.path:
    sys.path.insert(0, HIVISION_PATH)

app = FastAPI(title="AI一照成证", version="0.3.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── 静态前端 ──
_web_dir = os.path.join(PROJECT_ROOT, "web_frontend")
if os.path.exists(_web_dir):
    @app.get("/app", response_class=HTMLResponse)
    async def web_app():
        with open(os.path.join(_web_dir, "index.html"), "r", encoding="utf-8") as f:
            return f.read()

# ── 核心服务 (单例) ──
_service = None


def get_service() -> PhotoProcessingService:
    global _service
    if _service is None:
        _service = PhotoProcessingService(hivision_path=HIVISION_PATH)
    return _service


# ── API 端点 ──

@app.get("/")
async def root():
    return {"name": "AI一照成证", "version": "0.3.0", "frontend": "/app"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/specs")
async def get_specs():
    return {"specs": [
        {"id": s.name, "label": f"{s.name}（{s.usage}）" if s.usage else s.label,
         "width": s.width, "height": s.height}
        for s in SPECS.values()
    ]}


@app.get("/api/colors")
async def get_colors():
    return {"colors": [
        {"name": n, "hex": f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"}
        for n, c in COLORS.items() if n in ("white", "blue", "red")
    ]}


@app.get("/api/examples")
async def get_examples():
    """返回可用的示例图片列表"""
    examples = []
    demo_dir = DEMO_IMAGES_PATH
    if os.path.exists(demo_dir):
        for f in sorted(os.listdir(demo_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(demo_dir, f)
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    b64 = array_to_base64(img)
                    examples.append({"name": f, "width": w, "height": h, "thumbnail": b64})
    return {"examples": examples}


@app.post("/api/process")
async def process_photo(
    image: UploadFile = File(...),
    spec: str = Form("一寸"),
    color: str = Form("blue"),
    bg_mode: str = Form("pure_color"),
    beauty: str = Form("0"),
    brightness: str = Form("0"),
):
    """生成证件照 — 统一使用 PhotoProcessingService"""
    # 读取上传图片
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(400, "无法解析图片")
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"读取失败: {e}")

    # 构建请求
    t0 = time.time()
    req = ProcessRequest(
        image=img,
        spec=spec,
        color=color,
        bg_mode=bg_mode,
        beauty_strength=float(beauty),
        brightness_strength=float(brightness),
        enable_edge_refine=False,       # API 模式下跳过快边缘优化以提升速度
        enable_compliance=True,
        compliance_mode="smart",         # 使用智能检测（8项轻量）
    )

    try:
        result = get_service().process(req)
    except Exception as e:
        import traceback
        raise HTTPException(500, f"处理失败: {e}")

    return {
        "status": "ok",
        "image": array_to_base64(result.standard),
        "spec": {"name": result.spec_name, "width": result.spec_width, "height": result.spec_height},
        "processing_time": round(time.time() - t0, 3),
        "checks": result.checks,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--host", default=DEFAULT_HOST)
    args = p.parse_args()
    print(f"""
    ╔════════════════════════════════╗
    ║   AI一照成证 API v0.3         ║
    ║   前端: http://127.0.0.1:{args.port}/app
    ║   API文档: http://127.0.0.1:{args.port}/docs
    ╚════════════════════════════════╝
    """)
    uvicorn.run(app, host=args.host, port=args.port)
