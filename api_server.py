#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI一照成证 — REST API 服务器"""

import os, sys, time, io, base64
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, r"C:\Users\24817\HivisionIDPhotos")

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from hivision.utils import add_background
from idphoto_system.utils.image_utils import SPECS, COLORS, resolve_spec, resolve_color, array_to_base64
from idphoto_system.compliance.smart_engine import SmartComplianceEngine

app = FastAPI(title="AI一照成证", version="0.3.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_web_dir = os.path.join(PROJECT_ROOT, "web_frontend")
if os.path.exists(_web_dir):
    @app.get("/app", response_class=HTMLResponse)
    async def web_app():
        with open(os.path.join(_web_dir, "index.html"), "r", encoding="utf-8") as f:
            return f.read()

_creator = None
def get_creator():
    global _creator
    if _creator is None:
        _creator = IDCreator()
        choose_handler(_creator, "hivision_modnet", "mtcnn")
    return _creator

@app.get("/")
async def root():
    return {"name": "AI一照成证", "version": "0.3.0", "frontend": "/app"}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/specs")
async def get_specs():
    return {"specs": [{"id": s.name, "label": f"{s.name}（{s.usage}）" if s.usage else s.label,
            "width": s.width, "height": s.height} for s in SPECS.values()]}

@app.get("/api/colors")
async def get_colors():
    return {"colors": [{"name": n, "hex": f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"}
            for n, c in COLORS.items() if n in ("white","blue","red")]}

@app.get("/api/examples")
async def get_examples():
    """返回可用的示例图片列表"""
    examples = []
    demo_dir = r"C:\Users\24817\HivisionIDPhotos\demo\images"
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

def process_single(img, spec_name, color_name, bg_mode="pure_color", beauty_strength=0, brightness_strength=0):
    """纯证件照生成"""
    spec = resolve_spec(spec_name)
    rgb = resolve_color(color_name)
    bgr = (rgb[2], rgb[1], rgb[0])

    c = get_creator()
    result = c(img, size=spec.size, change_bg_only=False,
               head_measure_ratio=0.38, head_height_ratio=0.50, head_top_range=(0.10, 0.08),
               whitening_strength=int(beauty_strength),
               brightness_strength=int(brightness_strength),
               contrast_strength=int(beauty_strength)//3 if beauty_strength else 0)
    matting = result.matting
    alpha = matting[:, :, 3].astype(np.float32) / 255.0

    rgba = np.dstack([matting[:,:,:3], (alpha*255).astype(np.uint8)])
    composited = add_background(rgba, bgr=bgr, mode=bg_mode).astype(np.uint8)

    composited_bgra = cv2.cvtColor(composited, cv2.COLOR_BGR2BGRA)
    c2 = IDCreator()
    choose_handler(c2, "hivision_modnet", "mtcnn")
    result2 = c2(composited_bgra, size=spec.size, crop_only=True,
                 head_measure_ratio=0.38, head_height_ratio=0.50, head_top_range=(0.10, 0.08))

    # 智能检测
    checks = None
    try:
        face_info = result.face or {}
        fc = {"bbox": None, "roll_angle": 0}
        rect2 = face_info.get("rectangle")
        if rect2 and len(rect2) == 4:
            cx, cy, fw, fh = rect2
            fc["bbox"] = (int(cx-fw/2), int(cy-fh/2), int(cx+fw/2), int(cy+fh/2))
        fc["roll_angle"] = face_info.get("roll_angle", 0)
        engine = SmartComplianceEngine()
        checks = engine.check(img, fc, alpha).to_dict()
    except Exception:
        pass

    return cv2.cvtColor(result2.standard, cv2.COLOR_BGRA2BGR), checks

@app.post("/api/process")
async def process_photo(
    image: UploadFile = File(...),
    spec: str = Form("一寸"),
    color: str = Form("blue"),
    bg_mode: str = Form("pure_color"),
    beauty: str = Form("0"),
    brightness: str = Form("0"),
):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(400, "无法解析图片")
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        raise HTTPException(400, f"读取失败: {e}")

    t0 = time.time()
    try:
        result, checks = process_single(img, spec, color, bg_mode=bg_mode,
                               beauty_strength=float(beauty), brightness_strength=float(brightness))
    except Exception as e:
        import traceback
        raise HTTPException(500, f"处理失败: {e}")

    return {
        "status": "ok",
        "image": array_to_base64(result),
        "spec": {"name": resolve_spec(spec).name, "width": result.shape[1], "height": result.shape[0]},
        "processing_time": round(time.time() - t0, 3),
        "checks": checks,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"""
    AI一照成证 API v0.3
    前端: http://127.0.0.1:{args.port}/app
    """)
    uvicorn.run(app, host=args.host, port=args.port)
