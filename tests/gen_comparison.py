#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate side-by-side comparison images for PPT"""

import sys, os
sys.path.insert(0, r"C:\Users\24817\HivisionIDPhotos")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler
from idphoto_system.matting import cascade_refine

def gen_comparison(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    if max(h,w) > 2000:
        s = 2000 / max(h,w)
        img = cv2.resize(img, (int(w*s), int(h*s)))

    # Run MODNet
    c = IDCreator()
    choose_handler(c, "hivision_modnet", "mtcnn")
    from hivision.creator.context import Context, Params
    ctx = Context(Params(size=(413,295)))
    ctx.processing_image = img.copy()
    ctx.origin_image = img.copy()
    c.matting_handler(ctx)
    coarse = ctx.matting_image[:,:,3].astype(np.float32)/255.0

    # Run Guided Filter
    refined, _ = cascade_refine(img, coarse)

    # Create side-by-side visualization
    # 1. Zoom into a hair/edge region
    crop_y, crop_x = h//3, w//3
    crop_h, crop_w = 200, 300
    y1, x1 = crop_y, crop_x

    # 2. Build comparison strip: ORIGINAL | MODNet | +GF
    original_roi = img[y1:y1+crop_h, x1:x1+crop_w]
    coarse_roi = coarse[y1:y1+crop_h, x1:x1+crop_w]
    refined_roi = refined[y1:y1+crop_h, x1:x1+crop_w]

    # Convert alpha to 3-channel for visualization
    coarse_viz = (np.stack([coarse_roi]*3, axis=-1) * 255).astype(np.uint8)
    refined_viz = (np.stack([refined_roi]*3, axis=-1) * 255).astype(np.uint8)

    # Edge overlay: red where coarse has edge but refined is cleaner
    edge_diff = cv2.absdiff(coarse_viz, refined_viz)
    edge_diff = cv2.applyColorMap(edge_diff, cv2.COLORMAP_HOT)

    # Full image alpha comparison (downsized)
    fh, fw = 200, int(200 * w/h)
    coarse_full = (np.stack([cv2.resize(coarse,(fw,fh))]*3, axis=-1)*255).astype(np.uint8)
    refined_full = (np.stack([cv2.resize(refined,(fw,fh))]*3, axis=-1)*255).astype(np.uint8)
    diff_map = cv2.absdiff(coarse_full, refined_full)
    diff_color = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

    # Assemble final image - pad rows to same width
    gap = np.ones((crop_h, 4, 3), dtype=np.uint8)*255
    row1 = np.hstack([original_roi, gap, coarse_viz, gap, refined_viz, gap, edge_diff])

    # Build row2: full alpha maps
    pw = (row1.shape[1] - coarse_full.shape[1] - refined_full.shape[1] - diff_color.shape[1]) // 2
    pad = np.ones((coarse_full.shape[0], max(0, pw), 3), dtype=np.uint8)*255
    row2 = np.hstack([coarse_full, pad, refined_full, pad, diff_color])
    if row2.shape[1] < row1.shape[1]:
        row2 = np.hstack([row2, np.ones((row2.shape[0], row1.shape[1]-row2.shape[1], 3), dtype=np.uint8)*255])

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(row1, 'Original', (5,20), font, 0.5, (255,255,255), 1)
    cv2.putText(row1, 'MODNet', (crop_w+10,20), font, 0.5, (255,255,255), 1)
    cv2.putText(row1, '+GuidedFilter', (crop_w*2+20,20), font, 0.5, (255,255,255), 1)
    cv2.putText(row1, 'Diff', (crop_w*3+30,20), font, 0.5, (255,255,255), 1)
    cv2.putText(row2, 'MODNet Full Alpha', (5,20), font, 0.5, (255,255,255), 1)
    cv2.putText(row2, '+GF Full Alpha', (coarse_full.shape[1]+10,20), font, 0.5, (255,255,255), 1)
    cv2.putText(row2, 'Difference Map', (coarse_full.shape[1]+refined_full.shape[1]+pw+10,20), font, 0.5, (255,255,255), 1)

    final = np.vstack([row1, np.ones((4, row1.shape[1], 3), dtype=np.uint8)*255, row2])
    cv2.imwrite(output_path, final)

    return True

if __name__ == "__main__":
    test_dir = r"C:\Users\24817\HivisionIDPhotos\demo\images"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(test_dir)):
        if f.lower().endswith(('.jpg','.png')):
            out = os.path.join(out_dir, f"compare_{f}")
            if gen_comparison(os.path.join(test_dir,f), out):
                print(f"Generated: {out}")
    print("Done - comparison images saved to outputs/")
