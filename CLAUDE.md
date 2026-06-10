# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI ID photo generation system ("AI一照成证") — upload a portrait, AI handles matting → background replacement → smart cropping → standard-compliant ID photo output. Supports 10 Chinese standard sizes, 3 background colors, smart beautification, and gradient backgrounds. **All local processing, zero privacy leakage.**

Built on HivisionIDPhotos (Apache 2.0) with two custom P1 enhancement modules: cascade edge refinement and smart compliance engine.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web app (Windows one-click)
start.bat

# Start API server manually
python api_server.py --port 8000
# Web UI: http://127.0.0.1:8000/app
# API docs: http://127.0.0.1:8000/docs

# CLI: single photo
python cli.py --input photo.jpg --spec 一寸 --color blue

# CLI: batch processing
python cli.py --batch "photos/*.jpg" --spec 一寸 --color blue --output-dir results/

# CLI: list available specs/colors/layouts
python cli.py --list-specs
python cli.py --list-colors

# Demo with timed stages and compliance report
python run_demo.py --image photo.jpg --spec 一寸 --color blue --standard ISO

# Run all tests
python tests/test_pipeline.py          # Module imports, utils, processing (--skip-model to skip model-dependent tests)
python tests/test_compliance.py        # 20 compliance engine tests
python tests/test_matting.py           # 11 matting/edge-refinement tests
```

## Architecture

### Core package: `idphoto_system/`

```
idphoto_system/
├── pipeline/idphoto_pipeline.py   # IDPhotoPipeline — main orchestrator (load→face detect→matting→edge refine→bg replace→crop→compliance check→layout)
├── inference/                     # Wrappers for MTCNN face detection + MODNet human matting (ONNX Runtime)
├── processing/                    # BackgroundReplacer (pure/gradient), PhotoCropper, LayoutGenerator
├── compliance/                    # ★ P1 custom: 22+ rule compliance engine
│   ├── engine.py                  #   ComplianceEngine — orchestrates 5 categories of checks
│   ├── smart_engine.py            #   SmartComplianceEngine — lightweight 8-item check for the API frontend
│   ├── geometric_checks.py        #   G01-G04: head ratio, eye line, centering, chin margin
│   ├── pose_checks.py             #   P01-P03: yaw, pitch, roll
│   ├── facial_checks.py           #   F01-F06: eyes open, mouth closed, glasses, red-eye
│   ├── lighting_checks.py         #   L01-L04: face uniformity, shadow, bg uniformity, bg color
│   ├── quality_checks.py          #   Q01-Q04: resolution, sharpness, compression, DPI
│   ├── standards.py               #   5 country profiles: ISO, 中国, US_VISA, SCHENGEN, JAPAN
│   └── models.py                  #   Verdict enum, RuleResult, ComplianceReport dataclasses
├── matting/                       # ★ P1 custom: 3-stage cascade edge refinement
│   ├── guided_filter.py           #   Guided Filter with O(N) box-filter implementation
│   ├── edge_refiner.py            #   EdgeRefiner + TrimapGenerator
│   ├── adaptive_fallback.py       #   MattingQuality scoring + AdaptiveFallback (high/medium/low tiers) + cascade_refine()
│   └── __init__.py
└── utils/image_utils.py           #   PhotoSpec dataclass, 10 Chinese standard sizes (SPECS), COLORS, Base64 codec, DPI handling
```

### Entry points

- **`api_server.py`** — FastAPI app with endpoints: `POST /api/process` (main), `GET /api/specs`, `GET /api/colors`, `GET /api/examples`, `GET /app` (SPA frontend). Uses `SmartComplianceEngine` for lightweight checks returned in the API response. Lazy-initializes `IDCreator` from HivisionIDPhotos.
- **`cli.py`** — Argparse CLI wrapping `IDPhotoPipeline`. Supports single, batch, info listing, layout generation.
- **`run_demo.py`** — Standalone demo using HivisionIDPhotos native pipeline directly (not IDPhotoPipeline). Detailed per-stage timing and full compliance report output.

### Dependency: HivisionIDPhotos

The project depends on a local copy of HivisionIDPhotos (located at `C:\Users\24817\HivisionIDPhotos` on the original dev machine). It's gitignored. Key classes used:
- `IDCreator` — main HivisionIDPhotos pipeline
- `choose_handler()` — selects matting + face detection backends
- `add_background()` — background compositing with gradient modes
- ONNX models stored in `models/` directory (also gitignored)

**Hard-coded paths to fix when setting up on a new machine:**
- `api_server.py:8`: `sys.path.insert(0, r"C:\Users\24817\HivisionIDPhotos")`
- `api_server.py:64`: demo images path `r"C:\Users\24817\HivisionIDPhotos\demo\images"`
- `run_demo.py:10`: `HIVISION_PATH = r"C:\Users\24817\HivisionIDPhotos"`

### Data flow

```
Input image → FaceDetector (MTCNN) → HumanMatting (MODNet) → cascade_refine★ (Guided Filter → quality scoring → blend/fallback)
    → BackgroundReplacer (pure/gradient) → PhotoCropper (head-position-based centering) → ComplianceEngine★ (22+ rules, 5 dims, multi-standard)
    → output: standard size + HD + optional layout sheet + compliance report
```

★ = P1 custom innovation modules

### Key design decisions

- **Two compliance engines:** `ComplianceEngine` (full, 22+ rules, for `run_demo.py` and `IDPhotoPipeline`) vs `SmartComplianceEngine` (lightweight 8 checks, for API server responses). They are independent implementations, not inheriting from a shared base.
- **Pipeline vs direct HivisionIDPhotos usage:** `IDPhotoPipeline` wraps the full flow including P1 modules. `api_server.py` and `run_demo.py` call HivisionIDPhotos' `IDCreator` directly with custom pre/post-processing rather than using `IDPhotoPipeline`.
- **Standards system:** `StandardProfile` dataclass holds per-country thresholds. `get_standard()` supports fuzzy alias matching (e.g., "cn" → 中国, "申根" → SCHENGEN) with ISO fallback.
- **Frontend:** Zero-framework SPA (`web_frontend/index.html`), served as static HTML by FastAPI at `/app`.
- **Outputs:** All generated photos go to `outputs/` directory (gitignored except `.gitkeep`).
