# AI一照成证 — 智能证件照生成与优化系统

> **技术底座**: HivisionIDPhotos + 自研P1增强模块

## 项目简介

上传任意人像照片，AI自动完成**抠图 → 换底 → 智能裁剪 → 输出标准证件照**。支持10种规格、3种背景色、智能美颜和渐变背景。**全本地处理，隐私零泄露。**

## 功能演示

```
┌──────────────┬──────────────┐
│  示例照片     │  上传照片     │
│  规格下拉     │  背景颜色     │
│  ☑智能美颜   │  ☑渐变背景   │
│  ☑三色同出   │              │
├──────────────┴──────────────┤
│ [原图] [白底] [蓝底] [红底] │  ← 一页横排展示
│ ✅头部 ✅光照 ✅清晰 ...     │  ← 智能检测
└─────────────────────────────┘
```

## 快速开始

### 环境要求
- Python 3.10+
- Windows / macOS / Linux

### 安装

```bash
pip install -r requirements.txt
```

### 启动

**方式一：一键启动（Windows）**
```bash
start.bat
```

**方式二：命令行**
```bash
# 启动 API 服务器
python api_server.py --port 8000

# 浏览器打开
http://127.0.0.1:8000/app
```

## 项目结构

```
一照成证/
├── api_server.py               # FastAPI 后端服务
├── cli.py                      # 命令行批量处理工具
├── run_demo.py                 # 完整演示脚本（含分阶段计时）
├── web_frontend/               # Web 前端界面
│   └── index.html              # 暗色主题单页应用（零框架依赖）
├── idphoto_system/             # 核心模块
│   ├── config.py               # 集中配置（路径/参数/模型名）
│   ├── processing_service.py   # 统一处理服务（api/cli/demo 共用）
│   ├── compliance/             # ★ P1 自研：智能合规检测引擎
│   │   ├── engine.py           #   编排器 → 依次执行5维度检测
│   │   ├── smart_engine.py     #   轻量版 → 复用完整版函数 + 宽松阈值
│   │   ├── standards.py        #   5国标准阈值配置
│   │   ├── models.py           #   数据模型：RuleResult + ComplianceReport
│   │   ├── geometric_checks.py #   G01-G04 几何检测
│   │   ├── pose_checks.py      #   P01-P03 姿态检测
│   │   ├── facial_checks.py    #   F01-F06 面部状态检测
│   │   ├── lighting_checks.py  #   L01-L04 光照色彩检测
│   │   └── quality_checks.py   #   Q01-Q04 图像质量检测
│   ├── matting/                # ★ P1 自研：级联抠图边缘优化
│   │   ├── guided_filter.py    #   引导滤波 O(N) 盒式实现
│   │   ├── edge_refiner.py     #   边缘精细化 + Trimap生成
│   │   └── adaptive_fallback.py#   置信度评分 + 三档回退决策
│   ├── inference/              # 人脸检测 + 抠图封装
│   ├── processing/             # 背景替换 + 裁剪 + 排版
│   └── utils/                  # 工具函数 + 10种证件照规格定义
├── tests/                      # 测试套件 (44项)
├── outputs/                    # 生成结果输出目录
├── requirements.txt            # Python 依赖
└── README.md
```

## 核心架构与代码逻辑

### 整体数据流

```
用户输入 (照片)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  入口层: api_server.py / cli.py / run_demo.py           │
│  ↓ 统一委托给 PhotoProcessingService.process()          │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────────┐
    │           处理流水线 (6 阶段)                      │
    │                                                   │
    │  ① 预处理       缩放 ≤2000px, 确保 3ch BGR       │
    │  ② IDCreator    MTCNN人脸检测 + MODNet人像抠图    │
    │  ③ 边缘优化★    Guided Filter → 置信度 → 回退    │
    │  ④ 背景替换     纯色 / 上下渐变                   │
    │  ⑤ 精确裁剪     crop_only 模式居中裁剪            │
    │  ⑥ 合规检测★    Smart(8项) 或 Full(22+项)        │
    │                                                   │
    │  输出: 标准照 + HD高清 + 合规报告 + 排版(可选)      │
    └──────────────────────────────────────────────────┘
```

### 板块一：合规检测引擎 ★ 自研核心 (22+ 规则, 5 维度)

```
engine.check(image, face_info, alpha)
  │
  ├── geometric_checks:  G01头部占比  G02眼睛位置  G03居中  G04下颚位置
  │      └─ 用 StandardProfile 阈值判断 → RuleResult(PASS/FAIL/WARN)
  │
  ├── pose_checks:       P01偏航角    P02俯仰角    P03翻滚角
  │      └─ solvePnP(3D人脸模型, MTCNN 5点关键点) → 欧拉角 → 阈值对比
  │
  ├── facial_checks:     F01眼睛开合  F02嘴部闭合  F03表情
  │                      F04眼镜检测  F05镜片反光  F06红眼
  │      └─ Sobel梯度估算EAR(眼睛纵横比) / MAR(嘴巴暗像素比例)
  │
  ├── lighting_checks:   L01面部光照均匀  L02阴影  L03背景均匀  L04背景色
  │      └─ 左右脸灰度均值比 + Lab a*/b*通道方差分析
  │
  └── quality_checks:    Q01分辨率  Q02清晰度  Q03摩尔纹  Q04压缩质量
         └─ Laplacian方差 / 2D FFT频域分析 / 8×8块效应检测
```

**判定逻辑：**
- `critical_rules = [G01, P01, P02, P03, F01]` — 任一 FAIL 则整体不合格
- 非关键规则 ≤2 个 FAIL 仍可通过
- `overall_score = mean(所有规则分数)`

**双引擎设计：**

| 引擎 | 规则数 | 用途 | 阈值来源 |
|------|:-----:|------|---------|
| `ComplianceEngine` | 22+ | 命令行演示/批量 | 5国 StandardProfile 精确阈值 |
| `SmartComplianceEngine` | 8 | Web 前端实时反馈 | `quick_check` 宽松标准（仅严重问题才报警） |

重构后 `SmartComplianceEngine` 直接复用 `ComplianceEngine` 的检测函数，不再重复实现。

**多国标准支持：** ISO/IEC 19794-5 / 中国 GA/T / 美国签证 / 申根签证 / 日本签证 — 每套标准独立的头占比、角度、光照、清晰度阈值。

### 板块二：级联抠图边缘优化 ★ 自研核心

```
cascade_refine(image, coarse_alpha)
  │
  ├─ Stage 1: MODNet 粗抠图 (HivisionIDPhotos 提供)
  │     └─ alpha ∈ [0,1], 边缘有锯齿和过渡区
  │
  ├─ Stage 2: 引导滤波边缘精细化
  │     └─ EdgeRefiner.refine()
  │          ├─ 以原图灰度图为 guide, alpha 为输入
  │          ├─ 盒式滤波 O(N) 计算局部均值/方差/协方差 → a,b 系数
  │          └─ 输出: 保留原图边缘结构的细化 alpha
  │
  └─ Stage 3: 置信度评分 + 自适应回退
        └─ AdaptiveFallback.evaluate()
             ├─ 不确定区域占比 (0.05 < alpha < 0.95) → area_penalty
             ├─ 边缘梯度强度 (Sobel) → edge_bonus
             ├─ 边缘连续性 (Canny) → continuity_bonus
             └─ confidence = 100 - penalty + bonus
                  │
                  ├─ ≥70 "high"   → 直接使用细化结果
                  ├─ 40~70 "medium" → 混合细化+原始 (按置信度加权)
                  └─ <40 "low"    → 标记 needs_fallback=True
```

### 板块三：统一处理服务

`PhotoProcessingService` 封装了 `api_server.py`、`run_demo.py`、`cli.py` 三条入口的共有核心逻辑，消除代码重复。

```python
class PhotoProcessingService:
    def process(req: ProcessRequest) -> ProcessResult:
        img = _preprocess(req.image)              # 缩放+格式统一
        result = _run_creator(img, spec, req)      # IDCreator 主流程
        alpha = _extract_alpha(result.matting)     # BGRA→float32 alpha
        alpha, mq = _edge_refine(img, alpha)       # P1边缘优化(可选)
        composited = _composite(matting, alpha)    # 背景替换
        standard = _precise_crop(composited, spec) # 精确裁剪
        checks = _run_compliance(img, face, alpha) # 合规检测(可选)
        return ProcessResult(...)
```

### 配置管理

所有环境相关配置集中在 `idphoto_system/config.py`：HivisionIDPhotos 路径、模型名、处理参数、服务器端口。支持环境变量覆盖（如 `HIVISION_PATH`），换机器只需改一处。

## 自研创新点

### 1. 级联精细化抠图边缘优化
- **Guided Filter** 引导滤波 (O(N) 盒式滤波实现)
- **置信度评分** — 边缘梯度 + 过渡区分析
- **自适应回退** — 低置信度时混合/标记重算

### 2. 智能合规检测引擎
- 8项核心检测：头部大小、人脸居中、头部倾斜、光照均匀、清晰度、亮度、背景纯净度
- **自适应阈值** — 根据拍照距离自动调整判断标准
- 零额外模型依赖 (纯 OpenCV + NumPy)

### 3. 智能美颜 (基于 HivisionIDPhotos)
- 全本地美白 + 提亮
- 无须上传云端，隐私安全

### 4. 三色同出
- 一次请求同时生成白/蓝/红三底色
- 渐变背景模式（上下渐变过渡）

## 技术栈

| 层 | 技术 |
|------|------|
| AI推理 | MODNet (人像抠图), MTCNN (人脸检测) |
| 图像处理 | OpenCV, NumPy, PIL |
| 后端 | FastAPI, Uvicorn |
| 前端 | 原生 HTML/CSS/JS (零框架依赖) |
| 模型格式 | ONNX Runtime |

## 性能指标

| 指标 | 数值 |
|------|:--:|
| 全流程耗时 (CPU) | 1.5-3s |
| 模型总大小 | ~25MB |
| 支持规格数 | 10种 |
| 支持背景色 | 3色 + 渐变 |
| 测试通过率 | 44/44 (100%) |

## License

本项目基于 HivisionIDPhotos (Apache 2.0) 进行二次开发。
自研模块 (compliance/, matting/) 为原创代码。
