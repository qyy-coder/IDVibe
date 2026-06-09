/**
 * AI一照成证 — 首页逻辑
 */

const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    // 照片
    selectedPhoto: '',

    // 规格
    specs: [],
    currentSpec: '一寸',
    currentSpecInfo: null,

    // 颜色
    colors: [],
    currentColor: 'white',

    // 排版
    layouts: [],
    currentLayout: '',

    // 状态
    generating: false,
    loadingText: '正在处理...',
  },

  onLoad() {
    this.loadConfigData();
  },

  onShow() {
    // 每次显示时刷新配置
    if (this.data.specs.length === 0) {
      this.loadConfigData();
    }
  },

  /**
   * 加载配置数据（规格、颜色、排版）
   */
  async loadConfigData() {
    try {
      // 并行请求配置数据
      const [specsRes, colorsRes, layoutsRes] = await Promise.all([
        api.getSpecs().catch(() => ({ specs: [] })),
        api.getColors().catch(() => ({ colors: [] })),
        api.getLayouts().catch(() => ({ layouts: [] })),
      ]);

      const specs = specsRes.specs || [];
      const colors = colorsRes.colors || [];
      const layouts = layoutsRes.layouts || [];

      // 设置默认规格信息
      const defaultSpec = specs.find(s => s.name === '一寸');

      this.setData({
        specs: specs.slice(0, 6), // 显示常用规格
        colors: colors.slice(0, 3), // 白/蓝/红
        layouts: layouts,
        currentSpecInfo: defaultSpec || null,
      });

      console.log(`[配置] 加载了 ${specs.length} 种规格, ${colors.length} 种颜色`);
    } catch (err) {
      console.warn('[配置] 后端不可用，使用默认配置:', err.message);
      this.loadDefaultConfig();
    }
  },

  /**
   * 离线默认配置（后端不可用时的回退方案）
   */
  loadDefaultConfig() {
    this.setData({
      specs: [
        { name: '一寸', label: '一寸 (25×35mm)', usage: '考试报名、简历', mm_width: 25, mm_height: 35 },
        { name: '二寸', label: '二寸 (35×49mm)', usage: '毕业证、签证', mm_width: 35, mm_height: 49 },
        { name: '小一寸', label: '小一寸 (22×32mm)', usage: '驾驶证', mm_width: 22, mm_height: 32 },
        { name: '大一寸', label: '大一寸 (33×48mm)', usage: '护照', mm_width: 33, mm_height: 48 },
        { name: '小二寸', label: '小二寸 (35×45mm)', usage: '港澳通行证', mm_width: 35, mm_height: 45 },
      ],
      colors: [
        { name: 'white', label: '白色', hex: '#ffffff' },
        { name: 'blue', label: '蓝色', hex: '#638cce' },
        { name: 'red', label: '红色', hex: '#ce3535' },
      ],
      layouts: [
        { name: '一寸×8', rows: 4, cols: 2, spec: '一寸', total_photos: 8 },
        { name: '二寸×4', rows: 2, cols: 2, spec: '二寸', total_photos: 4 },
      ],
    });
  },

  /**
   * 选择照片
   */
  choosePhoto() {
    wx.chooseImage({
      count: 1,
      sizeType: ['original', 'compressed'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];
        this.setData({
          selectedPhoto: tempFilePath,
        });

        // 保存当前配置到全局
        app.setGlobalData('currentPhoto', {
          ...app.globalData.currentPhoto,
          imagePath: tempFilePath,
        });

        console.log('[照片] 已选择:', tempFilePath);
      },
      fail: (err) => {
        console.log('[照片] 取消选择:', err);
      },
    });
  },

  /**
   * 选择规格
   */
  selectSpec(e) {
    const spec = e.currentTarget.dataset.spec;
    const specInfo = this.data.specs.find(s => s.name === spec);

    this.setData({
      currentSpec: spec,
      currentSpecInfo: specInfo || null,
    });

    app.setGlobalData('currentPhoto', {
      ...app.globalData.currentPhoto,
      spec: spec,
    });

    console.log('[规格]', spec);
  },

  /**
   * 选择背景色
   */
  selectColor(e) {
    const color = e.currentTarget.dataset.color;
    this.setData({ currentColor: color });

    app.setGlobalData('currentPhoto', {
      ...app.globalData.currentPhoto,
      color: color,
    });

    console.log('[颜色]', color);
  },

  /**
   * 选择排版方案
   */
  selectLayout(e) {
    const layout = e.currentTarget.dataset.layout;
    this.setData({ currentLayout: layout });
    console.log('[排版]', layout || '无');
  },

  /**
   * 生成证件照
   */
  async generatePhoto() {
    const { selectedPhoto, currentSpec, currentColor, currentLayout } = this.data;

    if (!selectedPhoto) {
      wx.showToast({ title: '请先选择照片', icon: 'none' });
      return;
    }

    // 开始生成
    this.setData({
      generating: true,
      loadingText: '正在分析人脸...',
    });

    // 模拟进度更新
    const loadingSteps = [
      '正在检测人脸...',
      '正在抠图中...',
      '正在替换背景...',
      '正在裁剪调整...',
      '即将完成...',
    ];
    let stepIndex = 0;
    const progressTimer = setInterval(() => {
      if (stepIndex < loadingSteps.length) {
        this.setData({ loadingText: loadingSteps[stepIndex] });
        stepIndex++;
      }
    }, 800);

    try {
      const result = await api.processPhoto(
        selectedPhoto,
        currentSpec,
        currentColor,
        currentLayout || null,
      );

      clearInterval(progressTimer);

      if (result.status === 'ok' && result.image) {
        console.log('[生成] 成功! 耗时:', result.processing_time, 's');

        // 保存结果到全局数据
        app.setGlobalData('currentPhoto', {
          ...app.globalData.currentPhoto,
          resultImage: result.image,
          layoutImage: result.layout_image || '',
          processingTime: result.processing_time,
          compliance: result.compliance || null,
          mattingQuality: result.matting_quality || null,
        });

        // 跳转到结果页
        wx.navigateTo({
          url: '/pages/result/result',
        });
      } else {
        throw new Error(result.error || '处理失败');
      }
    } catch (err) {
      clearInterval(progressTimer);
      console.error('[生成] 失败:', err);

      wx.showModal({
        title: '生成失败',
        content: err.message || '请检查网络连接和后端服务状态',
        showCancel: false,
        confirmText: '知道了',
      });
    } finally {
      this.setData({ generating: false });
    }
  },

  /**
   * 分享
   */
  onShareAppMessage() {
    return {
      title: 'AI一照成证 — 智能证件照生成',
      path: '/pages/index/index',
      imageUrl: '',
    };
  },
});
