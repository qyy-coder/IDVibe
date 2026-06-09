/**
 * AI一照成证 — 结果页逻辑
 */

const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    resultImage: '',
    layoutImage: '',
    specLabel: '',
    colorLabel: '',
    processingTime: 0,
    saving: false,
    compliance: null,
    mattingQuality: null,
    showComplianceDetail: false,
  },

  onLoad() {
    // 从全局数据获取结果
    const currentPhoto = app.globalData.currentPhoto;
    console.log('[结果页] 加载结果:', currentPhoto.spec, currentPhoto.color);

    // 颜色标签映射
    const colorLabels = {
      white: '白色',
      blue: '蓝色',
      red: '红色',
    };

    this.setData({
      resultImage: currentPhoto.resultImage || '',
      layoutImage: currentPhoto.layoutImage || '',
      specLabel: currentPhoto.spec || '一寸',
      colorLabel: colorLabels[currentPhoto.color] || currentPhoto.color || '白色',
      processingTime: currentPhoto.processingTime || 0,
      compliance: currentPhoto.compliance || null,
      mattingQuality: currentPhoto.mattingQuality || null,
    });

    // 验证数据完整性
    if (!this.data.resultImage) {
      wx.showToast({
        title: '未找到结果数据',
        icon: 'error',
        duration: 2000,
      });
      setTimeout(() => {
        wx.navigateBack();
      }, 2000);
    }
  },

  /**
   * 保存证件照到相册
   */
  async savePhoto() {
    if (this.data.saving) return;

    this.setData({ saving: true });

    try {
      await api.saveToAlbum(this.data.resultImage);

      wx.showToast({
        title: '已保存到相册',
        icon: 'success',
        duration: 2000,
      });

      // 震动反馈
      wx.vibrateShort({ type: 'medium' });
    } catch (err) {
      console.error('[保存] 失败:', err);
      wx.showToast({
        title: err.message || '保存失败',
        icon: 'error',
        duration: 2000,
      });
    } finally {
      this.setData({ saving: false });
    }
  },

  /**
   * 保存排版照到相册
   */
  async saveLayout() {
    if (this.data.saving || !this.data.layoutImage) return;

    this.setData({ saving: true });

    try {
      await api.saveToAlbum(this.data.layoutImage);

      wx.showToast({
        title: '排版照已保存',
        icon: 'success',
        duration: 2000,
      });

      wx.vibrateShort({ type: 'medium' });
    } catch (err) {
      console.error('[保存排版] 失败:', err);
      wx.showToast({
        title: err.message || '保存失败',
        icon: 'error',
        duration: 2000,
      });
    } finally {
      this.setData({ saving: false });
    }
  },

  /**
   * 切换合规检测详情
   */
  toggleComplianceDetail() {
    this.setData({
      showComplianceDetail: !this.data.showComplianceDetail,
    });
  },

  /**
   * 返回首页继续生成
   */
  goBack() {
    wx.navigateBack();
  },

  /**
   * 长按图片保存（原生支持）
   */
  onImageLongPress(e) {
    const url = e.currentTarget.dataset.url;
    if (!url) return;

    wx.showActionSheet({
      itemList: ['保存图片到相册'],
      success: (res) => {
        if (res.tapIndex === 0) {
          this.savePhoto();
        }
      },
    });
  },

  /**
   * 分享结果
   */
  onShareAppMessage() {
    return {
      title: '我用 AI一照成证 生成了证件照！',
      path: '/pages/index/index',
      imageUrl: '',
    };
  },
});
