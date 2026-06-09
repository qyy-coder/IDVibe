/**
 * AI一照成证 — 微信小程序
 *
 * 智能证件照生成系统
 * 团队: 肖清越 / 高泽宇 / 龚剑
 * 版本: v0.1.0 (P0)
 */

App({
  /**
   * 全局数据
   */
  globalData: {
    // API 服务器地址（修改为实际部署地址）
    apiBaseUrl: 'http://localhost:8000',

    // 用户信息
    userInfo: null,

    // 当前处理的证件照信息
    currentPhoto: {
      spec: '一寸',
      color: 'white',
      imagePath: '',
      resultImage: '',
      layoutImage: '',
    },
  },

  /**
   * 小程序启动
   */
  onLaunch() {
    console.log('[AI一照成证] 小程序启动 v0.1.0');

    // 检查 API 服务器连通性
    this.checkServerHealth();
  },

  /**
   * 检查后端服务健康状态
   */
  checkServerHealth() {
    wx.request({
      url: `${this.globalData.apiBaseUrl}/api/health`,
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200 && res.data.status === 'healthy') {
          console.log('[健康检查] 后端服务正常');
        } else {
          console.warn('[健康检查] 后端服务异常:', res.data);
        }
      },
      fail: (err) => {
        console.error('[健康检查] 无法连接后端服务:', err);
      },
    });
  },

  /**
   * 获取全局数据
   */
  getGlobalData(key) {
    return this.globalData[key];
  },

  /**
   * 设置全局数据
   */
  setGlobalData(key, value) {
    this.globalData[key] = value;
  },
});
