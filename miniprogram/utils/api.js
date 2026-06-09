/**
 * API 客户端 — 与后端服务通信
 *
 * 封装所有网络请求，处理错误和重试逻辑。
 */

const app = getApp();

/**
 * 基础请求封装
 */
function request(endpoint, options = {}) {
  const baseUrl = app.globalData.apiBaseUrl || 'http://localhost:8000';
  const url = `${baseUrl}${endpoint}`;

  return new Promise((resolve, reject) => {
    wx.request({
      url,
      method: options.method || 'GET',
      data: options.data || {},
      header: {
        'Content-Type': options.isFormData
          ? 'multipart/form-data'
          : 'application/json',
        ...options.header,
      },
      timeout: options.timeout || 30000,
      success: (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data);
        } else {
          const error = new Error(
            res.data?.detail || res.data?.error || `请求失败 (${res.statusCode})`
          );
          error.statusCode = res.statusCode;
          error.data = res.data;
          reject(error);
        }
      },
      fail: (err) => {
        const error = new Error('网络连接失败，请检查网络设置');
        error.original = err;
        reject(error);
      },
    });
  });
}

/**
 * 健康检查
 */
function checkHealth() {
  return request('/api/health');
}

/**
 * 获取证件照规格列表
 */
function getSpecs() {
  return request('/api/specs');
}

/**
 * 获取背景色列表
 */
function getColors() {
  return request('/api/colors');
}

/**
 * 获取排版方案列表
 */
function getLayouts() {
  return request('/api/layouts');
}

/**
 * 上传图片并生成证件照
 *
 * @param {string} imagePath - 本地图片路径
 * @param {string} spec - 规格名称
 * @param {string} color - 背景色
 * @param {string} layout - 排版方案（可选）
 * @param {Function} onProgress - 上传进度回调
 * @returns {Promise<Object>} 处理结果
 */
function processPhoto(imagePath, spec = '一寸', color = 'white', layout = null, onProgress) {
  return new Promise((resolve, reject) => {
    // 先读取文件为 base64（微信小程序不支持直接上传文件）
    wx.getFileSystemManager().readFile({
      filePath: imagePath,
      encoding: 'base64',
      success: (fileRes) => {
        const ext = imagePath.split('.').pop().toLowerCase() || 'jpg';
        const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
        const base64Data = `data:${mimeType};base64,${fileRes.data}`;

        // 调用 Base64 接口
        request('/api/process_base64', {
          method: 'POST',
          data: {
            image: base64Data,
            spec: spec,
            color: color,
            layout: layout,
          },
          timeout: 60000, // 60秒超时
        })
          .then(resolve)
          .catch(reject);
      },
      fail: (err) => {
        reject(new Error('读取图片失败: ' + (err.errMsg || '未知错误')));
      },
    });
  });
}

/**
 * 保存图片到相册
 *
 * @param {string} base64Image - Base64 编码的图片
 * @returns {Promise<string>} 临时文件路径
 */
function saveToTempFile(base64Image) {
  return new Promise((resolve, reject) => {
    // 解析 base64
    let b64Data = base64Image;
    if (base64Image.startsWith('data:image')) {
      b64Data = base64Image.split(',')[1];
    }

    const filePath = `${wx.env.USER_DATA_PATH}/idphoto_${Date.now()}.png`;

    wx.getFileSystemManager().writeFile({
      filePath: filePath,
      data: b64Data,
      encoding: 'base64',
      success: () => {
        resolve(filePath);
      },
      fail: (err) => {
        reject(new Error('保存临时文件失败: ' + (err.errMsg || '未知错误')));
      },
    });
  });
}

/**
 * 保存到系统相册
 *
 * @param {string} base64Image - Base64 编码的图片
 */
async function saveToAlbum(base64Image) {
  const tempPath = await saveToTempFile(base64Image);

  return new Promise((resolve, reject) => {
    wx.saveImageToPhotosAlbum({
      filePath: tempPath,
      success: () => {
        resolve();
      },
      fail: (err) => {
        if (err.errMsg && err.errMsg.includes('auth deny')) {
          // 权限被拒绝，引导用户打开设置
          wx.showModal({
            title: '需要相册权限',
            content: '请允许小程序访问您的相册，以便保存证件照',
            success: (modalRes) => {
              if (modalRes.confirm) {
                wx.openSetting();
              }
            },
          });
          reject(new Error('相册权限未授权'));
        } else {
          reject(new Error('保存失败: ' + (err.errMsg || '未知错误')));
        }
      },
    });
  });
}

module.exports = {
  request,
  checkHealth,
  getSpecs,
  getColors,
  getLayouts,
  processPhoto,
  saveToTempFile,
  saveToAlbum,
};
