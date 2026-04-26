const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class DeepfakeDetector {
  constructor() {
    this.flaskCnnUrl = (process.env.FLASK_CNN_URL || 'http://localhost:5000').replace(/\/+$/, '');
    this.flaskVitUrl = (process.env.FLASK_VIT_URL || 'http://localhost:5001').replace(/\/+$/, '');
    this.flaskCnnTimeoutMs = Number(process.env.FLASK_CNN_TIMEOUT_MS) || 300000;
    this.flaskVitTimeoutMs = Number(process.env.FLASK_VIT_TIMEOUT_MS) || 300000;
  }

  async analyzeWithUrl(videoPath, serverUrl, timeoutMs) {
    try {
      const formData = new FormData();
      formData.append('video', fs.createReadStream(videoPath));

      const response = await axios.post(`${serverUrl}/analyze`, formData, {
        headers: {
          ...formData.getHeaders()
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: timeoutMs
      });

      return response.data;
    } catch (error) {
      console.error(`Error analyzing video with ${serverUrl}:`, error?.message || error);
      throw error;
    }
  }

  async analyze_cnn(videoPath) {
    return this.analyzeWithUrl(videoPath, this.flaskCnnUrl, this.flaskCnnTimeoutMs);
  }

  async analyze_vit(videoPath) {
    return this.analyzeWithUrl(videoPath, this.flaskVitUrl, this.flaskVitTimeoutMs);
  }

  async analyze_pipeline(videoPath) {
    const [cnnResult, vitResult] = await Promise.all([
      this.analyze_cnn(videoPath),
      this.analyze_vit(videoPath)
    ]);

    return {
      cnn_model: cnnResult,
      vit_model: vitResult
    };
  }

  async analyze_video(videoPath) {
    return this.analyze_pipeline(videoPath);
  }
}

module.exports = DeepfakeDetector; 