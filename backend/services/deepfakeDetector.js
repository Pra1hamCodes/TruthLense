const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class DeepfakeDetector {
  constructor() {
    this.flaskServerUrl = (process.env.FLASK_CNN_URL || 'http://localhost:5000').replace(/\/+$/, '');
  }

  async analyze_video(videoPath) {
    try {
      const formData = new FormData();
      formData.append('video', fs.createReadStream(videoPath));

      const response = await axios.post(`${this.flaskServerUrl}/analyze`, formData, {
        headers: {
          ...formData.getHeaders()
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 300000
      });

      return response.data;
    } catch (error) {
      console.error('Error analyzing video:', error);
      throw error;
    }
  }
}

module.exports = DeepfakeDetector; 