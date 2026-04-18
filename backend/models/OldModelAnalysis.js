const mongoose = require('mongoose');

const oldModelAnalysisSchema = new mongoose.Schema({
  post_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Post',
    required: true
  },
  is_fake: Boolean,
  confidence: Number,
  frames_analysis: [{
    frame: String,
    confidence: Number,
    is_fake: Boolean,
    frame_path: String
  }],
  summary: {
    status: String,
    confidence_percentage: Number,
    total_frames: Number,
    real_frames: Number,
    fake_frames: Number
  }
}, { timestamps: true });

module.exports = mongoose.model('OldModelAnalysis', oldModelAnalysisSchema); 