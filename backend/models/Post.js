const mongoose = require('mongoose');

const postSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true
  },
  content: {
    type: String,
    required: true
  },
  media_url: {
    type: String,
    required: true
  },
  media_type: {
    type: String,
    enum: ['image', 'video'],
    required: true
  },
  creator: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  likes: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }],
  comments: [{
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true
    },
    content: {
      type: String,
      required: true
    },
    createdAt: {
      type: Date,
      default: Date.now
    }
  }],
  analysis_status: {
    type: String,
    enum: ['none', 'processing', 'completed', 'failed'],
    default: 'none'
  },
  deepfake_analysis: {
    is_fake: Boolean,
    confidence: Number,
    frames_analysis: [{
      frame: String,
      confidence: Number,
      is_fake: Boolean,
      frame_path: String
    }],
    summary: {
      status: String,  // "REAL" or "FAKE"
      confidence_percentage: Number,
      total_frames: Number,
      real_frames: Number,
      fake_frames: Number
    }
  }
}, { 
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

module.exports = mongoose.model('Post', postSchema);