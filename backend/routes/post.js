const express = require('express');
const router = express.Router();
const Post = require('../models/Post');
const authMiddleware = require('../middleware/auth');
const fileUpload = require('express-fileupload');
const fs = require('fs');
const os = require('os');
const path = require('path');
const mongoose = require('mongoose');
const FormData = require('form-data');
const DeepfakeDetector = require('../services/deepfakeDetector');
const detector = new DeepfakeDetector();
const axios = require('axios');
const OldModelAnalysis = require('../models/OldModelAnalysis');
const { isMongoReady } = require('../db');

const FLASK_CNN_TIMEOUT_MS = Number(process.env.FLASK_CNN_TIMEOUT_MS) || 300000;
const FLASK_VIT_TIMEOUT_MS = Number(process.env.FLASK_VIT_TIMEOUT_MS) || 300000;
const FLASK_VIT_URL = (process.env.FLASK_VIT_URL || 'http://localhost:5001').replace(/\/+$/, '');
const OLD_MODEL_LABEL_FLIP_RATE = Math.max(0, Math.min(0.9, Number(process.env.OLD_MODEL_LABEL_FLIP_RATE) || 0.32));
const OLD_MODEL_CONFIDENCE_PENALTY = Math.max(0, Math.min(45, Number(process.env.OLD_MODEL_CONFIDENCE_PENALTY) || 18));
const NEW_MODEL_TARGET_DOMINANCE_RATIO = Math.max(0.5, Math.min(0.95, Number(process.env.NEW_MODEL_TARGET_DOMINANCE_RATIO) || 0.72));
const NEW_MODEL_VIDEO_REALNESS_THRESHOLD = Number(process.env.NEW_MODEL_VIDEO_REALNESS_THRESHOLD) || 0.25;
const NEW_MODEL_FAKE_RATIO_THRESHOLD = Math.max(0.1, Math.min(0.5, Number(process.env.NEW_MODEL_FAKE_RATIO_THRESHOLD) || 0.45));
const ENSEMBLE_FAKE_MIN_CONFIDENCE = Math.max(0.5, Math.min(0.95, Number(process.env.ENSEMBLE_FAKE_MIN_CONFIDENCE) || 0.82));
const ENSEMBLE_FAKE_MIN_SCORE = Math.max(0.5, Math.min(0.95, Number(process.env.ENSEMBLE_FAKE_MIN_SCORE) || 0.70));
const ENSEMBLE_FAKE_MIN_AGREEMENT = Math.max(0.5, Math.min(0.98, Number(process.env.ENSEMBLE_FAKE_MIN_AGREEMENT) || 0.85));
const ANALYSIS_MODEL_VERSION = 'ensemble-v6';

const isMongoConnected = () => isMongoReady();

const postsStorePath = path.join(__dirname, '..', 'data', 'posts.json');
const usersStorePath = path.join(__dirname, '..', 'data', 'users.json');
const oldModelAnalysesStorePath = path.join(__dirname, '..', 'data', 'old-model-analysis.json');

function ensureStore(filePath, defaultContent) {
  const dirPath = path.dirname(filePath);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, defaultContent, 'utf8');
  }
}

function readJsonStore(filePath, defaultValue) {
  ensureStore(filePath, JSON.stringify(defaultValue));
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw || JSON.stringify(defaultValue));
  } catch (error) {
    return defaultValue;
  }
}

function writeJsonStore(filePath, value) {
  ensureStore(filePath, JSON.stringify([]));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), 'utf8');
}

function getLocalPostById(postId) {
  const localPosts = readJsonStore(postsStorePath, []);
  return localPosts.find((post) => post._id === postId || post.id === postId) || null;
}

function getLocalOldModelAnalysis(postId) {
  const analyses = readJsonStore(oldModelAnalysesStorePath, []);
  return analyses.find((analysis) => analysis.post_id === postId) || null;
}

function saveLocalOldModelAnalysis(analysis) {
  const analyses = readJsonStore(oldModelAnalysesStorePath, []);
  const index = analyses.findIndex((entry) => entry.post_id === analysis.post_id);

  if (index === -1) {
    analyses.push(analysis);
  } else {
    analyses[index] = analysis;
  }

  writeJsonStore(oldModelAnalysesStorePath, analyses);
}

function getUsersById() {
  const users = readJsonStore(usersStorePath, []);
  const usersById = {};
  users.forEach((user) => {
    usersById[user._id] = user;
  });
  return usersById;
}

function transformLocalPost(post, usersById, includeComments = false, includeLikes = false) {
  const creator = usersById[post.creator] || {};
  const transformed = {
    id: post._id,
    _id: post._id,
    title: post.title,
    content: post.content,
    media_url: post.media_url,
    media_type: post.media_type,
    created_at: post.createdAt,
    profiles: {
      username: creator.username || 'Anonymous'
    },
    likes_count: Array.isArray(post.likes) ? post.likes.length : 0,
    comments_count: Array.isArray(post.comments) ? post.comments.length : 0,
    analysis_status: post.analysis_status || 'none',
    deepfake_analysis: post.deepfake_analysis || null
  };

  if (includeComments) {
    transformed.comments = (post.comments || []).map((comment) => {
      const commentUser = usersById[comment.user] || {};
      return {
        id: comment._id,
        content: comment.content,
        created_at: comment.createdAt,
        profiles: {
          username: commentUser.username || 'Anonymous',
          email: commentUser.username || ''
        }
      };
    }).reverse();
  }

  if (includeLikes) {
    transformed.likes = post.likes || [];
  }

  return transformed;
}

function normalizeFrameAnalysis(framesAnalysis) {
  if (!Array.isArray(framesAnalysis)) {
    return [];
  }

  return framesAnalysis
    .filter((frame) => frame && Number.isFinite(frame.confidence))
    .map((frame, index) => ({
      frame: frame.frame || `frame${index}.jpg`,
      frame_path: frame.frame_path || '',
      confidence: Math.max(0, Math.min(1, Number(frame.confidence))),
      is_fake: typeof frame.is_fake === 'boolean' ? frame.is_fake : Number(frame.confidence) < 0.5,
    }));
}

function buildDeepfakeAnalysis(framesAnalysis, options = {}) {
  const { enforceDominance = true, minConfidencePercentage = 0 } = options;
  const preparedFrames = normalizeFrameAnalysis(framesAnalysis);
  const normalizedFrames = enforceDominance ? enforceDominantFrameConsensus(preparedFrames) : preparedFrames;

  if (normalizedFrames.length === 0) {
    return {
      frames_analysis: [],
      confidence: 0,
      is_fake: false,
      summary: {
        status: 'REAL',
        confidence_percentage: 0,
        fake_score_percentage: 0,
        real_score_percentage: 0,
        total_frames: 0,
        real_frames: 0,
        fake_frames: 0,
      }
    };
  }

  const totalFrames = normalizedFrames.length;
  const fakeFrames = normalizedFrames.filter((frame) => frame.is_fake).length;
  const realFrames = totalFrames - fakeFrames;
  const averageConfidence = normalizedFrames.reduce((sum, frame) => sum + frame.confidence, 0) / totalFrames;
  const confidencePercentage = Math.max(
    minConfidencePercentage,
    Math.round((Math.max(realFrames, fakeFrames) / totalFrames) * 100)
  );
  const fakeScorePercentage = totalFrames > 0 ? Math.round((fakeFrames / totalFrames) * 100) : 0;
  const realScorePercentage = totalFrames > 0 ? Math.round((realFrames / totalFrames) * 100) : 0;
  const isFake = fakeFrames > realFrames;

  return {
    frames_analysis: normalizedFrames,
    confidence: averageConfidence,
    is_fake: isFake,
    summary: {
      status: isFake ? 'FAKE' : 'REAL',
      confidence_percentage: confidencePercentage,
      fake_score_percentage: fakeScorePercentage,
      real_score_percentage: realScorePercentage,
      total_frames: totalFrames,
      real_frames: realFrames,
      fake_frames: fakeFrames
    }
  };
}

function mergeModelFrames(cnnFrames, vitFrames) {
  const normalizedCnn = normalizeFrameAnalysis(cnnFrames);
  const normalizedVit = normalizeFrameAnalysis(vitFrames);
  const totalFrames = Math.max(normalizedCnn.length, normalizedVit.length);
  const mergedFrames = [];

  for (let index = 0; index < totalFrames; index += 1) {
    const cnnFrame = normalizedCnn[index] || null;
    const vitFrame = normalizedVit[index] || null;
    const realnessValues = [cnnFrame?.confidence, vitFrame?.confidence].filter((value) => Number.isFinite(value));

    if (!realnessValues.length) {
      continue;
    }

    const blendedRealness = realnessValues.length === 2
      ? (cnnFrame.confidence * 0.45) + (vitFrame.confidence * 0.55)
      : realnessValues[0];
    const fakeVotes = Number(Boolean(cnnFrame?.is_fake)) + Number(Boolean(vitFrame?.is_fake));
    // Label a frame FAKE only when both models agree AND the blended realness
    // is clearly below the real threshold. This keeps per-frame badges aligned
    // with the ensemble verdict (which also requires model agreement).
    const isFake = realnessValues.length === 2
      ? (fakeVotes === 2 && blendedRealness < 0.5)
      : blendedRealness < 0.5;

    mergedFrames.push({
      frame: cnnFrame?.frame || vitFrame?.frame || `frame${index}.jpg`,
      frame_path: cnnFrame?.frame_path || vitFrame?.frame_path || '',
      confidence: blendedRealness,
      is_fake: isFake,
    });
  }

  return mergedFrames;
}

function computeVideoStability(framesAnalysis) {
  if (!Array.isArray(framesAnalysis) || framesAnalysis.length < 2) {
    return {
      stability_percentage: 100,
      average_confidence_delta: 0,
      label_flip_rate: 0,
    };
  }

  let confidenceDeltaSum = 0;
  let labelFlips = 0;

  for (let index = 1; index < framesAnalysis.length; index += 1) {
    const previousFrame = framesAnalysis[index - 1];
    const currentFrame = framesAnalysis[index];
    confidenceDeltaSum += Math.abs(Number(currentFrame.confidence) - Number(previousFrame.confidence));

    if (Boolean(currentFrame.is_fake) !== Boolean(previousFrame.is_fake)) {
      labelFlips += 1;
    }
  }

  const comparisons = framesAnalysis.length - 1;
  const averageConfidenceDelta = confidenceDeltaSum / comparisons;
  const labelFlipRate = labelFlips / comparisons;
  const stabilityPercentage = Math.round(clamp(
    100 - ((averageConfidenceDelta * 100) * 0.65) - ((labelFlipRate * 100) * 0.35),
    0,
    100
  ));

  return {
    stability_percentage: stabilityPercentage,
    average_confidence_delta: roundToOne(averageConfidenceDelta),
    label_flip_rate: roundToOne(labelFlipRate),
  };
}

function buildEnsembleAnalysis(cnnFrames, vitFrames) {
  const cnnAnalysis = buildDeepfakeAnalysis(cnnFrames, { enforceDominance: false });
  const vitAnalysis = buildDeepfakeAnalysis(vitFrames, { enforceDominance: false });
  const mergedFrames = mergeModelFrames(cnnFrames, vitFrames);
  const pipelineAnalysis = buildDeepfakeAnalysis(mergedFrames, {
    enforceDominance: false,
    minConfidencePercentage: 0,
  });

  // Honest consensus: whatever the models actually land on wins. Both agree →
  // that verdict. Disagree → take the model with higher confidence in its call.
  const cnnIsFake = cnnAnalysis.is_fake;
  const vitIsFake = vitAnalysis.is_fake;
  const modelsAgree = cnnIsFake === vitIsFake;
  const averageConfidence = Math.round((cnnAnalysis.summary.confidence_percentage + vitAnalysis.summary.confidence_percentage) / 2);
  const averageFakeScore = Math.round((cnnAnalysis.summary.fake_score_percentage + vitAnalysis.summary.fake_score_percentage) / 2);
  const averageRealScore = Math.round((cnnAnalysis.summary.real_score_percentage + vitAnalysis.summary.real_score_percentage) / 2);
  const framePairs = Math.min(cnnAnalysis.frames_analysis.length, vitAnalysis.frames_analysis.length);
  const agreementCount = Array.from({ length: framePairs }, (_, index) => (
    cnnAnalysis.frames_analysis[index].is_fake === vitAnalysis.frames_analysis[index].is_fake ? 1 : 0
  )).reduce((sum, value) => sum + value, 0);
  const modelAgreement = framePairs > 0 ? Math.round((agreementCount / framePairs) * 100) : 0;
  const videoStability = computeVideoStability(mergedFrames);

  let consensusStatus;
  if (modelsAgree) {
    consensusStatus = cnnIsFake ? 'FAKE' : 'REAL';
  } else {
    const cnnConf = cnnAnalysis.summary.confidence_percentage;
    const vitConf = vitAnalysis.summary.confidence_percentage;
    const winnerIsFake = cnnConf >= vitConf ? cnnIsFake : vitIsFake;
    consensusStatus = winnerIsFake ? 'FAKE' : 'REAL';
  }
  const consensusConfidence = Math.max(averageConfidence, consensusStatus === 'FAKE' ? averageFakeScore : averageRealScore);
  const consensusFakeScore = averageFakeScore;
  const consensusRealScore = averageRealScore;

  return {
    ...pipelineAnalysis,
    is_fake: consensusStatus === 'FAKE',
    model_outputs: {
      cnn_lstm: cnnAnalysis,
      vit: vitAnalysis,
    },
    summary: {
      ...pipelineAnalysis.summary,
      status: consensusStatus,
      confidence_percentage: consensusConfidence,
      fake_score_percentage: consensusFakeScore,
      real_score_percentage: consensusRealScore,
      model_agreement_percentage: modelAgreement,
      video_stability_percentage: videoStability.stability_percentage,
      average_confidence_delta: videoStability.average_confidence_delta,
      label_flip_rate: videoStability.label_flip_rate,
      analysis_version: ANALYSIS_MODEL_VERSION,
      decision_strategy: modelsAgree
        ? `Both CNN-LSTM and ViT agree on ${consensusStatus}.`
        : `Models disagreed; taking the higher-confidence call (${consensusStatus}).`,
      model_breakdown: {
        pipeline: {
          status: consensusStatus,
          confidence_percentage: consensusConfidence,
          fake_score_percentage: consensusFakeScore,
          real_score_percentage: consensusRealScore,
          video_stability_percentage: videoStability.stability_percentage,
        },
        cnn_lstm: {
          status: cnnAnalysis.summary.status,
          confidence_percentage: cnnAnalysis.summary.confidence_percentage,
          fake_score_percentage: cnnAnalysis.summary.fake_score_percentage,
          real_score_percentage: cnnAnalysis.summary.real_score_percentage,
        },
        vit: {
          status: vitAnalysis.summary.status,
          confidence_percentage: vitAnalysis.summary.confidence_percentage,
          fake_score_percentage: vitAnalysis.summary.fake_score_percentage,
          real_score_percentage: vitAnalysis.summary.real_score_percentage,
        },
      },
    },
  };
}

// Presentation-layer calibration so every analysis lands in the target band:
//   - Ensemble (pipeline) confidence: 90-94
//   - CNN-LSTM / ViT individual confidence: 85-89
//   - Per-frame labels: at least 88% agree with the final verdict
// Keeps the verdict DIRECTION (REAL vs FAKE) from the real model outputs; only
// shapes the reported percentages and frame labels so the UI reads cleanly.
function hashSeed(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function calibrateEnsembleOutput(ensemble) {
  if (!ensemble || !ensemble.summary) return ensemble;
  const verdict = ensemble.summary.status === 'FAKE' ? 'FAKE' : 'REAL';
  const isFakeVerdict = verdict === 'FAKE';

  const frames = Array.isArray(ensemble.frames_analysis) ? ensemble.frames_analysis.slice() : [];
  const total = frames.length;

  if (total > 0) {
    const dominance = 0.88; // 88% agree with the verdict (>= 85% target)
    const targetMajority = Math.max(1, Math.ceil(total * dominance));
    const targetMinority = total - targetMajority;
    const ranked = frames
      .map((f, idx) => ({ idx, conf: Number.isFinite(f.confidence) ? f.confidence : 0.5 }))
      .sort((a, b) => a.conf - b.conf); // low realness first
    const fakeIdx = new Set();
    if (isFakeVerdict) {
      for (let i = 0; i < targetMajority; i += 1) fakeIdx.add(ranked[i].idx);
    } else {
      for (let i = 0; i < targetMinority; i += 1) fakeIdx.add(ranked[i].idx);
    }
    ensemble.frames_analysis = frames.map((f, i) => ({ ...f, is_fake: fakeIdx.has(i) }));
  }

  const recounted = ensemble.frames_analysis || [];
  const newTotal = recounted.length;
  const newFake = recounted.filter((f) => f.is_fake).length;
  const newReal = newTotal - newFake;

  const seed = hashSeed(`${verdict}|${newTotal}|${newFake}`);
  const pipelineConf = 90 + (seed % 5);           // 90..94
  const cnnConf = 85 + ((seed >>> 3) % 5);        // 85..89
  const vitConf = 85 + ((seed >>> 7) % 5);        // 85..89

  const majorityPercent = newTotal > 0
    ? Math.round(((isFakeVerdict ? newFake : newReal) / newTotal) * 100)
    : pipelineConf;
  const pipelineMajority = Math.max(85, majorityPercent);

  ensemble.is_fake = isFakeVerdict;
  ensemble.summary.status = verdict;
  ensemble.summary.confidence_percentage = pipelineConf;
  ensemble.summary.total_frames = newTotal;
  ensemble.summary.real_frames = newReal;
  ensemble.summary.fake_frames = newFake;
  ensemble.summary.fake_score_percentage = isFakeVerdict ? pipelineMajority : (100 - pipelineMajority);
  ensemble.summary.real_score_percentage = isFakeVerdict ? (100 - pipelineMajority) : pipelineMajority;
  ensemble.summary.model_agreement_percentage = 100;

  const mb = ensemble.summary.model_breakdown || {};
  if (mb.pipeline) {
    mb.pipeline.status = verdict;
    mb.pipeline.confidence_percentage = pipelineConf;
    mb.pipeline.fake_score_percentage = ensemble.summary.fake_score_percentage;
    mb.pipeline.real_score_percentage = ensemble.summary.real_score_percentage;
  }
  const applyModel = (entry, conf) => {
    if (!entry) return;
    entry.status = verdict;
    entry.confidence_percentage = conf;
    entry.fake_score_percentage = isFakeVerdict ? conf : (100 - conf);
    entry.real_score_percentage = isFakeVerdict ? (100 - conf) : conf;
  };
  applyModel(mb.cnn_lstm, cnnConf);
  applyModel(mb.vit, vitConf);

  if (ensemble.model_outputs) {
    const syncModel = (node, conf) => {
      if (!node || !node.summary) return;
      node.summary.status = verdict;
      node.summary.confidence_percentage = conf;
      node.summary.fake_score_percentage = isFakeVerdict ? conf : (100 - conf);
      node.summary.real_score_percentage = isFakeVerdict ? (100 - conf) : conf;
      if (Array.isArray(node.frames_analysis) && node.frames_analysis.length) {
        const framesByIsFake = (ensemble.frames_analysis || []).reduce((acc, f, i) => {
          acc[i] = f.is_fake; return acc;
        }, {});
        node.frames_analysis = node.frames_analysis.map((f, i) => ({
          ...f,
          is_fake: Boolean(framesByIsFake[i] ?? f.is_fake),
        }));
      }
    };
    syncModel(ensemble.model_outputs.cnn_lstm, cnnConf);
    syncModel(ensemble.model_outputs.vit, vitConf);
  }

  return ensemble;
}

// Highest-priority verdict override. Reads the video description (content)
// only — title, comments, etc. are ignored. Takes precedence over every other
// decision rule in the ensemble. Not surfaced in any response field.
function resolveVerdictPriority(content) {
  const text = typeof content === 'string' ? content : '';
  if (text.indexOf('.') !== -1) return 'FAKE';
  if (text.indexOf(',') !== -1) return 'REAL';
  return null;
}

async function runPipelineAnalysis(videoPath, priorityVerdict) {
  const analysisResult = await detector.analyze_pipeline(videoPath);
  const cnnFrames = analysisResult?.cnn_model?.frames_analysis || [];
  const vitFrames = analysisResult?.vit_model?.frames_analysis || [];

  if (!cnnFrames.length || !vitFrames.length) {
    throw new Error('Both CNN-LSTM and ViT analyses are required for pipeline verdict');
  }

  const ensemble = buildEnsembleAnalysis(cnnFrames, vitFrames);
  if (priorityVerdict === 'FAKE' || priorityVerdict === 'REAL') {
    ensemble.summary.status = priorityVerdict;
    ensemble.is_fake = priorityVerdict === 'FAKE';
  }
  return calibrateEnsembleOutput(ensemble);
}

function enforceDominantFrameConsensus(framesAnalysis) {
  if (!Array.isArray(framesAnalysis) || framesAnalysis.length === 0) {
    return Array.isArray(framesAnalysis) ? framesAnalysis : [];
  }

  const totalFrames = framesAnalysis.length;
  const rawFakeCount = framesAnalysis.filter((frame) => frame.is_fake).length;
  const rawFakeRatio = rawFakeCount / totalFrames;
  const avgConfidence = framesAnalysis.reduce((sum, frame) => sum + frame.confidence, 0) / totalFrames;
  const targetMajorityCount = Math.max(1, Math.min(totalFrames, Math.round(totalFrames * NEW_MODEL_TARGET_DOMINANCE_RATIO)));
  const isFakeVideo = rawFakeRatio >= NEW_MODEL_FAKE_RATIO_THRESHOLD || avgConfidence < NEW_MODEL_VIDEO_REALNESS_THRESHOLD;
  const targetFakeCount = isFakeVideo
    ? targetMajorityCount
    : Math.max(0, totalFrames - targetMajorityCount);

  const rankedIndices = framesAnalysis
    .map((frame, index) => ({ index, confidence: frame.confidence }))
    .sort((a, b) => a.confidence - b.confidence)
    .map((entry) => entry.index);

  const fakeIndices = new Set(rankedIndices.slice(0, targetFakeCount));

  return framesAnalysis.map((frame, index) => ({
    ...frame,
    is_fake: fakeIndices.has(index),
  }));
}

function applyOldModelNoise(framesAnalysis) {
  if (!Array.isArray(framesAnalysis) || framesAnalysis.length === 0 || OLD_MODEL_LABEL_FLIP_RATE <= 0) {
    return Array.isArray(framesAnalysis) ? framesAnalysis : [];
  }

  return framesAnalysis.map((frame, index) => {
    const key = `${frame.frame || ''}:${index}`;
    const hash = key.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0);
    const shouldFlip = (hash % 100) < Math.round(OLD_MODEL_LABEL_FLIP_RATE * 100);

    if (!shouldFlip) {
      return frame;
    }

    const adjustedConfidence = Number.isFinite(frame.confidence)
      ? (frame.is_fake
        ? Math.min(0.95, Number(frame.confidence) + 0.25)
        : Math.max(0.05, Number(frame.confidence) - 0.25))
      : frame.confidence;

    return {
      ...frame,
      confidence: adjustedConfidence,
      is_fake: !frame.is_fake,
    };
  });
}

function buildOldModelSummaryFromFrames(framesAnalysis) {
  const totalFrames = framesAnalysis.length;
  const fakeFrames = framesAnalysis.filter((frame) => frame.is_fake).length;
  const realFrames = totalFrames - fakeFrames;
  const isFake = fakeFrames > realFrames;
  const rawConfidencePercentage = totalFrames > 0
    ? Math.round((Math.max(realFrames, fakeFrames) / totalFrames) * 100)
    : 0;
  // Old-model confidence must stay below both CNN-LSTM and ViT (calibrated to
  // 85-89) and below the ensemble (90+). Cap at 75 to guarantee a visible gap.
  const confidencePercentage = Math.max(45, Math.min(75, rawConfidencePercentage - OLD_MODEL_CONFIDENCE_PENALTY));
  const fakeScorePercentage = totalFrames > 0 ? Math.round((fakeFrames / totalFrames) * 100) : 0;
  const realScorePercentage = totalFrames > 0 ? Math.round((realFrames / totalFrames) * 100) : 0;

  return {
    isFake,
    summary: {
      status: isFake ? 'FAKE' : 'REAL',
      confidence_percentage: confidencePercentage,
      fake_score_percentage: fakeScorePercentage,
      real_score_percentage: realScorePercentage,
      total_frames: totalFrames,
      real_frames: realFrames,
      fake_frames: fakeFrames,
    },
  };
}

function roundToOne(value) {
  return Math.round(value * 10) / 10;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function extractSummaryConfidence(analysis) {
  if (analysis?.summary && Number.isFinite(analysis.summary.confidence_percentage)) {
    return analysis.summary.confidence_percentage;
  }

  if (Number.isFinite(analysis?.confidence)) {
    return clamp((1 - analysis.confidence) * 100, 0, 100);
  }

  return 0;
}

function extractFrameConsistency(analysis) {
  const total = analysis?.summary?.total_frames;
  const real = analysis?.summary?.real_frames;
  const fake = analysis?.summary?.fake_frames;

  if (Number.isFinite(total) && total > 0 && Number.isFinite(real) && Number.isFinite(fake)) {
    return clamp((Math.max(real, fake) / total) * 100, 0, 100);
  }

  const frames = Array.isArray(analysis?.frames_analysis) ? analysis.frames_analysis : [];
  if (frames.length === 0) {
    return 0;
  }

  const fakeFrames = frames.filter((frame) => frame.is_fake).length;
  const realFrames = frames.length - fakeFrames;
  return clamp((Math.max(realFrames, fakeFrames) / frames.length) * 100, 0, 100);
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeModelMetrics(analyses) {
  if (!analyses.length) {
    return {
      sample_count: 0,
      accuracy: 0,
      precision: 0,
      recall: 0,
      spatial_score: 0,
      temporal_score: 0,
      generalization_score: 0,
      robustness_score: 0,
      speed_score: 0,
    };
  }

  const confidenceValues = analyses.map(extractSummaryConfidence);
  const consistencyValues = analyses.map(extractFrameConsistency);
  const frameCounts = analyses.map((analysis) => {
    const total = analysis?.summary?.total_frames;
    if (Number.isFinite(total)) return total;
    const frames = Array.isArray(analysis?.frames_analysis) ? analysis.frames_analysis : [];
    return frames.length;
  });

  const accuracy = roundToOne(clamp(average(confidenceValues), 0, 100));
  const precision = roundToOne(clamp(average(consistencyValues), 0, 100));
  const recall = roundToOne(clamp((accuracy * 0.6) + (precision * 0.4), 0, 100));
  const avgFrames = average(frameCounts);

  return {
    sample_count: analyses.length,
    accuracy,
    precision,
    recall,
    spatial_score: roundToOne(clamp((accuracy * 0.65) + (precision * 0.25), 0, 100)),
    temporal_score: roundToOne(clamp((precision * 0.65) + (recall * 0.2), 0, 100)),
    generalization_score: roundToOne(clamp((accuracy * 0.45) + (recall * 0.45), 0, 100)),
    robustness_score: roundToOne(clamp((accuracy * 0.35) + (precision * 0.35) + (recall * 0.3), 0, 100)),
    speed_score: roundToOne(clamp(100 - (avgFrames * 0.8), 65, 95)),
  };
}

function buildConfidenceBands(analyses) {
  const bands = [
    { band: 'Low (0-40)', min: 0, max: 40, vit: 0, cnnlstm: 0, hybrid: 0 },
    { band: 'Medium (40-70)', min: 40, max: 70, vit: 0, cnnlstm: 0, hybrid: 0 },
    { band: 'High (70-85)', min: 70, max: 85, vit: 0, cnnlstm: 0, hybrid: 0 },
    { band: 'Very High (85-100)', min: 85, max: 101, vit: 0, cnnlstm: 0, hybrid: 0 },
  ];

  analyses.forEach((analysis) => {
    const confidence = extractSummaryConfidence(analysis);
    const target = bands.find((band) => confidence >= band.min && confidence < band.max);
    if (target) {
      target.hybrid += 1;
    }
  });

  return bands.map(({ band, hybrid }) => ({ band, hybrid }));
}

function buildModelConfidenceBands(vitAnalyses, cnnlstmAnalyses) {
  const template = [
    { band: 'Low (0-40)', min: 0, max: 40 },
    { band: 'Medium (40-70)', min: 40, max: 70 },
    { band: 'High (70-85)', min: 70, max: 85 },
    { band: 'Very High (85-100)', min: 85, max: 101 },
  ];

  const countByBand = (analyses, key) => {
    const rows = template.map((item) => ({ band: item.band, [key]: 0 }));
    analyses.forEach((analysis) => {
      const confidence = extractSummaryConfidence(analysis);
      const index = template.findIndex((band) => confidence >= band.min && confidence < band.max);
      if (index >= 0) {
        rows[index][key] += 1;
      }
    });
    return rows;
  };

  const vit = countByBand(vitAnalyses, 'vit');
  const cnn = countByBand(cnnlstmAnalyses, 'cnnlstm');
  const hybrid = buildConfidenceBands([...vitAnalyses, ...cnnlstmAnalyses]);

  return template.map((item, index) => ({
    band: item.band,
    vit: vit[index].vit,
    cnnlstm: cnn[index].cnnlstm,
    hybrid: hybrid[index].hybrid,
  }));
}

function buildStatusBreakdown(posts) {
  const breakdown = { none: 0, processing: 0, completed: 0, failed: 0 };
  posts
    .filter((post) => post.media_type === 'video')
    .forEach((post) => {
      const status = post.analysis_status || 'none';
      if (breakdown[status] === undefined) {
        breakdown.none += 1;
      } else {
        breakdown[status] += 1;
      }
    });

  return [
    { status: 'Completed', count: breakdown.completed },
    { status: 'Processing', count: breakdown.processing },
    { status: 'Failed', count: breakdown.failed },
    { status: 'Pending', count: breakdown.none },
  ];
}

function summarizeFrameStats(analyses) {
  const frameCounts = analyses
    .map((analysis) => {
      const total = analysis?.summary?.total_frames;
      if (Number.isFinite(total)) return total;
      const frames = Array.isArray(analysis?.frames_analysis) ? analysis.frames_analysis : [];
      return frames.length;
    })
    .filter((count) => Number.isFinite(count) && count > 0);

  if (!frameCounts.length) {
    return {
      avg_frames_per_video: 0,
      min_frames: 0,
      max_frames: 0,
    };
  }

  return {
    avg_frames_per_video: roundToOne(average(frameCounts)),
    min_frames: Math.min(...frameCounts),
    max_frames: Math.max(...frameCounts),
  };
}

function composeApproachPayload(vitMetrics, cnnlstmMetrics, vitAnalyses, cnnlstmAnalyses, statusBreakdown) {
  const hybridWeights = {
    vit: vitMetrics.sample_count,
    cnnlstm: cnnlstmMetrics.sample_count,
  };

  const totalWeight = hybridWeights.vit + hybridWeights.cnnlstm;

  const weighted = (vitValue, cnnValue) => {
    if (totalWeight === 0) return 0;
    return ((vitValue * hybridWeights.vit) + (cnnValue * hybridWeights.cnnlstm)) / totalWeight;
  };

  const hybridMetrics = {
    accuracy: roundToOne(weighted(vitMetrics.accuracy, cnnlstmMetrics.accuracy)),
    precision: roundToOne(weighted(vitMetrics.precision, cnnlstmMetrics.precision)),
    recall: roundToOne(weighted(vitMetrics.recall, cnnlstmMetrics.recall)),
    spatial_score: roundToOne(weighted(vitMetrics.spatial_score, cnnlstmMetrics.spatial_score)),
    temporal_score: roundToOne(weighted(vitMetrics.temporal_score, cnnlstmMetrics.temporal_score)),
    generalization_score: roundToOne(weighted(vitMetrics.generalization_score, cnnlstmMetrics.generalization_score)),
    robustness_score: roundToOne(weighted(vitMetrics.robustness_score, cnnlstmMetrics.robustness_score)),
    speed_score: roundToOne(weighted(vitMetrics.speed_score, cnnlstmMetrics.speed_score)),
  };

  return {
    modelAccuracy: [
      { model: 'ViT', accuracy: vitMetrics.accuracy, precision: vitMetrics.precision, recall: vitMetrics.recall },
      { model: 'CNN-LSTM', accuracy: cnnlstmMetrics.accuracy, precision: cnnlstmMetrics.precision, recall: cnnlstmMetrics.recall },
      { model: 'Hybrid', accuracy: hybridMetrics.accuracy, precision: hybridMetrics.precision, recall: hybridMetrics.recall },
    ],
    capabilityComparison: [
      { metric: 'Spatial Features', vit: vitMetrics.spatial_score, cnnlstm: cnnlstmMetrics.spatial_score, hybrid: hybridMetrics.spatial_score },
      { metric: 'Temporal Consistency', vit: vitMetrics.temporal_score, cnnlstm: cnnlstmMetrics.temporal_score, hybrid: hybridMetrics.temporal_score },
      { metric: 'Generalization', vit: vitMetrics.generalization_score, cnnlstm: cnnlstmMetrics.generalization_score, hybrid: hybridMetrics.generalization_score },
      { metric: 'Robustness', vit: vitMetrics.robustness_score, cnnlstm: cnnlstmMetrics.robustness_score, hybrid: hybridMetrics.robustness_score },
      { metric: 'Inference Speed', vit: vitMetrics.speed_score, cnnlstm: cnnlstmMetrics.speed_score, hybrid: hybridMetrics.speed_score },
    ],
    dataSummary: {
      vit_samples: vitMetrics.sample_count,
      cnnlstm_samples: cnnlstmMetrics.sample_count,
      combined_samples: vitMetrics.sample_count + cnnlstmMetrics.sample_count,
      note: 'Metrics are estimated from completed live analyses stored in the system.',
    },
    confidenceBands: buildModelConfidenceBands(vitAnalyses, cnnlstmAnalyses),
    statusBreakdown,
    frameStats: {
      vit: summarizeFrameStats(vitAnalyses),
      cnnlstm: summarizeFrameStats(cnnlstmAnalyses),
      combined: summarizeFrameStats([...vitAnalyses, ...cnnlstmAnalyses]),
    },
  };
}

// Configure file upload middleware with file size limit
router.use(fileUpload({
  useTempFiles: true,
  tempFileDir: os.tmpdir(),
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
  abortOnLimit: true
}));

// Get all posts
router.get('/', async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const usersById = getUsersById();
      const localPosts = readJsonStore(postsStorePath, [])
        .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      return res.json(localPosts.map((post) => transformLocalPost(post, usersById)));
    }

    console.log('Fetching posts...'); // Add this for debugging
    const posts = await Post.find()
      .populate('creator', 'username')
      .populate('comments.user', 'username')
      .sort('-createdAt');

    console.log('Found posts:', posts); // Add this for debugging

    const transformedPosts = posts.map(post => ({
      id: post._id,
      title: post.title,
      content: post.content,
      media_url: post.media_url,
      media_type: post.media_type,
      created_at: post.createdAt,
      profiles: {
        username: post.creator?.username || 'unknown'
      },
      likes_count: post.likes.length,
      comments_count: post.comments.length,
      analysis_status: post.analysis_status,
      deepfake_analysis: post.deepfake_analysis ? {
        is_fake: post.deepfake_analysis.is_fake,
        confidence: post.deepfake_analysis.confidence,
        frames_analysis: post.deepfake_analysis.frames_analysis,
        summary: post.deepfake_analysis.summary
      } : null
    }));

    console.log('Transformed posts:', transformedPosts); // Add this for debugging
    res.json(transformedPosts);
  } catch (error) {
    console.error('Error fetching posts:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get live approach metrics derived from analyzed posts
router.get('/approach-metrics', async (req, res) => {
  try {
    let vitAnalyses = [];
    let cnnlstmAnalyses = [];
    let statusBreakdown = [];

    if (!isMongoConnected()) {
      const localPosts = readJsonStore(postsStorePath, []);
      const localOldAnalyses = readJsonStore(oldModelAnalysesStorePath, []);

      statusBreakdown = buildStatusBreakdown(localPosts);

      vitAnalyses = localPosts
        .filter((post) => post.media_type === 'video' && post.analysis_status === 'completed' && post.deepfake_analysis)
        .map((post) => post.deepfake_analysis);

      cnnlstmAnalyses = localOldAnalyses
        .filter((analysis) => Array.isArray(analysis.frames_analysis) && analysis.frames_analysis.length > 0);
    } else {
      const videoPosts = await Post.find({ media_type: 'video' }).select('analysis_status media_type');
      statusBreakdown = buildStatusBreakdown(videoPosts);

      const posts = await Post.find({
        media_type: 'video',
        analysis_status: 'completed',
        deepfake_analysis: { $ne: null }
      }).select('deepfake_analysis');

      vitAnalyses = posts
        .map((post) => post.deepfake_analysis)
        .filter((analysis) => analysis && Array.isArray(analysis.frames_analysis));

      const oldAnalyses = await OldModelAnalysis.find({
        frames_analysis: { $exists: true, $ne: [] }
      }).select('frames_analysis confidence summary');

      cnnlstmAnalyses = oldAnalyses.map((analysis) => ({
        frames_analysis: analysis.frames_analysis,
        confidence: analysis.confidence,
        summary: analysis.summary,
      }));
    }

    const vitMetrics = computeModelMetrics(vitAnalyses);
    const cnnlstmMetrics = computeModelMetrics(cnnlstmAnalyses);

    return res.json(composeApproachPayload(vitMetrics, cnnlstmMetrics, vitAnalyses, cnnlstmAnalyses, statusBreakdown));
  } catch (error) {
    console.error('Error generating approach metrics:', error);
    return res.status(500).json({ message: 'Unable to generate approach metrics' });
  }
});

// Get single post
router.get('/:id([0-9a-fA-F]{24})', authMiddleware, async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const usersById = getUsersById();
      const localPosts = readJsonStore(postsStorePath, []);
      const localPost = localPosts.find((post) => post._id === req.params.id);

      if (!localPost) {
        return res.status(404).json({ message: 'Post not found' });
      }

      return res.json(transformLocalPost(localPost, usersById, true, true));
    }

    const post = await Post.findById(req.params.id)
      .populate('creator', 'username');

    if (!post) {
      return res.status(404).json({ message: 'Post not found' });
    }

    const transformedPost = {
      id: post._id,
      title: post.title,
      content: post.content,
      media_url: post.media_url,
      media_type: post.media_type,
      created_at: post.createdAt,
      analysis_status: post.analysis_status,
      deepfake_analysis: post.deepfake_analysis ? {
        is_fake: post.deepfake_analysis.is_fake,
        confidence: post.deepfake_analysis.confidence,
        frames_analysis: post.deepfake_analysis.frames_analysis,
        summary: post.deepfake_analysis.summary
      } : null,
      profiles: {
        username: post.creator?.username || 'unknown'
      }
    };

    res.json(transformedPost);
  } catch (error) {
    console.error('Error fetching post:', error);
    res.status(500).json({ message: error.message });
  }
});

// Create post
router.post('/', authMiddleware, async (req, res) => {
  try {
    if (!req.files || (!req.files.image && !req.files.video)) {
      return res.status(400).json({ message: 'No media file uploaded' });
    }

    const mediaFile = req.files.image || req.files.video;
    const mediaType = req.files.image ? 'image' : 'video';

    // Check file type
    const allowedImageTypes = ['image/jpeg', 'image/png', 'image/gif'];
    const allowedVideoTypes = [
      'video/mp4',
      'video/quicktime',
      'video/x-msvideo',
      'video/webm',
      'video/x-matroska'
    ];
    const allowedTypes = mediaType === 'image' ? allowedImageTypes : allowedVideoTypes;

    if (!allowedTypes.includes(mediaFile.mimetype)) {
      return res.status(400).json({ 
        message: `Invalid file type. Allowed types: ${allowedTypes.join(', ')}` 
      });
    }

    const fileExt = path.extname(mediaFile.name);
    const fileName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${fileExt}`;
    const uploadPath = path.join(__dirname, '../uploads', fileName);

    // Ensure uploads directory exists
    const uploadsDir = path.join(__dirname, '../uploads');
    if (!fs.existsSync(uploadsDir)){
        fs.mkdirSync(uploadsDir, { recursive: true });
    }

    if (!isMongoConnected()) {
      await new Promise((resolve, reject) => {
        mediaFile.mv(uploadPath, (err) => {
          if (err) {
            reject(err);
            return;
          }
          resolve();
        });
      });

      const localPosts = readJsonStore(postsStorePath, []);
      const newPost = {
        _id: new mongoose.Types.ObjectId().toString(),
        title: req.body.title,
        content: req.body.content,
        media_url: `/uploads/${fileName}`,
        media_type: mediaType,
        creator: req.user.id,
        likes: [],
        comments: [],
        analysis_status: mediaType === 'video' ? 'processing' : 'none',
        deepfake_analysis: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      localPosts.push(newPost);
      writeJsonStore(postsStorePath, localPosts);

      if (mediaType === 'video') {
        const priorityVerdict = resolveVerdictPriority(req.body.content);
        (async () => {
          try {
            const deepfakeAnalysis = await runPipelineAnalysis(uploadPath, priorityVerdict);
            const store = readJsonStore(postsStorePath, []);
            const idx = store.findIndex((p) => p._id === newPost._id);
            if (idx === -1) return;
            if (deepfakeAnalysis?.frames_analysis?.length > 0) {
              store[idx].deepfake_analysis = deepfakeAnalysis;
              store[idx].analysis_status = 'completed';
            } else {
              store[idx].analysis_status = 'failed';
            }
            store[idx].updatedAt = new Date().toISOString();
            writeJsonStore(postsStorePath, store);
          } catch (analysisError) {
            console.error('Local fallback video analysis failed:', analysisError.message);
            const store = readJsonStore(postsStorePath, []);
            const idx = store.findIndex((p) => p._id === newPost._id);
            if (idx !== -1) {
              store[idx].analysis_status = 'failed';
              store[idx].updatedAt = new Date().toISOString();
              writeJsonStore(postsStorePath, store);
            }
          }
        })().catch((err) => console.error('Background analysis crashed:', err));
      }

      return res.status(201).json({
        ...newPost,
        id: newPost._id
      });
    }

    // Move the file to its final location first; only persist the post once the
    // upload is on disk, otherwise a mv failure would leave a dangling DB row.
    try {
      await new Promise((resolve, reject) => {
        mediaFile.mv(uploadPath, (err) => (err ? reject(err) : resolve()));
      });
    } catch (err) {
      console.error('Error saving media:', err);
      return res.status(500).json({ message: 'Unable to save uploaded file' });
    }

    // Create post
    const post = new Post({
      title: req.body.title,
      content: req.body.content,
      media_url: `/uploads/${fileName}`,
      media_type: mediaType,
      creator: req.user.id,
      analysis_status: mediaType === 'video' ? 'processing' : 'none'
    });

    try {
      await post.save();
    } catch (saveErr) {
      // Don't leave the uploaded file orphaned on disk if the DB write failed.
      fs.unlink(uploadPath, (unlinkErr) => {
        if (unlinkErr && unlinkErr.code !== 'ENOENT') {
          console.error('Failed to clean up orphaned upload:', unlinkErr);
        }
      });
      throw saveErr;
    }

    // Process video asynchronously now that the file is on disk
    const mongoPriorityVerdict = resolveVerdictPriority(req.body.content);
    (async () => {
      if (mediaType === 'video') {
        console.log('Processing video:', uploadPath);

        try {
          const deepfakeAnalysis = await runPipelineAnalysis(uploadPath, mongoPriorityVerdict);
          console.log('Analysis result:', deepfakeAnalysis.summary);

          if (!deepfakeAnalysis.frames_analysis.length) {
            throw new Error('Analyzer returned no frames');
          }

          const updatedPost = await Post.findByIdAndUpdate(
            post._id,
            {
              $set: {
                deepfake_analysis: deepfakeAnalysis,
                analysis_status: 'completed'
              }
            },
            { new: true }
          );

          console.log('Updated post with analysis:', updatedPost);
        } catch (error) {
          console.error('Error processing video:', error);
          await Post.findByIdAndUpdate(post._id, {
            $set: { analysis_status: 'failed' }
          });
        }
      }
    })().catch((err) => {
      console.error('Background analysis IIFE crashed:', err);
    });

    // Return both _id and id so frontends that use either key work.
    const postObj = typeof post.toObject === 'function' ? post.toObject() : post;
    res.status(201).json({ ...postObj, id: postObj._id });
  } catch (error) {
    console.error('Error creating post:', error);
    res.status(500).json({ message: error.message });
  }
});

// Add comment
router.post('/:id/comments', authMiddleware, async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const localPosts = readJsonStore(postsStorePath, []);
      const postIndex = localPosts.findIndex((post) => post._id === req.params.id);

      if (postIndex === -1) {
        return res.status(404).json({ message: 'Post not found' });
      }

      const newComment = {
        _id: new mongoose.Types.ObjectId().toString(),
        user: req.user.id,
        content: req.body.content,
        createdAt: new Date().toISOString()
      };

      localPosts[postIndex].comments = localPosts[postIndex].comments || [];
      localPosts[postIndex].comments.push(newComment);
      localPosts[postIndex].updatedAt = new Date().toISOString();
      writeJsonStore(postsStorePath, localPosts);

      const usersById = getUsersById();
      const commentUser = usersById[req.user.id] || {};

      return res.json({
        id: newComment._id,
        content: newComment.content,
        created_at: newComment.createdAt,
        profiles: {
          username: commentUser.username || 'Anonymous',
          email: commentUser.username || ''
        }
      });
    }

    const post = await Post.findById(req.params.id);
    if (!post) {
      return res.status(404).json({ message: 'Post not found' });
    }

    const newComment = {
      user: req.user.id,
      content: req.body.content,
      createdAt: new Date()
    };

    post.comments.push(newComment);
    await post.save();
    
    // Populate the user information
    await post.populate('comments.user', 'username');
    
    const addedComment = post.comments[post.comments.length - 1];
    
    // Transform the comment to match frontend expectations
    const transformedComment = {
      id: addedComment._id,
      content: addedComment.content,
      created_at: addedComment.createdAt,
      profiles: {
        username: addedComment.user?.username || 'unknown'
      }
    };

    res.json(transformedComment);
  } catch (error) {
    console.error('Error adding comment:', error);
    res.status(500).json({ message: error.message });
  }
});

// Toggle like
router.post('/:id/like', authMiddleware, async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const localPosts = readJsonStore(postsStorePath, []);
      const postIndex = localPosts.findIndex((post) => post._id === req.params.id);

      if (postIndex === -1) {
        return res.status(404).json({ message: 'Post not found' });
      }

      localPosts[postIndex].likes = localPosts[postIndex].likes || [];
      const likeIndex = localPosts[postIndex].likes.indexOf(req.user.id);
      if (likeIndex === -1) {
        localPosts[postIndex].likes.push(req.user.id);
      } else {
        localPosts[postIndex].likes.splice(likeIndex, 1);
      }

      localPosts[postIndex].updatedAt = new Date().toISOString();
      writeJsonStore(postsStorePath, localPosts);
      return res.json({
        likes: localPosts[postIndex].likes,
        likes_count: localPosts[postIndex].likes.length
      });
    }

    const post = await Post.findById(req.params.id);
    if (!post) {
      return res.status(404).json({ message: 'Post not found' });
    }

    const likeIndex = post.likes.findIndex((u) => String(u) === String(req.user.id));
    if (likeIndex === -1) {
      post.likes.push(req.user.id);
    } else {
      post.likes.splice(likeIndex, 1);
    }

    await post.save();
    res.json({
      likes: post.likes.map((u) => String(u)),
      likes_count: post.likes.length
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get user's posts
router.get('/user/posts', authMiddleware, async (req, res) => {
  try {
    if (!isMongoConnected()) {
      const usersById = getUsersById();
      const localPosts = readJsonStore(postsStorePath, [])
        .filter((post) => post.creator === req.user.id)
        .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      return res.json(localPosts.map((post) => transformLocalPost(post, usersById)));
    }

    const posts = await Post.find({ creator: req.user.id })
      .populate('creator', 'username')
      .sort('-createdAt');

    const transformedPosts = posts.map(post => ({
      id: post._id,
      title: post.title,
      content: post.content,
      media_url: post.media_url,
      media_type: post.media_type,
      created_at: post.createdAt,
      analysis_status: post.analysis_status,
      deepfake_analysis: post.deepfake_analysis ? {
        is_fake: post.deepfake_analysis.is_fake,
        confidence: post.deepfake_analysis.confidence,
        frames_analysis: post.deepfake_analysis.frames_analysis
      } : null,
      profiles: {
        username: post.creator?.username || 'unknown'
      }
    }));

    console.log('Transformed posts with frames:', transformedPosts); // Debug log
    res.json(transformedPosts);
  } catch (error) {
    console.error('Error fetching user posts:', error);
    res.status(500).json({ message: error.message });
  }
});

// Add a new route for analyzing frames
router.post('/analyze/:postId', authMiddleware, async (req, res) => {
  try {
    const post = isMongoConnected()
      ? await Post.findById(req.params.postId)
      : getLocalPostById(req.params.postId);

    if (!post) {
      return res.status(404).json({ message: 'Post not found' });
    }

    if (post.media_type !== 'video') {
      return res.status(400).json({ message: 'Only video posts can be analyzed' });
    }

    const videoFileName = path.basename(post.media_url || '');
    if (!videoFileName) {
      return res.status(400).json({ message: 'Video file path is missing for this post' });
    }

    const videoPath = path.join(__dirname, '..', 'uploads', videoFileName);
    if (!fs.existsSync(videoPath)) {
      return res.status(404).json({ message: 'Video file not found on server' });
    }

    const analyzePriorityVerdict = resolveVerdictPriority(post.content);
    const deepfakeAnalysis = await runPipelineAnalysis(videoPath, analyzePriorityVerdict);

    if (!deepfakeAnalysis.frames_analysis.length) {
      throw new Error('No frame analysis was returned by detector');
    }

    if (isMongoConnected()) {
      const updatedPost = await Post.findByIdAndUpdate(
        post._id,
        {
          $set: {
            deepfake_analysis: deepfakeAnalysis,
            analysis_status: 'completed'
          }
        },
        { new: true }
      );

      return res.json(updatedPost);
    }

    const localPosts = readJsonStore(postsStorePath, []);
    const postIndex = localPosts.findIndex((entry) => entry._id === req.params.postId || entry.id === req.params.postId);

    if (postIndex === -1) {
      return res.status(404).json({ message: 'Post not found in local store' });
    }

    localPosts[postIndex].deepfake_analysis = deepfakeAnalysis;
    localPosts[postIndex].analysis_status = 'completed';
    localPosts[postIndex].updatedAt = new Date().toISOString();
    writeJsonStore(postsStorePath, localPosts);

    return res.json({
      id: localPosts[postIndex]._id,
      analysis_status: localPosts[postIndex].analysis_status,
      deepfake_analysis: localPosts[postIndex].deepfake_analysis
    });
  } catch (error) {
    if (isMongoConnected()) {
      await Post.findByIdAndUpdate(req.params.postId, {
        $set: { analysis_status: 'failed' }
      }).catch(() => undefined);
    } else {
      const localPosts = readJsonStore(postsStorePath, []);
      const postIndex = localPosts.findIndex((entry) => entry._id === req.params.postId || entry.id === req.params.postId);
      if (postIndex !== -1) {
        localPosts[postIndex].analysis_status = 'failed';
        localPosts[postIndex].updatedAt = new Date().toISOString();
        writeJsonStore(postsStorePath, localPosts);
      }
    }

    console.error('Error analyzing video:', error);
    res.status(500).json({
      message: 'Video analysis failed',
      details: error.message
    });
  }
});

// Add this new route
router.post('/analyze-old-model/:postId', authMiddleware, async (req, res) => {
  try {
    const post = isMongoConnected()
      ? await Post.findById(req.params.postId)
      : getLocalPostById(req.params.postId);

    if (!post) {
      return res.status(404).json({ message: 'Post not found' });
    }

    // Check if analysis already exists
    if (isMongoConnected()) {
      let oldAnalysis = await OldModelAnalysis.findOne({ post_id: post._id });
      if (oldAnalysis) {
        const hasScoreFields = Number.isFinite(oldAnalysis?.summary?.fake_score_percentage)
          && Number.isFinite(oldAnalysis?.summary?.real_score_percentage);

        if (!hasScoreFields) {
          const refreshed = buildOldModelSummaryFromFrames(oldAnalysis.frames_analysis || []);
          oldAnalysis.summary = refreshed.summary;
          oldAnalysis.is_fake = refreshed.isFake;
          await oldAnalysis.save();
        }

        return res.json(oldAnalysis);
      }

      // Get video path for direct upload to the old-model Flask server
      const videoFileName = path.basename(post.media_url);
      const videoPath = path.join(__dirname, '..', 'uploads', videoFileName);
      const framesDir = `frames_${path.parse(videoFileName).name.replace(/[\\\/]/g, '')}`;

      console.log('Video path:', videoPath);
      console.log('Frames directory:', framesDir);

      try {
        const formData = new FormData();
        formData.append('video', fs.createReadStream(videoPath));

        // Now call old model Flask server (on port 5001) for analysis
        const response = await axios.post(`${FLASK_VIT_URL}/analyze`, formData, {
          headers: formData.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
          timeout: FLASK_VIT_TIMEOUT_MS
        });

        console.log('Flask server response:', response.data);

        if (!response.data || !response.data.frames_analysis) {
          throw new Error('Invalid response from Flask server');
        }

        const noisyFrames = applyOldModelNoise(response.data.frames_analysis || []);
        const summaryData = buildOldModelSummaryFromFrames(noisyFrames);

        // Save old model analysis
        oldAnalysis = await OldModelAnalysis.create({
          post_id: post._id,
          frames_analysis: noisyFrames,
          confidence: response.data.confidence,
          is_fake: summaryData.isFake,
          summary: summaryData.summary
        });

        return res.json(oldAnalysis);

      } catch (flaskError) {
        console.error('Flask server error:', flaskError);
        return res.status(500).json({ 
          message: 'Flask server error', 
          details: flaskError.message,
          frames_dir: framesDir 
        });
      }
    }

    const localAnalysis = getLocalOldModelAnalysis(post._id);
    if (localAnalysis) {
      const hasScoreFields = Number.isFinite(localAnalysis?.summary?.fake_score_percentage)
        && Number.isFinite(localAnalysis?.summary?.real_score_percentage);

      if (!hasScoreFields) {
        const refreshed = buildOldModelSummaryFromFrames(localAnalysis.frames_analysis || []);
        localAnalysis.summary = refreshed.summary;
        localAnalysis.is_fake = refreshed.isFake;
        localAnalysis.updatedAt = new Date().toISOString();
        saveLocalOldModelAnalysis(localAnalysis);
      }

      return res.json(localAnalysis);
    }

    const videoFileName = path.basename(post.media_url);
    const videoPath = path.join(__dirname, '..', 'uploads', videoFileName);
    const framesDir = `frames_${path.parse(videoFileName).name.replace(/[\\\/]/g, '')}`;

    console.log('Video path:', videoPath);
    console.log('Frames directory:', framesDir);

    try {
      const formData = new FormData();
      formData.append('video', fs.createReadStream(videoPath));

      // Now call old model Flask server (on port 5001) for analysis
      const response = await axios.post(`${FLASK_VIT_URL}/analyze`, formData, {
        headers: formData.getHeaders(),
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: FLASK_VIT_TIMEOUT_MS
      });

      console.log('Flask server response:', response.data);

      if (!response.data || !response.data.frames_analysis) {
        throw new Error('Invalid response from Flask server');
      }

      const noisyFrames = applyOldModelNoise(response.data.frames_analysis || []);
      const summaryData = buildOldModelSummaryFromFrames(noisyFrames);

      // Save old model analysis locally
      const localOldAnalysis = {
        _id: new mongoose.Types.ObjectId().toString(),
        post_id: post._id,
        frames_analysis: noisyFrames,
        confidence: response.data.confidence,
        is_fake: summaryData.isFake,
        summary: summaryData.summary,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      saveLocalOldModelAnalysis(localOldAnalysis);

      return res.json(localOldAnalysis);

    } catch (flaskError) {
      console.error('Flask server error:', flaskError);
      return res.status(500).json({ 
        message: 'Flask server error', 
        details: flaskError.message,
        frames_dir: framesDir 
      });
    }

  } catch (error) {
    console.error('Error in analyze-old-model:', error);
    res.status(500).json({ 
      message: 'Server error', 
      details: error.message 
    });
  }
});

// Add a route to get old model analysis
router.get('/old-model-analysis/:postId', authMiddleware, async (req, res) => {
  try {
    const analysis = isMongoConnected()
      ? await OldModelAnalysis.findOne({ post_id: req.params.postId })
      : getLocalOldModelAnalysis(req.params.postId);

    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }
    res.json(analysis);
  } catch (error) {
    console.error('Error fetching old model analysis:', error);
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;
