const router = require('express').Router();

/**
 * Synthetic demo metrics for the evaluation dashboard.
 *
 * The dashboard renders three "models":
 *   - cnnlstm : the TensorFlow CNN+LSTM hybrid (.h5 weights in flask_server/)
 *   - vit     : the PyTorch Vision Transformer (.pt weights in flask_server_2/)
 *   - pipeline: a late-fusion ensemble (averaged confidence) of the two above
 *
 * These numbers are NOT measured from real inference — they're deterministic-looking
 * synthetic curves designed to exercise the dashboard UI. The response is tagged
 * `demo: true` so the frontend can surface that honestly. To wire up real metrics,
 * replace the body of each buildModelMetrics(...) block with evaluation output
 * from your own test-set run and keep the response schema unchanged.
 */

function seededNoise(seed) {
  // Lightweight deterministic pseudo-noise so curves look realistic but stable.
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280 - 0.5;
  };
}

function buildTrainingCurves({ epochs, peakAcc, peakValAcc, startAcc, lossFloor, startLoss, seedBase }) {
  const epochList = Array.from({ length: epochs }, (_, i) => i + 1);
  const trainNoise = seededNoise(seedBase);
  const valNoise = seededNoise(seedBase + 101);
  const trainLossNoise = seededNoise(seedBase + 203);
  const valLossNoise = seededNoise(seedBase + 307);

  const train_accuracy = epochList.map((epoch) => {
    const base = startAcc + (peakAcc - startAcc) * (1 - Math.exp(-epoch / 10));
    return Math.min(Math.max(base + trainNoise() * 0.014, startAcc), peakAcc + 0.01);
  });
  const val_accuracy = epochList.map((epoch) => {
    const base = startAcc - 0.02 + (peakValAcc - (startAcc - 0.02)) * (1 - Math.exp(-epoch / 10.8));
    return Math.min(Math.max(base + valNoise() * 0.017 - 0.006, startAcc - 0.02), peakValAcc + 0.01);
  });
  const train_loss = epochList.map((epoch) => {
    const base = (startLoss - lossFloor) * Math.exp(-epoch / 12) + lossFloor;
    return Math.max(base + trainLossNoise() * 0.04, lossFloor - 0.02);
  });
  const val_loss = epochList.map((epoch) => {
    const base = (startLoss + 0.05 - lossFloor) * Math.exp(-epoch / 13) + lossFloor + 0.02;
    return Math.max(base + valLossNoise() * 0.05, lossFloor);
  });

  return { epochs: epochList, train_accuracy, val_accuracy, train_loss, val_loss };
}

function buildDetailedMetrics(trainingHistory, { seedBase, precisionPeak, recallPeak, f1Peak }) {
  const precNoise = seededNoise(seedBase + 11);
  const recNoise = seededNoise(seedBase + 13);
  const f1Noise = seededNoise(seedBase + 17);
  return trainingHistory.epochs.map((epoch, i) => ({
    epoch,
    train_acc: trainingHistory.train_accuracy[i],
    val_acc: trainingHistory.val_accuracy[i],
    train_loss: trainingHistory.train_loss[i],
    val_loss: trainingHistory.val_loss[i],
    precision: Math.min(precisionPeak, 0.80 + (precisionPeak - 0.80) * (1 - Math.exp(-epoch / 12)) + precNoise() * 0.01),
    recall: Math.min(recallPeak, 0.78 + (recallPeak - 0.78) * (1 - Math.exp(-epoch / 11)) + recNoise() * 0.012),
    f1_score: Math.min(f1Peak, 0.79 + (f1Peak - 0.79) * (1 - Math.exp(-epoch / 11.5)) + f1Noise() * 0.01),
  }));
}

function buildModelMetrics({
  id,
  displayName,
  description,
  architectureNotes,
  trainingConfig,
  performance,
  classWise,
  confusionMatrices,
  sourcePerformance,
  overfittingAnalysis,
  inferenceTimeMs,
  parameterCount,
}) {
  const trainingHistory = buildTrainingCurves(trainingConfig);
  const detailedMetrics = buildDetailedMetrics(trainingHistory, {
    seedBase: trainingConfig.seedBase + 500,
    precisionPeak: performance.precision,
    recallPeak: performance.recall,
    f1Peak: performance.f1_score,
  });

  return {
    id,
    displayName,
    description,
    architectureNotes,
    parameterCount,
    inferenceTimeMs,
    trainingHistory,
    detailedMetrics,
    modelPerformance: performance,
    classWisePerformance: classWise,
    confusionMatrices,
    sourcePerformance,
    overfittingAnalysis,
  };
}

// Shared ROC curves across the three variants so they can be plotted together.
const rocMetrics = [
  {
    model: 'ViT (Vision Transformer)',
    auc: 0.942,
    fpr: [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.55, 0.75, 1.0],
    tpr: [0, 0.78, 0.84, 0.87, 0.89, 0.905, 0.920, 0.935, 0.955, 0.980, 1.0],
  },
  {
    model: 'CNN-LSTM Hybrid',
    auc: 0.928,
    fpr: [0, 0.03, 0.07, 0.10, 0.15, 0.22, 0.30, 0.40, 0.60, 0.78, 1.0],
    tpr: [0, 0.75, 0.81, 0.84, 0.86, 0.880, 0.900, 0.920, 0.945, 0.975, 1.0],
  },
  {
    model: 'Pipeline (ViT + CNN-LSTM ensemble)',
    auc: 0.958,
    fpr: [0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.50, 0.72, 1.0],
    tpr: [0, 0.82, 0.88, 0.91, 0.93, 0.944, 0.956, 0.970, 0.982, 0.992, 1.0],
  },
];

const modelComparison = [
  {
    name: 'ViT (Vision Transformer)',
    parameters: '86.5M',
    inference_time_ms: 52,
    accuracy: 0.876,
    precision: 0.884,
    recall: 0.863,
    f1: 0.873,
    auc: 0.942,
  },
  {
    name: 'CNN-LSTM Hybrid',
    parameters: '42.3M',
    inference_time_ms: 41,
    accuracy: 0.859,
    precision: 0.867,
    recall: 0.847,
    f1: 0.857,
    auc: 0.928,
  },
  {
    name: 'Pipeline (Ensemble)',
    parameters: '128.8M',
    inference_time_ms: 93,
    accuracy: 0.901,
    precision: 0.912,
    recall: 0.889,
    f1: 0.900,
    auc: 0.958,
  },
];

const datasetInfo = {
  total_samples: 2482,
  training_samples: 1489,
  validation_samples: 497,
  test_samples: 496,
  real_videos: 1241,
  fake_videos: 1241,
  total_frames_analyzed: 26847,
  avg_frames_per_video: 10.79,
};

function buildVitMetrics() {
  return buildModelMetrics({
    id: 'vit',
    displayName: 'ViT (Vision Transformer)',
    description:
      'PyTorch Vision Transformer fine-tuned for deepfake detection. Splits each 224x224 frame into 32px patches, processes them through 6 transformer blocks with 16-head attention, and produces a single sigmoid confidence. This is the active production path (flask_server_2/).',
    architectureNotes: [
      'Input: 224x224 RGB, ImageNet-normalized',
      'Patch size: 32 | Embedding dim: 1024 | Depth: 6 | Heads: 16',
      'Output head: 1 logit -> sigmoid',
      'Weights: as_model_0.837.pt (~214 MB, Git LFS)',
    ],
    trainingConfig: { epochs: 50, peakAcc: 0.902, peakValAcc: 0.880, startAcc: 0.50, lossFloor: 0.25, startLoss: 0.70, seedBase: 101 },
    performance: {
      overall_accuracy: 0.876,
      precision: 0.884,
      recall: 0.863,
      f1_score: 0.873,
      specificity: 0.889,
      sensitivity: 0.863,
    },
    parameterCount: '86.5M',
    inferenceTimeMs: 52,
    classWise: [
      { class: 'Real Videos', precision: 0.889, recall: 0.863, f1: 0.876, support: 1241 },
      { class: 'Fake Videos', precision: 0.879, recall: 0.863, f1: 0.871, support: 1241 },
    ],
    confusionMatrices: {
      threshold_50: { threshold: 0.50, true_positives: 1089, true_negatives: 1098, false_positives: 120, false_negatives: 175 },
      threshold_60: { threshold: 0.60, true_positives: 1042, true_negatives: 1145, false_positives: 73, false_negatives: 222 },
      threshold_70: { threshold: 0.70, true_positives: 965, true_negatives: 1189, false_positives: 29, false_negatives: 299 },
    },
    sourcePerformance: [
      { source: 'Face-Swap', accuracy: 0.892, samples: 410 },
      { source: 'Lip-Sync', accuracy: 0.871, samples: 385 },
      { source: 'Expression-Puppet', accuracy: 0.855, samples: 365 },
      { source: 'Full-Body', accuracy: 0.838, samples: 340 },
      { source: 'Authentic', accuracy: 0.889, samples: 982 },
    ],
    overfittingAnalysis: {
      gap_at_epoch_10: 0.028,
      gap_at_epoch_25: 0.018,
      gap_at_epoch_50: 0.012,
      final_gap: 0.012,
      overfitting_status: 'MINIMAL OVERFITTING - Strong spatial feature learning, model generalizes well',
      gap_trend: 'STABLE - Curves converge cleanly, no late-epoch divergence',
    },
  });
}

function buildCnnLstmMetrics() {
  return buildModelMetrics({
    id: 'cnnlstm',
    displayName: 'CNN-LSTM Hybrid',
    description:
      'TensorFlow CNN-LSTM hybrid. A CNN backbone extracts per-frame features, which are fed through an LSTM to capture temporal inconsistencies (frame-to-frame flicker, motion artefacts) that often give away face-swap deepfakes. Bundled here as a comparison baseline and fallback (flask_server/).',
    architectureNotes: [
      'Input: 224x224 RGB, scaled to [0, 1]',
      'CNN feature extractor -> LSTM (temporal aggregation) -> dense sigmoid',
      'Output: single sigmoid confidence',
      'Weights: deepfake_detection_model.h5 (~95 MB, committed directly)',
    ],
    trainingConfig: { epochs: 50, peakAcc: 0.888, peakValAcc: 0.864, startAcc: 0.48, lossFloor: 0.28, startLoss: 0.75, seedBase: 211 },
    performance: {
      overall_accuracy: 0.859,
      precision: 0.867,
      recall: 0.847,
      f1_score: 0.857,
      specificity: 0.872,
      sensitivity: 0.847,
    },
    parameterCount: '42.3M',
    inferenceTimeMs: 41,
    classWise: [
      { class: 'Real Videos', precision: 0.872, recall: 0.847, f1: 0.859, support: 1241 },
      { class: 'Fake Videos', precision: 0.861, recall: 0.847, f1: 0.854, support: 1241 },
    ],
    confusionMatrices: {
      threshold_50: { threshold: 0.50, true_positives: 1059, true_negatives: 1072, false_positives: 148, false_negatives: 203 },
      threshold_60: { threshold: 0.60, true_positives: 1005, true_negatives: 1122, false_positives: 98, false_negatives: 257 },
      threshold_70: { threshold: 0.70, true_positives: 921, true_negatives: 1167, false_positives: 53, false_negatives: 341 },
    },
    sourcePerformance: [
      { source: 'Face-Swap', accuracy: 0.861, samples: 410 },
      { source: 'Lip-Sync', accuracy: 0.878, samples: 385 },
      { source: 'Expression-Puppet', accuracy: 0.839, samples: 365 },
      { source: 'Full-Body', accuracy: 0.821, samples: 340 },
      { source: 'Authentic', accuracy: 0.872, samples: 982 },
    ],
    overfittingAnalysis: {
      gap_at_epoch_10: 0.034,
      gap_at_epoch_25: 0.022,
      gap_at_epoch_50: 0.017,
      final_gap: 0.017,
      overfitting_status: 'MILD OVERFITTING - LSTM component memorises some short-horizon patterns',
      gap_trend: 'STABLE - Gap shrinks steadily after epoch 20',
    },
  });
}

function buildPipelineMetrics() {
  return buildModelMetrics({
    id: 'pipeline',
    displayName: 'Pipeline (ViT + CNN-LSTM ensemble)',
    description:
      'Late-fusion ensemble that averages confidences from ViT and CNN-LSTM (weighted 0.6 / 0.4 in favour of ViT). ViT captures per-frame spatial artefacts; CNN-LSTM captures temporal inconsistency. Combining them reduces false negatives on lip-sync and full-body manipulations where a single model tends to be weaker.',
    architectureNotes: [
      'Pipeline: frames -> ViT (confidence_v) & CNN-LSTM (confidence_c) in parallel',
      'Ensemble rule: p_final = 0.6 * confidence_v + 0.4 * confidence_c',
      'Per-video decision: majority-vote across sampled frames with ties -> fake (safer over-flag)',
      'Total parameters: 128.8M | End-to-end latency: ~93ms per frame on CPU',
    ],
    trainingConfig: { epochs: 50, peakAcc: 0.923, peakValAcc: 0.905, startAcc: 0.54, lossFloor: 0.22, startLoss: 0.66, seedBase: 307 },
    performance: {
      overall_accuracy: 0.901,
      precision: 0.912,
      recall: 0.889,
      f1_score: 0.900,
      specificity: 0.913,
      sensitivity: 0.889,
    },
    parameterCount: '128.8M',
    inferenceTimeMs: 93,
    classWise: [
      { class: 'Real Videos', precision: 0.916, recall: 0.889, f1: 0.902, support: 1241 },
      { class: 'Fake Videos', precision: 0.908, recall: 0.889, f1: 0.898, support: 1241 },
    ],
    confusionMatrices: {
      threshold_50: { threshold: 0.50, true_positives: 1121, true_negatives: 1132, false_positives: 86, false_negatives: 143 },
      threshold_60: { threshold: 0.60, true_positives: 1082, true_negatives: 1178, false_positives: 47, false_negatives: 175 },
      threshold_70: { threshold: 0.70, true_positives: 1013, true_negatives: 1214, false_positives: 16, false_negatives: 239 },
    },
    sourcePerformance: [
      { source: 'Face-Swap', accuracy: 0.921, samples: 410 },
      { source: 'Lip-Sync', accuracy: 0.905, samples: 385 },
      { source: 'Expression-Puppet', accuracy: 0.882, samples: 365 },
      { source: 'Full-Body', accuracy: 0.872, samples: 340 },
      { source: 'Authentic', accuracy: 0.913, samples: 982 },
    ],
    overfittingAnalysis: {
      gap_at_epoch_10: 0.022,
      gap_at_epoch_25: 0.014,
      gap_at_epoch_50: 0.009,
      final_gap: 0.009,
      overfitting_status: 'MINIMAL OVERFITTING - Ensembling averages out individual models\u2019 failure modes',
      gap_trend: 'CONVERGING - Ensemble curves track each other closely throughout training',
    },
  });
}

router.get('/model-evaluation', (req, res) => {
  try {
    const models = {
      cnnlstm: buildCnnLstmMetrics(),
      vit: buildVitMetrics(),
      pipeline: buildPipelineMetrics(),
    };

    res.json({
      demo: true,
      note:
        'These metrics are synthetically generated demo data for the evaluation dashboard and are NOT measured from real inference runs. Swap the route body to emit real test-set metrics without changing the schema.',
      defaultModel: 'pipeline',
      modelOrder: ['cnnlstm', 'vit', 'pipeline'],
      models,
      rocMetrics,
      modelComparison,
      datasetInfo,
      timestamp: new Date(),
    });
  } catch (error) {
    console.error('Error building evaluation metrics:', error);
    res.status(500).json({ error: 'Failed to build evaluation metrics' });
  }
});

module.exports = router;
