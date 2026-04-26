const router = require('express').Router();

router.get('/model-evaluation', async (req, res) => {
  try {
    const epochs = Array.from({ length: 50 }, (_, i) => i + 1);

    const train_accuracy = epochs.map((epoch) => {
      const base = 0.85 + 0.08 * (1 - Math.exp(-epoch / 9));
      const noise = (Math.random() - 0.5) * 0.012;
      return Math.min(Math.max(base + noise, 0.85), 0.95);
    });

    const val_accuracy = epochs.map((epoch) => {
      const base = 0.845 + 0.075 * (1 - Math.exp(-epoch / 10.5));
      const noise = (Math.random() - 0.5) * 0.014;
      return Math.min(Math.max(base + noise, 0.85), 0.94);
    });

    const train_loss = epochs.map((epoch) => {
      const base = 0.62 * Math.exp(-epoch / 11.5) + 0.2;
      const noise = (Math.random() - 0.5) * 0.03;
      return Math.max(base + noise, 0.16);
    });

    const val_loss = epochs.map((epoch) => {
      const base = 0.65 * Math.exp(-epoch / 12.5) + 0.22;
      const noise = (Math.random() - 0.5) * 0.035;
      return Math.max(base + noise, 0.18);
    });

    const trainingHistory = {
      epochs,
      train_accuracy,
      val_accuracy,
      train_loss,
      val_loss,
    };

    const detailedMetrics = epochs.map((epoch, i) => ({
      epoch,
      train_acc: train_accuracy[i],
      val_acc: val_accuracy[i],
      train_loss: train_loss[i],
      val_loss: val_loss[i],
      precision: 0.85 + 0.07 * (1 - Math.exp(-epoch / 11)) + (Math.random() - 0.5) * 0.01,
      recall: 0.845 + 0.07 * (1 - Math.exp(-epoch / 10.5)) + (Math.random() - 0.5) * 0.01,
      f1_score: 0.847 + 0.07 * (1 - Math.exp(-epoch / 10.8)) + (Math.random() - 0.5) * 0.009,
    }));

    const modelComparison = [
      {
        name: 'Pipeline Ensemble',
        architecture: 'CNN-LSTM + ViT (weighted consensus)',
        parameters: '128.8M',
        inference_time_ms: 67,
        accuracy: 0.926,
        precision: 0.912,
        recall: 0.902,
        f1: 0.907,
        auc: 0.964,
      },
      {
        name: 'ViT',
        architecture: 'Vision Transformer (.pt)',
        parameters: '86.5M',
        inference_time_ms: 52,
        accuracy: 0.903,
        precision: 0.884,
        recall: 0.872,
        f1: 0.878,
        auc: 0.943,
      },
      {
        name: 'CNN-LSTM',
        architecture: 'CNN-LSTM (.h5)',
        parameters: '42.3M',
        inference_time_ms: 41,
        accuracy: 0.887,
        precision: 0.871,
        recall: 0.858,
        f1: 0.864,
        auc: 0.931,
      },
    ];

    const pipelineModel = modelComparison[0];

    const modelPerformance = {
      overall_accuracy: pipelineModel.accuracy,
      precision: pipelineModel.precision,
      recall: pipelineModel.recall,
      f1_score: pipelineModel.f1,
      specificity: 0.914,
      sensitivity: pipelineModel.recall,
    };

    const rocMetrics = [
      {
        model: 'Pipeline Ensemble',
        auc: 0.964,
        fpr: [0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.42, 0.68, 1.0],
        tpr: [0, 0.81, 0.87, 0.90, 0.92, 0.94, 0.955, 0.97, 0.982, 0.992, 1.0],
      },
      {
        model: 'ViT',
        auc: 0.943,
        fpr: [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.55, 0.75, 1.0],
        tpr: [0, 0.78, 0.84, 0.87, 0.89, 0.905, 0.92, 0.935, 0.955, 0.98, 1.0],
      },
      {
        model: 'CNN-LSTM',
        auc: 0.931,
        fpr: [0, 0.03, 0.07, 0.1, 0.15, 0.22, 0.3, 0.4, 0.6, 0.78, 1.0],
        tpr: [0, 0.75, 0.81, 0.84, 0.86, 0.88, 0.9, 0.92, 0.945, 0.975, 1.0],
      },
    ];

    const confusionMatrices = {
      threshold_50: {
        threshold: 0.50,
        true_positives: 1142,
        true_negatives: 1158,
        false_positives: 86,
        false_negatives: 96,
      },
      threshold_60: {
        threshold: 0.60,
        true_positives: 1108,
        true_negatives: 1181,
        false_positives: 63,
        false_negatives: 130,
      },
      threshold_70: {
        threshold: 0.70,
        true_positives: 1041,
        true_negatives: 1210,
        false_positives: 34,
        false_negatives: 197,
      },
    };

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

    const classWisePerformance = [
      {
        class: 'Real Videos',
        precision: 0.918,
        recall: 0.904,
        f1: 0.911,
        support: 1241,
      },
      {
        class: 'Fake Videos',
        precision: 0.906,
        recall: 0.899,
        f1: 0.902,
        support: 1241,
      },
    ];

    const sourcePerformance = [
      { source: 'Face-Swap', accuracy: 0.927, samples: 410 },
      { source: 'Lip-Sync', accuracy: 0.919, samples: 385 },
      { source: 'Expression-Puppet', accuracy: 0.904, samples: 365 },
      { source: 'Full-Body', accuracy: 0.892, samples: 340 },
      { source: 'Authentic', accuracy: 0.934, samples: 982 },
    ];

    const overfittingAnalysis = {
      gap_at_epoch_10: 0.021,
      gap_at_epoch_25: 0.014,
      gap_at_epoch_50: 0.011,
      final_gap: 0.011,
      overfitting_status: 'MINIMAL OVERFITTING - Ensemble generalizes well',
      gap_trend: 'STABLE - Training and validation curves converge cleanly',
    };

    res.json({
      demo: true,
      note: 'Demo metrics for dashboard visualization. Comparison now includes Pipeline Ensemble, ViT, and CNN-LSTM with pipeline as the primary production approach.',
      trainingHistory,
      detailedMetrics,
      modelPerformance,
      rocMetrics,
      confusionMatrices,
      datasetInfo,
      modelComparison,
      classWisePerformance,
      sourcePerformance,
      overfittingAnalysis,
      timestamp: new Date(),
    });
  } catch (error) {
    console.error('Error fetching evaluation metrics:', error);
    res.status(500).json({ error: 'Failed to fetch evaluation metrics' });
  }
});

module.exports = router;
