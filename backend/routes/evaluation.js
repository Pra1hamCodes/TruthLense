const router = require('express').Router();

// Realistic model evaluation metrics (85-90% accuracy range)
router.get('/model-evaluation', async (req, res) => {
  try {
    // Generate more realistic training curves with natural noise and plateaus
    const epochs = Array.from({ length: 50 }, (_, i) => i + 1);
    
    // Training accuracy - rises but with realistic noise, plateaus around 88-89%
    const train_accuracy = epochs.map((epoch) => {
      const base = 0.50 + 0.38 * (1 - Math.exp(-epoch / 10));
      const noise = (Math.random() - 0.5) * 0.015; // realistic fluctuation
      return Math.min(Math.max(base + noise, 0.5), 0.90);
    });
    
    // Validation accuracy - slightly behind training, more noisy, plateaus ~87%
    const val_accuracy = epochs.map((epoch) => {
      const base = 0.48 + 0.38 * (1 - Math.exp(-epoch / 10.8));
      const noise = (Math.random() - 0.5) * 0.018; // slightly more noise
      return Math.min(Math.max(base + noise - 0.008, 0.48), 0.88);
    });
    
    // Loss curves - gradual decrease with plateaus
    const train_loss = epochs.map((epoch) => {
      const base = 0.7 * Math.exp(-epoch / 12) + 0.25;
      const noise = (Math.random() - 0.5) * 0.04;
      return Math.max(base + noise, 0.22);
    });
    
    const val_loss = epochs.map((epoch) => {
      const base = 0.75 * Math.exp(-epoch / 13) + 0.27;
      const noise = (Math.random() - 0.5) * 0.05;
      return Math.max(base + noise, 0.24);
    });

    const trainingHistory = {
      epochs,
      train_accuracy,
      val_accuracy,
      train_loss,
      val_loss,
    };

    // Detailed epoch metrics with realistic variation
    const detailedMetrics = epochs.map((epoch, i) => ({
      epoch,
      train_acc: train_accuracy[i],
      val_acc: val_accuracy[i],
      train_loss: train_loss[i],
      val_loss: val_loss[i],
      precision: 0.82 + 0.06 * (1 - Math.exp(-epoch / 12)) + (Math.random() - 0.5) * 0.01,
      recall: 0.80 + 0.07 * (1 - Math.exp(-epoch / 11)) + (Math.random() - 0.5) * 0.012,
      f1_score: 0.81 + 0.065 * (1 - Math.exp(-epoch / 11.5)) + (Math.random() - 0.5) * 0.01,
    }));

    // Final model performance - realistic 85-90% range
    const modelPerformance = {
      overall_accuracy: 0.876,
      precision: 0.884,
      recall: 0.863,
      f1_score: 0.873,
      specificity: 0.889,
      sensitivity: 0.863,
    };

    // ROC metrics with realistic AUC values (0.92-0.96)
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
        model: 'EfficientNet-B4',
        auc: 0.915,
        fpr: [0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.33, 0.43, 0.63, 0.80, 1.0],
        tpr: [0, 0.72, 0.78, 0.81, 0.83, 0.860, 0.880, 0.905, 0.935, 0.970, 1.0],
      },
    ];

    // Confusion matrices with realistic distributions
    const confusionMatrices = {
      threshold_50: {
        threshold: 0.50,
        true_positives: 1089,
        true_negatives: 1098,
        false_positives: 120,
        false_negatives: 175,
      },
      threshold_60: {
        threshold: 0.60,
        true_positives: 1042,
        true_negatives: 1145,
        false_positives: 73,
        false_negatives: 222,
      },
      threshold_70: {
        threshold: 0.70,
        true_positives: 965,
        true_negatives: 1189,
        false_positives: 29,
        false_negatives: 299,
      },
    };

    // Dataset distribution
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

    // Model architecture comparison - realistic range
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
        name: 'EfficientNet-B4',
        parameters: '19.4M',
        inference_time_ms: 32,
        accuracy: 0.845,
        precision: 0.852,
        recall: 0.835,
        f1: 0.843,
        auc: 0.915,
      },
      {
        name: 'ResNet-101',
        parameters: '44.5M',
        inference_time_ms: 38,
        accuracy: 0.838,
        precision: 0.846,
        recall: 0.827,
        f1: 0.836,
        auc: 0.908,
      },
    ];

    // Class-wise performance
    const classWisePerformance = [
      {
        class: 'Real Videos',
        precision: 0.889,
        recall: 0.863,
        f1: 0.876,
        support: 1241,
      },
      {
        class: 'Fake Videos',
        precision: 0.879,
        recall: 0.863,
        f1: 0.871,
        support: 1241,
      },
    ];

    // Validation on different sources/deepfake types - realistic results
    const sourcePerformance = [
      { source: 'Face-Swap', accuracy: 0.892, samples: 410 },
      { source: 'Lip-Sync', accuracy: 0.871, samples: 385 },
      { source: 'Expression-Puppet', accuracy: 0.855, samples: 365 },
      { source: 'Full-Body', accuracy: 0.838, samples: 340 },
      { source: 'Authentic', accuracy: 0.889, samples: 982 },
    ];

    // Overfitting analysis - realistic gap
    const overfittingAnalysis = {
      gap_at_epoch_10: 0.028,
      gap_at_epoch_25: 0.018,
      gap_at_epoch_50: 0.012,
      final_gap: 0.012,
      overfitting_status: 'MINIMAL OVERFITTING - Model generalizes reasonably well',
      gap_trend: 'STABLE - Training and validation curves converging',
    };

    res.json({
      demo: true,
      note: 'These metrics are synthetically generated demo data for the evaluation dashboard and are NOT measured from real inference runs.',
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
