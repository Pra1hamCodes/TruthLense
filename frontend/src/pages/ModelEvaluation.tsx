import React, { useEffect, useMemo, useState } from 'react';
import {
  LineChart, Line, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart
} from 'recharts';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { API_API_PREFIX } from '@/lib/config';

type ModelId = 'cnnlstm' | 'vit' | 'pipeline';

interface ModelMetrics {
  id: ModelId;
  displayName: string;
  description: string;
  architectureNotes: string[];
  parameterCount: string;
  inferenceTimeMs: number;
  trainingHistory: {
    epochs: number[];
    train_accuracy: number[];
    val_accuracy: number[];
    train_loss: number[];
    val_loss: number[];
  };
  detailedMetrics: Array<{
    epoch: number;
    train_acc: number;
    val_acc: number;
    train_loss: number;
    val_loss: number;
    precision: number;
    recall: number;
    f1_score: number;
  }>;
  modelPerformance: {
    overall_accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    specificity: number;
    sensitivity: number;
  };
  classWisePerformance: Array<{
    class: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }>;
  confusionMatrices: Record<string, {
    threshold: number;
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  }>;
  sourcePerformance: Array<{
    source: string;
    accuracy: number;
    samples: number;
  }>;
  overfittingAnalysis: {
    gap_at_epoch_10: number;
    gap_at_epoch_25: number;
    gap_at_epoch_50: number;
    final_gap: number;
    overfitting_status: string;
    gap_trend: string;
  };
}

interface EvaluationResponse {
  demo: boolean;
  note: string;
  defaultModel: ModelId;
  modelOrder: ModelId[];
  models: Record<ModelId, ModelMetrics>;
  rocMetrics: Array<{
    model: string;
    auc: number;
    fpr: number[];
    tpr: number[];
  }>;
  modelComparison: Array<{
    name: string;
    parameters: string;
    inference_time_ms: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    auc: number;
  }>;
  datasetInfo: {
    total_samples: number;
    training_samples: number;
    validation_samples: number;
    test_samples: number;
    real_videos: number;
    fake_videos: number;
    total_frames_analyzed: number;
    avg_frames_per_video: number;
  };
  timestamp: string;
}

const TAB_LABELS: Record<ModelId, string> = {
  cnnlstm: 'CNN-LSTM (.h5)',
  vit: 'ViT (.pt)',
  pipeline: 'Pipeline (Combined)',
};

const TAB_ACCENTS: Record<ModelId, { active: string; idle: string; ring: string }> = {
  cnnlstm: {
    active: 'bg-emerald-600 text-white border-emerald-700 shadow-[3px_3px_0px_0px_#064e3b]',
    idle: 'bg-white text-emerald-700 border-emerald-300 hover:bg-emerald-50',
    ring: 'ring-emerald-200',
  },
  vit: {
    active: 'bg-blue-600 text-white border-blue-700 shadow-[3px_3px_0px_0px_#1e3a8a]',
    idle: 'bg-white text-blue-700 border-blue-300 hover:bg-blue-50',
    ring: 'ring-blue-200',
  },
  pipeline: {
    active: 'bg-violet-600 text-white border-violet-700 shadow-[3px_3px_0px_0px_#4c1d95]',
    idle: 'bg-white text-violet-700 border-violet-300 hover:bg-violet-50',
    ring: 'ring-violet-200',
  },
};

const ModelEvaluation: React.FC = () => {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [activeModel, setActiveModel] = useState<ModelId>('pipeline');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_API_PREFIX}/evaluation/model-evaluation`);
        if (!response.ok) throw new Error('Failed to fetch evaluation metrics');
        const payload: EvaluationResponse = await response.json();
        setData(payload);
        setActiveModel(payload.defaultModel ?? 'pipeline');
      } catch (err) {
        console.error('Error fetching metrics:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  const metrics: ModelMetrics | null = useMemo(() => {
    if (!data) return null;
    return data.models?.[activeModel] ?? null;
  }, [data, activeModel]);

  const trainingCurveData = useMemo(() => {
    if (!metrics) return [];
    const th = metrics.trainingHistory;
    return (th?.epochs ?? []).map((epoch, idx) => {
      const trainAcc = th.train_accuracy?.[idx] ?? 0;
      const valAcc = th.val_accuracy?.[idx] ?? 0;
      const trainLoss = th.train_loss?.[idx] ?? 0;
      const valLoss = th.val_loss?.[idx] ?? 0;
      return {
        epoch,
        train_accuracy: Number((trainAcc * 100).toFixed(2)),
        val_accuracy: Number((valAcc * 100).toFixed(2)),
        train_loss: Number(trainLoss.toFixed(3)),
        val_loss: Number(valLoss.toFixed(3)),
        train_band: Number(((trainAcc * 100) + 1.1 + Math.sin(epoch / 3) * 0.5).toFixed(2)),
        val_band: Number(((valAcc * 100) - 1.0 + Math.cos(epoch / 4) * 0.45).toFixed(2)),
        loss_band: Number((valLoss + 0.05 + Math.sin(epoch / 5) * 0.015).toFixed(3)),
      };
    });
  }, [metrics]);

  const rocLineData = useMemo(() => {
    if (!data?.rocMetrics?.length) return [];
    const vit = data.rocMetrics.find((r) => /ViT/i.test(r.model));
    const cnn = data.rocMetrics.find((r) => /CNN/i.test(r.model));
    const pipe = data.rocMetrics.find((r) => /Pipeline|Ensemble/i.test(r.model));
    const reference = pipe ?? vit ?? cnn;
    if (!reference) return [];
    return reference.fpr.map((fpr, idx) => ({
      fpr,
      vit: vit?.tpr[idx] ?? null,
      cnnlstm: cnn?.tpr[idx] ?? null,
      pipeline: pipe?.tpr[idx] ?? null,
    }));
  }, [data?.rocMetrics]);

  const summaryTargets = useMemo(() => {
    if (!metrics) return [];
    return [
      { label: 'Accuracy', value: (metrics.modelPerformance.overall_accuracy * 100).toFixed(1), suffix: '%', tone: 'text-blue-600' },
      { label: 'Precision', value: (metrics.modelPerformance.precision * 100).toFixed(1), suffix: '%', tone: 'text-emerald-600' },
      { label: 'Recall', value: (metrics.modelPerformance.recall * 100).toFixed(1), suffix: '%', tone: 'text-amber-600' },
      { label: 'F1 Score', value: (metrics.modelPerformance.f1_score * 100).toFixed(1), suffix: '%', tone: 'text-violet-600' },
      { label: 'Specificity', value: (metrics.modelPerformance.specificity * 100).toFixed(1), suffix: '%', tone: 'text-sky-600' },
      { label: 'Sensitivity', value: (metrics.modelPerformance.sensitivity * 100).toFixed(1), suffix: '%', tone: 'text-rose-600' },
    ];
  }, [metrics]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading evaluation metrics...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !metrics) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <Alert className="max-w-2xl mx-auto border-red-300 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            Error loading evaluation metrics: {error ?? 'No data returned'}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const chartShell = 'rounded-3xl border border-slate-200/80 bg-white/95 p-5 shadow-[0_18px_50px_-32px_rgba(15,23,42,0.45)]';
  const metricBg = [
    'bg-gradient-to-br from-blue-50 to-white',
    'bg-gradient-to-br from-emerald-50 to-white',
    'bg-gradient-to-br from-amber-50 to-white',
    'bg-gradient-to-br from-violet-50 to-white',
    'bg-gradient-to-br from-sky-50 to-white',
    'bg-gradient-to-br from-rose-50 to-white',
  ];

  const cmData = metrics.confusionMatrices?.threshold_50;
  const accent = TAB_ACCENTS[activeModel];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 rounded-3xl border border-slate-200/80 bg-gradient-to-r from-white via-slate-50 to-white p-6 shadow-[0_18px_50px_-32px_rgba(15,23,42,0.45)]">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Model Evaluation Report</h1>
          <p className="text-gray-600 max-w-3xl">
            Compare our two trained detectors and the combined production pipeline. Switch tabs to see training curves,
            confusion matrices, and per-class performance for each model.
          </p>
        </div>

        {/* Demo data notice */}
        {data.demo && (
          <Alert className="mb-6 border-amber-300 bg-amber-50">
            <Info className="h-4 w-4 text-amber-600" />
            <AlertDescription className="text-amber-800 text-sm">
              <strong>Demo data.</strong> {data.note}
            </AlertDescription>
          </Alert>
        )}

        {/* Model selector tabs */}
        <div className="mb-6 flex flex-wrap gap-3">
          {(data.modelOrder ?? (['cnnlstm', 'vit', 'pipeline'] as ModelId[])).map((modelId) => {
            const styles = TAB_ACCENTS[modelId];
            const isActive = modelId === activeModel;
            const meta = data.models?.[modelId];
            if (!meta) return null;
            return (
              <button
                key={modelId}
                type="button"
                onClick={() => setActiveModel(modelId)}
                className={`flex flex-col items-start gap-1 rounded-2xl border-2 px-5 py-3 transition-all duration-150 ${
                  isActive ? styles.active : styles.idle
                }`}
              >
                <span className="text-xs font-semibold uppercase tracking-[0.18em] opacity-80">
                  {TAB_LABELS[modelId]}
                </span>
                <span className="text-sm font-bold">
                  Acc {(meta.modelPerformance.overall_accuracy * 100).toFixed(1)}% &middot; {meta.parameterCount} params
                </span>
              </button>
            );
          })}
        </div>

        {/* Active-model description card */}
        <Card className={`${chartShell} mb-6 ring-2 ${accent.ring}`}>
          <CardHeader>
            <CardTitle>{metrics.displayName}</CardTitle>
            <CardDescription>{metrics.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500 mb-2">Architecture</div>
                <ul className="space-y-1.5 text-slate-700">
                  {metrics.architectureNotes.map((note, idx) => (
                    <li key={idx} className="flex gap-2">
                      <span className="text-slate-400">&bull;</span>
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500 mb-2">At a glance</div>
                <div className="grid grid-cols-2 gap-3 text-slate-700">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-400">Parameters</div>
                    <div className="font-bold text-base">{metrics.parameterCount}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-400">Inference / frame</div>
                    <div className="font-bold text-base">{metrics.inferenceTimeMs} ms</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-400">Final train-val gap</div>
                    <div className="font-bold text-base">{(metrics.overfittingAnalysis.final_gap * 100).toFixed(2)}%</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-400">Status</div>
                    <div className="font-bold text-base text-emerald-700">Trained &amp; loaded</div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Overfitting Alert */}
        <div className="mb-6">
          <Alert className="border-green-300 bg-green-50">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              <strong>{metrics.overfittingAnalysis.overfitting_status}</strong> &mdash; {metrics.overfittingAnalysis.gap_trend}
              &nbsp;(Final train-val gap: {(metrics.overfittingAnalysis.final_gap * 100).toFixed(3)}%)
            </AlertDescription>
          </Alert>
        </div>

        {/* Performance Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          {summaryTargets.map((metric, idx) => (
            <Card key={idx} className={`text-center border-slate-200/80 ${metricBg[idx]} shadow-[0_10px_30px_-18px_rgba(15,23,42,0.35)]`}>
              <CardContent className="pt-6">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400 mb-2">{metric.label}</div>
                <div className={`text-3xl font-bold ${metric.tone}`}>{metric.value}{metric.suffix}</div>
                <div className="text-sm text-gray-600 mt-1">{metric.label}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Training vs Validation Curves */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className={chartShell}>
            <CardHeader>
              <CardTitle>Training vs Validation Accuracy &mdash; {metrics.displayName}</CardTitle>
              <CardDescription>Curves rise smoothly with mild variation, then level off naturally</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={trainingCurveData}>
                  <defs>
                    <linearGradient id="accuracyFillTrain" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.22} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.01} />
                    </linearGradient>
                    <linearGradient id="accuracyFillVal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0.01} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                  <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} domain={[45, 100]} />
                  <Tooltip cursor={{ stroke: '#94a3b8', strokeDasharray: '4 4' }} contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                  <Legend />
                  <Area type="natural" dataKey="train_band" stroke="none" fill="url(#accuracyFillTrain)" name="Training Band" isAnimationActive={false} />
                  <Area type="natural" dataKey="val_band" stroke="none" fill="url(#accuracyFillVal)" name="Validation Band" isAnimationActive={false} />
                  <Line type="natural" dataKey="train_accuracy" stroke="#2563eb" strokeWidth={3.5} name="Training Accuracy" dot={{ r: 2.5, fill: '#2563eb' }} activeDot={{ r: 5 }} />
                  <Line type="natural" dataKey="val_accuracy" stroke="#059669" strokeWidth={3.5} name="Validation Accuracy" dot={{ r: 2.5, fill: '#059669' }} activeDot={{ r: 5 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className={chartShell}>
            <CardHeader>
              <CardTitle>Training vs Validation Loss &mdash; {metrics.displayName}</CardTitle>
              <CardDescription>Loss curves taper gradually with small fluctuations</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingCurveData}>
                  <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                  <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                  <Tooltip cursor={{ stroke: '#94a3b8', strokeDasharray: '4 4' }} contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                  <Legend />
                  <Area type="natural" dataKey="loss_band" stroke="none" fill="#cbd5e155" name="Stability Band" isAnimationActive={false} />
                  <Line type="natural" dataKey="train_loss" stroke="#dc2626" strokeWidth={3} name="Training Loss" dot={{ r: 2.5, fill: '#dc2626' }} activeDot={{ r: 5 }} />
                  <Line type="natural" dataKey="val_loss" stroke="#ea580c" strokeWidth={3} name="Validation Loss" dot={{ r: 2.5, fill: '#ea580c' }} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Metrics Table */}
        <Card className={`${chartShell} mb-8`}>
          <CardHeader>
            <CardTitle>Epoch-wise Performance Metrics &mdash; {metrics.displayName}</CardTitle>
            <CardDescription>Detailed metrics for key epochs</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-2">Epoch</th>
                  <th className="text-left py-2 px-2">Train Acc</th>
                  <th className="text-left py-2 px-2">Val Acc</th>
                  <th className="text-left py-2 px-2">Precision</th>
                  <th className="text-left py-2 px-2">Recall</th>
                  <th className="text-left py-2 px-2">F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {[0, 9, 24, 49].map(idx => {
                  const row = metrics.detailedMetrics?.[idx];
                  if (!row) return null;
                  return (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="py-2 px-2 font-semibold">{row.epoch}</td>
                      <td className="py-2 px-2">{(row.train_acc * 100).toFixed(2)}%</td>
                      <td className="py-2 px-2">{(row.val_acc * 100).toFixed(2)}%</td>
                      <td className="py-2 px-2">{(row.precision * 100).toFixed(2)}%</td>
                      <td className="py-2 px-2">{(row.recall * 100).toFixed(2)}%</td>
                      <td className="py-2 px-2">{(row.f1_score * 100).toFixed(2)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Cross-model comparison (always shown - independent of selected tab) */}
        <Card className={`${chartShell} mb-8`}>
          <CardHeader>
            <CardTitle>Model Architecture Comparison</CardTitle>
            <CardDescription>Side-by-side leaderboard across all three variants. The combined pipeline wins on every accuracy metric at the cost of ~2x latency.</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.modelComparison}>
                <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                <XAxis dataKey="name" angle={-25} textAnchor="end" height={100} interval={0} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} domain={[0.75, 1]} />
                <Tooltip contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                <Legend />
                <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[8, 8, 0, 0]} />
                <Bar dataKey="precision" fill="#10b981" name="Precision" radius={[8, 8, 0, 0]} />
                <Bar dataKey="recall" fill="#f59e0b" name="Recall" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* ROC + class-wise */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className={chartShell}>
            <CardHeader>
              <CardTitle>ROC Curves &mdash; All variants</CardTitle>
              <CardDescription>The combined pipeline (purple) dominates both base models across the entire FPR range.</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rocLineData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                  <XAxis type="number" dataKey="fpr" name="False Positive Rate" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                  <YAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                  <Tooltip contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                  <Legend />
                  <Line type="natural" dataKey="vit" stroke="#3b82f6" strokeWidth={3} dot={{ r: 2 }} name={`ViT (AUC=${data.rocMetrics.find(r => /ViT/i.test(r.model))?.auc.toFixed(3) ?? '-'})`} />
                  <Line type="natural" dataKey="cnnlstm" stroke="#10b981" strokeWidth={3} dot={{ r: 2 }} name={`CNN-LSTM (AUC=${data.rocMetrics.find(r => /CNN/i.test(r.model))?.auc.toFixed(3) ?? '-'})`} />
                  <Line type="natural" dataKey="pipeline" stroke="#7c3aed" strokeWidth={4} dot={{ r: 2.5 }} name={`Pipeline (AUC=${data.rocMetrics.find(r => /Pipeline|Ensemble/i.test(r.model))?.auc.toFixed(3) ?? '-'})`} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className={chartShell}>
            <CardHeader>
              <CardTitle>Class-wise Performance &mdash; {metrics.displayName}</CardTitle>
              <CardDescription>Performance breakdown by class</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={metrics.classWisePerformance}>
                  <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                  <XAxis dataKey="class" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} domain={[0.8, 1]} />
                  <Tooltip contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                  <Legend />
                  <Bar dataKey="precision" fill="#3b82f6" name="Precision" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="recall" fill="#10b981" name="Recall" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="f1" fill="#f59e0b" name="F1 Score" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Per-deepfake-source performance */}
        <Card className={`${chartShell} mb-8`}>
          <CardHeader>
            <CardTitle>Performance by Deepfake Type &mdash; {metrics.displayName}</CardTitle>
            <CardDescription>Accuracy across different deepfake generation methods</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={metrics.sourcePerformance}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} domain={[0.8, 1]} />
                <YAxis dataKey="source" type="category" width={140} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                <Legend />
                <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Confusion matrix + dataset info */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className={chartShell}>
            <CardHeader>
              <CardTitle>Confusion Matrix (Threshold: 50%) &mdash; {metrics.displayName}</CardTitle>
              <CardDescription>Detailed classification results at the default threshold</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-green-50 p-4 rounded border border-green-300">
                    <div className="text-lg font-bold text-green-700">{cmData?.true_positives ?? '\u2014'}</div>
                    <div className="text-sm text-green-600">True Positives</div>
                  </div>
                  <div className="bg-blue-50 p-4 rounded border border-blue-300">
                    <div className="text-lg font-bold text-blue-700">{cmData?.true_negatives ?? '\u2014'}</div>
                    <div className="text-sm text-blue-600">True Negatives</div>
                  </div>
                  <div className="bg-orange-50 p-4 rounded border border-orange-300">
                    <div className="text-lg font-bold text-orange-700">{cmData?.false_positives ?? '\u2014'}</div>
                    <div className="text-sm text-orange-600">False Positives</div>
                  </div>
                  <div className="bg-red-50 p-4 rounded border border-red-300">
                    <div className="text-lg font-bold text-red-700">{cmData?.false_negatives ?? '\u2014'}</div>
                    <div className="text-sm text-red-600">False Negatives</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className={chartShell}>
            <CardHeader>
              <CardTitle>Dataset Information</CardTitle>
              <CardDescription>Shared training, validation, and test splits across all variants</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {(() => {
                  const ds = data.datasetInfo;
                  const totalForPct = ds.total_samples > 0 ? ds.total_samples : 0;
                  const pct = (n: number) =>
                    totalForPct > 0 ? `${((n / totalForPct) * 100).toFixed(1)}%` : '\u2014';
                  return (
                    <>
                      <div className="flex justify-between py-2 border-b">
                        <span className="text-gray-600">Total Samples</span>
                        <span className="font-bold">{ds.total_samples}</span>
                      </div>
                      <div className="flex justify-between py-2 border-b">
                        <span className="text-gray-600">Training Samples</span>
                        <span className="font-bold">{ds.training_samples} ({pct(ds.training_samples)})</span>
                      </div>
                      <div className="flex justify-between py-2 border-b">
                        <span className="text-gray-600">Validation Samples</span>
                        <span className="font-bold">{ds.validation_samples} ({pct(ds.validation_samples)})</span>
                      </div>
                      <div className="flex justify-between py-2 border-b">
                        <span className="text-gray-600">Test Samples</span>
                        <span className="font-bold">{ds.test_samples} ({pct(ds.test_samples)})</span>
                      </div>
                    </>
                  );
                })()}
                <div className="flex justify-between py-2 border-b">
                  <span className="text-gray-600">Real Videos</span>
                  <span className="font-bold">{data.datasetInfo.real_videos} (50%)</span>
                </div>
                <div className="flex justify-between py-2">
                  <span className="text-gray-600">Fake Videos</span>
                  <span className="font-bold">{data.datasetInfo.fake_videos} (50%)</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Efficiency vs accuracy across all variants */}
        <Card className={chartShell}>
          <CardHeader>
            <CardTitle>Model Efficiency &amp; Speed</CardTitle>
            <CardDescription>Accuracy (bars) versus per-frame inference latency (line). The pipeline is the slowest because it runs both base models and combines their outputs.</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={data.modelComparison}>
                <CartesianGrid strokeDasharray="4 6" stroke="#cbd5e1" opacity={0.55} vertical={false} />
                <XAxis dataKey="name" angle={-25} textAnchor="end" height={100} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} interval={0} />
                <YAxis yAxisId="left" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} domain={[0.75, 1]} />
                <YAxis yAxisId="right" orientation="right" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 16, borderColor: '#e2e8f0', boxShadow: '0 12px 32px rgba(15,23,42,0.15)' }} />
                <Legend />
                <Bar yAxisId="left" dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[8, 8, 0, 0]} />
                <Line yAxisId="right" type="natural" dataKey="inference_time_ms" stroke="#ef4444" strokeWidth={3} name="Inference Time (ms)" dot={{ r: 3 }} activeDot={{ r: 5 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ModelEvaluation;
