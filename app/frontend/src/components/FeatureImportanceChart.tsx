import ReactECharts from 'echarts-for-react';
import type { TopFeature } from '../types';

interface FeatureImportanceChartProps {
  features: TopFeature[];
  height?: number;
}

export default function FeatureImportanceChart({ features, height = 300 }: FeatureImportanceChartProps) {
  if (!features.length) {
    return <div style={{ color: '#6b7280', textAlign: 'center', padding: 40 }}>No feature data available</div>;
  }

  const sorted = [...features].sort((a, b) => a.z_score - b.z_score);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1f2937',
      borderColor: '#374151',
      textStyle: { color: '#f9fafb', fontSize: 12 },
      formatter: (params: any[]) => {
        const p = params[0];
        const feat = sorted[p.dataIndex];
        return `
          <div style="font-weight:600">${feat.feature_name}</div>
          <div>Z-score: <b>${feat.z_score.toFixed(2)}</b></div>
          <div>Value: ${feat.current_value.toFixed(4)}</div>
          ${feat.interpretation ? `<div style="color:#9ca3af">${feat.interpretation}</div>` : ''}
        `;
      },
    },
    grid: { top: 10, right: 30, bottom: 10, left: 140 },
    xAxis: {
      type: 'value' as const,
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#6b7280', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1f2937', type: 'dashed' as const } },
      name: 'Z-Score',
      nameTextStyle: { color: '#6b7280', fontSize: 10 },
    },
    yAxis: {
      type: 'category' as const,
      data: sorted.map((f) => f.feature_name),
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af', fontSize: 11 },
    },
    series: [
      {
        type: 'bar',
        data: sorted.map((f) => ({
          value: f.z_score,
          itemStyle: {
            color: f.z_score > 1.5
              ? '#10b981'
              : f.z_score > 0.5
              ? '#3b82f6'
              : '#f59e0b',
            borderRadius: [0, 4, 4, 0],
          },
        })),
        barWidth: 20,
      },
    ],
  };

  return <ReactECharts option={option} style={{ height }} notMerge />;
}
