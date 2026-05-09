import ReactECharts from 'echarts-for-react';

interface SentimentGaugeProps {
  score: number; // -1 to +1
  label: string;
  confidence: number;
  height?: number;
}

export default function SentimentGauge({ score, label, confidence, height = 280 }: SentimentGaugeProps) {
  const normalizedScore = ((score + 1) / 2) * 100; // map -1..+1 to 0..100

  const option = {
    backgroundColor: 'transparent',
    series: [
      {
        type: 'gauge',
        startAngle: 200,
        endAngle: -20,
        min: 0,
        max: 100,
        center: ['50%', '60%'],
        radius: '80%',
        progress: {
          show: true,
          width: 16,
          itemStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 1, y2: 0,
              colorStops: [
                { offset: 0, color: '#ef4444' },
                { offset: 0.5, color: '#f59e0b' },
                { offset: 1, color: '#10b981' },
              ],
            },
          },
        },
        axisLine: {
          lineStyle: { width: 16, color: [[1, '#1f2937']] },
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: {
          show: true,
          distance: 24,
          fontSize: 10,
          color: '#6b7280',
          formatter: (value: number) => {
            if (value === 0) return '-1.0';
            if (value === 50) return '0.0';
            if (value === 100) return '+1.0';
            return '';
          },
        },
        pointer: {
          length: '55%',
          width: 5,
          itemStyle: { color: '#f9fafb' },
        },
        anchor: {
          show: true,
          size: 14,
          itemStyle: { color: '#3b82f6', borderWidth: 3, borderColor: '#0a0e1a' },
        },
        title: {
          show: true,
          offsetCenter: [0, '30%'],
          fontSize: 14,
          fontWeight: 600,
          color: label === 'POSITIVE' ? '#10b981' : label === 'NEGATIVE' ? '#ef4444' : '#f59e0b',
        },
        detail: {
          valueAnimation: true,
          offsetCenter: [0, '55%'],
          fontSize: 22,
          fontWeight: 700,
          color: '#f9fafb',
          formatter: () => `${score >= 0 ? '+' : ''}${score.toFixed(3)}`,
        },
        data: [{ value: normalizedScore, name: label }],
      },
    ],
  };

  return (
    <div>
      <ReactECharts option={option} style={{ height }} notMerge />
      <div style={{ textAlign: 'center', marginTop: -20 }}>
        <span style={{ fontSize: 12, color: '#6b7280' }}>
          Confidence: {(confidence * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
