import ReactECharts from 'echarts-for-react';
import type { PricePoint, PriceHorizonPoint } from '../types';

interface PriceHorizonChartProps {
  history: PricePoint[];
  predicted: PricePoint[];
  horizon?: PriceHorizonPoint[];
  direction?: string;
  height?: number;
}

export default function PriceHorizonChart({
  history,
  predicted,
  direction,
  height = 400,
}: PriceHorizonChartProps) {
  const upColor = '#10b981';
  const downColor = '#ef4444';
  const lineColor = direction === 'DOWN' ? downColor : upColor;

  const allDates = history.map((p) => p.date);
  const realPrices = history.map((p) => p.close);
  const n = allDates.length;

  // Always place predicted values on the last 7 timeline points.
  const predValues = predicted.slice(0, 7).map((p) => p.close);
  const predStart = Math.max(0, n - predValues.length);

  const predPrices = new Array(n).fill(null) as (number | null)[];
  predValues.forEach((value, i) => {
    const idx = predStart + i;
    if (idx >= 0 && idx < n) {
      predPrices[idx] = value;
    }
  });

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1f2937',
      borderColor: '#374151',
      textStyle: { color: '#f9fafb', fontSize: 12 },
      formatter: (params: any[]) => {
        const date = params[0]?.axisValue || '';
        let html = `<div style="font-weight:600;margin-bottom:4px">${date}</div>`;
        for (const p of params) {
          if (p.value != null) {
            let label = '📊 Actual';
            if (p.seriesName === 'Predicted (V2)') {
              label = '🔮 Predicted (V2)';
            }
            html += `<div>${label}: <b>${Number(p.value).toFixed(3)} DT</b></div>`;
          }
        }
        return html;
      },
    },
    toolbox: {
      show: true,
      right: 10,
      top: 0,
      iconStyle: { borderColor: '#9ca3af' },
      emphasis: { iconStyle: { borderColor: '#f9fafb' } },
      feature: {
        restore: { title: 'Reset Zoom' },
      },
    },
    grid: { top: 30, right: 20, bottom: 74, left: 60 },
    dataZoom: [
      {
        type: 'inside' as const,
        xAxisIndex: 0,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: true,
      },
      {
        type: 'slider' as const,
        xAxisIndex: 0,
        height: 18,
        bottom: 8,
        start: 70,
        end: 100,
        borderColor: '#1f2937',
        backgroundColor: '#0b1220',
        fillerColor: 'rgba(96,165,250,0.2)',
        textStyle: { color: '#9ca3af' },
      },
    ],
    xAxis: {
      type: 'category' as const,
      data: allDates,
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#6b7280', fontSize: 10, rotate: 45 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value' as const,
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#6b7280', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1f2937', type: 'dashed' as const } },
      scale: true,
    },
    series: [
      {
        name: 'Actual',
        type: 'line',
        data: realPrices,
        smooth: true,
        lineStyle: { width: 2, color: '#60a5fa' },
        itemStyle: { color: '#60a5fa' },
        showSymbol: false,
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(96,165,250,0.15)' },
              { offset: 1, color: 'rgba(96,165,250,0)' },
            ],
          },
        },
      },
      {
        name: 'Predicted (V2)',
        type: 'line',
        data: predPrices,
        smooth: true,
        lineStyle: { width: 2.5, color: lineColor, type: 'dashed' as const },
        itemStyle: { color: lineColor },
        showSymbol: true,
        symbolSize: 6,
        connectNulls: true,
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: `${lineColor}20` },
              { offset: 1, color: `${lineColor}00` },
            ],
          },
        },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height }} notMerge />;
}
