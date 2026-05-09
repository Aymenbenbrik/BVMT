import type { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon?: ReactNode;
  color?: string;
  suffix?: string;
  subtitle?: string;
}

export default function MetricCard({ title, value, icon, color = '#3b82f6', suffix, subtitle }: MetricCardProps) {
  return (
    <div className="metric-card fade-in">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        {icon && (
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              background: `${color}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color,
              fontSize: 16,
            }}
          >
            {icon}
          </div>
        )}
        <span className="metric-label">{title}</span>
      </div>
      <div className="metric-value" style={{ color }}>
        {value}
        {suffix && <span style={{ fontSize: 14, fontWeight: 400, color: '#9ca3af', marginLeft: 4 }}>{suffix}</span>}
      </div>
      {subtitle && (
        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>{subtitle}</div>
      )}
    </div>
  );
}
