import type { NeighborInfo } from '../types';

interface NeighborCardProps {
  neighbor: NeighborInfo;
  rank?: number;
}

export default function NeighborCard({ neighbor, rank }: NeighborCardProps) {
  const corrColor = neighbor.correlation >= 0.6
    ? '#10b981'
    : neighbor.correlation >= 0.4
    ? '#3b82f6'
    : '#f59e0b';

  const typeColors: Record<string, string> = {
    leading: '#8b5cf6',
    lagging: '#f59e0b',
    concurrent: '#3b82f6',
  };

  return (
    <div
      style={{
        background: '#0d1321',
        border: '1px solid #1f2937',
        borderRadius: 10,
        padding: '14px 18px',
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        transition: 'border-color 0.2s',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#374151')}
      onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#1f2937')}
    >
      {/* Rank */}
      {rank && (
        <div style={{
          width: 28,
          height: 28,
          borderRadius: '50%',
          background: 'rgba(59,130,246,0.1)',
          color: '#60a5fa',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          fontWeight: 700,
          flexShrink: 0,
        }}>
          {rank}
        </div>
      )}

      {/* Info */}
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb' }}>
            {neighbor.ticker}
          </span>
          {neighbor.sector && (
            <span style={{ fontSize: 10, color: '#6b7280' }}>{neighbor.sector}</span>
          )}
        </div>
        <div style={{ fontSize: 12, color: '#6b7280' }}>
          {neighbor.explanation || `Relation score: ${neighbor.relation_score.toFixed(3)}`}
        </div>
      </div>

      {/* Correlation */}
      <div style={{ textAlign: 'right', flexShrink: 0 }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: corrColor }}>
          {neighbor.correlation.toFixed(2)}
        </div>
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: typeColors[neighbor.influence_type] || '#6b7280',
            textTransform: 'uppercase',
            letterSpacing: 0.5,
          }}
        >
          {neighbor.influence_type}
        </div>
      </div>
    </div>
  );
}
