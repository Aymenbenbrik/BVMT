interface ArticleCardProps {
  title: string | null;
  date: string | null;
  sentimentScore: number;
  source?: string;
}

export default function ArticleCard({ title, date, sentimentScore, source = 'ilboursa.com' }: ArticleCardProps) {
  const sentColor =
    sentimentScore > 0.2 ? '#10b981' :
    sentimentScore < -0.2 ? '#ef4444' :
    '#f59e0b';

  const sentLabel =
    sentimentScore > 0.2 ? 'Positive' :
    sentimentScore < -0.2 ? 'Negative' :
    'Neutral';

  return (
    <div
      style={{
        background: '#0d1321',
        border: '1px solid #1f2937',
        borderRadius: 10,
        padding: '14px 18px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: 12,
        transition: 'border-color 0.2s',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#374151')}
      onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#1f2937')}
    >
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: '#f9fafb', marginBottom: 6, lineHeight: 1.4 }}>
          {title ? (title.length > 80 ? title.slice(0, 80) + '…' : title) : 'Untitled article'}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 11, color: '#6b7280' }}>
          {date && <span>{date}</span>}
          <span>📰 {source}</span>
        </div>
      </div>
      <div
        style={{
          background: `${sentColor}15`,
          color: sentColor,
          fontSize: 11,
          fontWeight: 600,
          padding: '4px 10px',
          borderRadius: 12,
          border: `1px solid ${sentColor}40`,
          whiteSpace: 'nowrap',
        }}
      >
        {sentLabel} ({sentimentScore >= 0 ? '+' : ''}{sentimentScore.toFixed(2)})
      </div>
    </div>
  );
}
