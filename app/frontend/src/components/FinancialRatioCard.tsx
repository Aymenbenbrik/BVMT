interface FinancialRatioCardProps {
  ratios: Record<string, any>;
  companyType?: string | null;
}

export default function FinancialRatioCard({ ratios }: FinancialRatioCardProps) {
  const entries = Object.entries(ratios);
  if (!entries.length) {
    return <div style={{ color: '#6b7280', textAlign: 'center', padding: 40 }}>No ratio data available</div>;
  }

  // Group by category
  const profitability = ['ROE (%)', 'ROA (%)', 'Net margin (%)', 'NIB margin (%)'];
  const efficiency = ['Cost-income (%)', 'Cost/assets (%)', 'Asset growth (%)', 'Revenue growth (%)'];
  const risk = ['Equity ratio (%)', 'Debt ratio (%)', 'Loan/deposit (%)', 'Log total assets'];

  const groups = [
    { label: 'Profitability', keys: profitability, color: '#10b981' },
    { label: 'Efficiency', keys: efficiency, color: '#3b82f6' },
    { label: 'Risk & Size', keys: risk, color: '#f59e0b' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {groups.map((group) => {
        const groupEntries = entries.filter(([k]) => group.keys.includes(k));
        if (!groupEntries.length) return null;

        return (
          <div key={group.label}>
            <div style={{
              fontSize: 13,
              fontWeight: 600,
              color: group.color,
              marginBottom: 12,
              textTransform: 'uppercase',
              letterSpacing: 0.5,
            }}>
              {group.label}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 12 }}>
              {groupEntries.map(([name, value]) => (
                <div
                  key={name}
                  style={{
                    background: '#0d1321',
                    border: '1px solid #1f2937',
                    borderRadius: 8,
                    padding: '12px 16px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ color: '#9ca3af', fontSize: 13 }}>{name}</span>
                  <span style={{
                    fontSize: 15,
                    fontWeight: 600,
                    color: typeof value === 'number'
                      ? value > 0 ? '#10b981' : value < 0 ? '#ef4444' : '#f9fafb'
                      : '#f9fafb',
                  }}>
                    {typeof value === 'number' ? value.toFixed(2) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
