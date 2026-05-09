interface ConfidenceBarProps {
  value: number; // 0-1
  direction?: string | null;
  height?: number;
}

export default function ConfidenceBar({ value, direction, height = 6 }: ConfidenceBarProps) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const dir = (direction || '').toUpperCase();

  const color =
    dir === 'UP' ? '#10b981' :
    dir === 'DOWN' ? '#ef4444' :
    '#f59e0b';

  return (
    <div
      style={{
        width: '100%',
        height,
        background: '#1f2937',
        borderRadius: height / 2,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: '100%',
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          borderRadius: height / 2,
          transition: 'width 0.6s ease',
        }}
      />
    </div>
  );
}
