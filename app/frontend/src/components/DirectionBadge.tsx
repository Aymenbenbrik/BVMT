import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined } from '@ant-design/icons';

interface DirectionBadgeProps {
  direction: string | null | undefined;
  size?: 'small' | 'default' | 'large';
  showLabel?: boolean;
}

export default function DirectionBadge({ direction, size = 'default', showLabel = true }: DirectionBadgeProps) {
  const dir = (direction || '').toUpperCase();

  const config = {
    UP: { icon: <ArrowUpOutlined />, label: 'UP', className: 'up' },
    DOWN: { icon: <ArrowDownOutlined />, label: 'DOWN', className: 'down' },
    NEUTRAL: { icon: <MinusOutlined />, label: 'HOLD', className: 'neutral' },
    HOLD: { icon: <MinusOutlined />, label: 'HOLD', className: 'hold' },
  }[dir] || { icon: <MinusOutlined />, label: dir || '—', className: 'unknown' };

  const sizeStyles = {
    small: { fontSize: 11, padding: '2px 8px' },
    default: { fontSize: 12, padding: '4px 12px' },
    large: { fontSize: 16, padding: '8px 20px', fontWeight: 700 as const },
  };

  return (
    <span className={`direction-badge ${config.className}`} style={sizeStyles[size]}>
      {config.icon}
      {showLabel && <span>{config.label}</span>}
    </span>
  );
}
