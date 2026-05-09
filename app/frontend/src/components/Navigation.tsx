import { useNavigate, useLocation } from 'react-router-dom';
import {
  StockOutlined,
  FundOutlined,
  ShareAltOutlined,
} from '@ant-design/icons';

const navItems = [
  { key: '/', label: 'Market Overview', icon: <StockOutlined /> },
  { key: '/graph', label: 'Graph Network', icon: <ShareAltOutlined /> },
];

export default function Navigation() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 32px',
        height: 56,
        background: '#0d1321',
        borderBottom: '1px solid #1f2937',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      {/* Logo */}
      <div
        onClick={() => navigate('/')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          cursor: 'pointer',
        }}
      >
        <FundOutlined style={{ fontSize: 22, color: '#3b82f6' }} />
        <span
          style={{
            fontSize: 16,
            fontWeight: 700,
            color: '#f9fafb',
            letterSpacing: 0.5,
          }}
        >
          BVMT AI
        </span>
        <span
          style={{
            fontSize: 10,
            fontWeight: 500,
            color: '#3b82f6',
            background: 'rgba(59,130,246,0.12)',
            padding: '2px 8px',
            borderRadius: 10,
            letterSpacing: 0.5,
          }}
        >
          PLATFORM
        </span>
      </div>

      {/* Nav Items */}
      <nav style={{ display: 'flex', gap: 4 }}>
        {navItems.map((item) => {
          const active =
            item.key === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.key);

          return (
            <button
              key={item.key}
              onClick={() => navigate(item.key)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '8px 16px',
                border: 'none',
                borderRadius: 8,
                background: active ? 'rgba(59,130,246,0.15)' : 'transparent',
                color: active ? '#60a5fa' : '#9ca3af',
                fontSize: 13,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s',
                fontFamily: 'inherit',
              }}
              onMouseEnter={(e) => {
                if (!active) e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
              }}
              onMouseLeave={(e) => {
                if (!active) e.currentTarget.style.background = 'transparent';
              }}
            >
              {item.icon}
              {item.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
