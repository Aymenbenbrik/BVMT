/**
 * Ant Design dark financial theme configuration.
 * Applied via ConfigProvider in App.tsx.
 */
import type { ThemeConfig } from 'antd';

export const colors = {
  bgPrimary: '#0a0e1a',
  bgCard: '#111827',
  bgCardHover: '#1a2235',
  border: '#1f2937',
  borderLight: '#374151',
  accent: '#3b82f6',
  accentLight: '#60a5fa',
  up: '#10b981',
  upBg: 'rgba(16, 185, 129, 0.1)',
  down: '#ef4444',
  downBg: 'rgba(239, 68, 68, 0.1)',
  hold: '#f59e0b',
  holdBg: 'rgba(245, 158, 11, 0.1)',
  textPrimary: '#f9fafb',
  textSecondary: '#9ca3af',
  textMuted: '#6b7280',
};

export const darkTheme: ThemeConfig = {
  token: {
    colorPrimary: colors.accent,
    colorBgBase: colors.bgPrimary,
    colorBgContainer: colors.bgCard,
    colorBgElevated: colors.bgCard,
    colorBgLayout: colors.bgPrimary,
    colorBorder: colors.border,
    colorBorderSecondary: colors.border,
    colorText: colors.textPrimary,
    colorTextSecondary: colors.textSecondary,
    colorTextTertiary: colors.textMuted,
    colorSuccess: colors.up,
    colorError: colors.down,
    colorWarning: colors.hold,
    colorInfo: colors.accent,
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    borderRadius: 8,
    fontSize: 14,
    controlHeight: 36,
  },
  components: {
    Layout: {
      headerBg: '#0d1321',
      bodyBg: colors.bgPrimary,
      siderBg: colors.bgCard,
    },
    Table: {
      headerBg: '#0d1321',
      rowHoverBg: colors.bgCardHover,
      borderColor: colors.border,
    },
    Card: {
      colorBgContainer: colors.bgCard,
      colorBorderSecondary: colors.border,
    },
    Menu: {
      darkItemBg: 'transparent',
      darkItemSelectedBg: 'rgba(59, 130, 246, 0.15)',
    },
    Tabs: {
      colorBgContainer: 'transparent',
    },
  },
  algorithm: undefined, // We handle dark mode manually via tokens
};
