import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Table, Input, Button, Select, message, Row, Col, Tooltip } from 'antd';
import {
  SearchOutlined,
  ReloadOutlined,
  RocketOutlined,
  RiseOutlined,
  FallOutlined,
  DashboardOutlined,
  BarChartOutlined,
  CheckCircleFilled,
  ClockCircleOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import type { ColumnsType } from 'antd/es/table';
import type { StockListItem } from '../types';
import { fetchMarketOverview, triggerPrecompute, refreshTicker } from '../api/client';
import MetricCard from '../components/MetricCard';
import DirectionBadge from '../components/DirectionBadge';
import ConfidenceBar from '../components/ConfidenceBar';

export default function MarketOverview() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [dirFilter, setDirFilter] = useState<string | null>(null);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['market-overview'],
    queryFn: fetchMarketOverview,
    refetchInterval: 5 * 60 * 1000, // 5 min
  });

  const handlePrecompute = async () => {
    try {
      await triggerPrecompute();
      message.success('Background precompute started! Poll cache status for progress.');
    } catch {
      message.error('Failed to start precompute');
    }
  };

  const handleRefresh = async (ticker: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      message.loading({ content: `Analyzing ${ticker}…`, key: ticker, duration: 0 });
      await refreshTicker(ticker);
      message.success({ content: `${ticker} analysis complete`, key: ticker });
      refetch();
    } catch {
      message.error({ content: `Failed to analyze ${ticker}`, key: ticker });
    }
  };

  const stocks = data?.stocks || [];
  const filtered = stocks.filter((s) => {
    if (search && !s.ticker.toLowerCase().includes(search.toLowerCase())) return false;
    if (typeFilter && s.company_type !== typeFilter) return false;
    if (dirFilter) {
      if (dirFilter === 'UNCACHED' && s.cached) return false;
      else if (dirFilter !== 'UNCACHED' && s.direction !== dirFilter) return false;
    }
    return true;
  });

  // ── Market Distribution Chart ─────────────────────────────────
  const distOption = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item' as const, backgroundColor: '#1f2937', borderColor: '#374151', textStyle: { color: '#f9fafb' } },
    series: [
      {
        type: 'pie',
        radius: ['45%', '70%'],
        center: ['50%', '50%'],
        avoidLabelOverlap: true,
        itemStyle: { borderRadius: 6, borderColor: '#0a0e1a', borderWidth: 2 },
        label: { show: true, color: '#9ca3af', fontSize: 12, formatter: '{b}: {c}' },
        data: [
          { value: data?.bull_count || 0, name: 'Bullish', itemStyle: { color: '#10b981' } },
          { value: data?.bear_count || 0, name: 'Bearish', itemStyle: { color: '#ef4444' } },
          { value: data?.hold_count || 0, name: 'Hold', itemStyle: { color: '#f59e0b' } },
          { value: data?.uncached_count || 0, name: 'Pending', itemStyle: { color: '#374151' } },
        ],
      },
    ],
  };

  // ── Bar chart ─────────────────────────────────────────────────
  const barOption = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' as const, backgroundColor: '#1f2937', borderColor: '#374151', textStyle: { color: '#f9fafb' } },
    grid: { top: 20, bottom: 30, left: 10, right: 10 },
    xAxis: { type: 'category' as const, data: ['Market Distribution'], show: false },
    yAxis: { type: 'value' as const, show: false },
    series: [
      { name: 'UP', type: 'bar', stack: 'total', data: [data?.bull_count || 0], itemStyle: { color: '#10b981', borderRadius: [4, 0, 0, 4] }, barWidth: 40 },
      { name: 'HOLD', type: 'bar', stack: 'total', data: [data?.hold_count || 0], itemStyle: { color: '#f59e0b' }, barWidth: 40 },
      { name: 'DOWN', type: 'bar', stack: 'total', data: [data?.bear_count || 0], itemStyle: { color: '#ef4444', borderRadius: [0, 4, 4, 0] }, barWidth: 40 },
    ],
  };

  // ── Table Columns ─────────────────────────────────────────────
  const columns: ColumnsType<StockListItem> = [
    {
      title: 'Ticker',
      dataIndex: 'ticker',
      key: 'ticker',
      sorter: (a, b) => a.ticker.localeCompare(b.ticker),
      render: (ticker: string) => (
        <span style={{ fontWeight: 600, color: '#f9fafb', cursor: 'pointer' }}>{ticker}</span>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'company_type',
      key: 'company_type',
      width: 110,
      render: (type: string) => (
        <span className={`company-badge ${type || 'non_bank'}`}>
          {type === 'bank' ? '🏦 Bank' : type === 'insurance' ? '🛡️ Insurance' : '🏢 Corp'}
        </span>
      ),
    },
    {
      title: 'Direction',
      dataIndex: 'direction',
      key: 'direction',
      width: 110,
      sorter: (a, b) => (a.direction || '').localeCompare(b.direction || ''),
      render: (_: any, record: StockListItem) =>
        record.cached ? <DirectionBadge direction={record.direction} /> : <span style={{ color: '#4b5563' }}>—</span>,
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 140,
      sorter: (a, b) => (a.confidence || 0) - (b.confidence || 0),
      render: (_: any, record: StockListItem) =>
        record.cached && record.confidence != null ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ConfidenceBar value={record.confidence} direction={record.direction} />
            <span style={{ fontSize: 11, color: '#9ca3af', minWidth: 32 }}>
              {(record.confidence * 100).toFixed(0)}%
            </span>
          </div>
        ) : <span style={{ color: '#4b5563' }}>—</span>,
    },
    {
      title: 'Score',
      dataIndex: 'final_score',
      key: 'final_score',
      width: 80,
      sorter: (a, b) => (a.final_score || 0) - (b.final_score || 0),
      render: (score: number | null) => {
        if (score == null) return <span style={{ color: '#4b5563' }}>—</span>;
        const color = score >= 65 ? '#10b981' : score <= 35 ? '#ef4444' : '#f59e0b';
        return <span style={{ fontWeight: 700, color }}>{score.toFixed(0)}</span>;
      },
    },
    {
      title: 'Expected Move',
      dataIndex: 'expected_move_pct',
      key: 'expected_move_pct',
      width: 130,
      sorter: (a, b) => (a.expected_move_pct || 0) - (b.expected_move_pct || 0),
      render: (move: number | null) => {
        if (move == null) return <span style={{ color: '#4b5563' }}>—</span>;
        const color = move > 0 ? '#10b981' : move < 0 ? '#ef4444' : '#9ca3af';
        return (
          <span style={{ fontWeight: 600, color }}>
            {move > 0 ? '+' : ''}{move.toFixed(2)}%
          </span>
        );
      },
    },
    {
      title: 'Last Price',
      dataIndex: 'current_close',
      key: 'current_close',
      width: 110,
      sorter: (a, b) => (a.current_close || 0) - (b.current_close || 0),
      render: (price: number | null) => {
        if (price == null) return <span style={{ color: '#4b5563' }}>—</span>;
        return <span style={{ color: '#d1d5db' }}>{price.toFixed(3)} DT</span>;
      },
    },
    {
      title: 'Status',
      key: 'status',
      width: 90,
      render: (_: any, record: StockListItem) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {record.cached ? (
            <Tooltip title="Cached"><CheckCircleFilled style={{ color: '#10b981', fontSize: 14 }} /></Tooltip>
          ) : (
            <Tooltip title="Not analyzed"><ClockCircleOutlined style={{ color: '#4b5563', fontSize: 14 }} /></Tooltip>
          )}
          <Tooltip title="Refresh analysis">
            <ReloadOutlined
              style={{ color: '#6b7280', fontSize: 12, cursor: 'pointer' }}
              onClick={(e) => handleRefresh(record.ticker, e)}
            />
          </Tooltip>
        </div>
      ),
    },
  ];

  return (
    <div className="page-container fade-in">
      {/* Title Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div className="page-title">
          <BarChartOutlined style={{ color: '#3b82f6' }} />
          Market Overview
          <span className="subtitle">BVMT — Tunisian Stock Exchange</span>
        </div>
        <Button
          type="primary"
          icon={<RocketOutlined />}
          onClick={handlePrecompute}
          style={{ borderRadius: 8 }}
        >
          Analyze All
        </Button>
      </div>

      {/* Metric Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <MetricCard
            title="Stocks Analyzed"
            value={`${(data?.total || 0) - (data?.uncached_count || 0)}`}
            suffix={`/ ${data?.total || 0}`}
            icon={<DashboardOutlined />}
            color="#3b82f6"
          />
        </Col>
        <Col xs={12} sm={6}>
          <MetricCard
            title="Bullish"
            value={data?.bull_count || 0}
            suffix={data?.total ? `${(((data?.bull_count || 0) / Math.max(data.total - (data?.uncached_count || 0), 1)) * 100).toFixed(0)}%` : ''}
            icon={<RiseOutlined />}
            color="#10b981"
          />
        </Col>
        <Col xs={12} sm={6}>
          <MetricCard
            title="Bearish"
            value={data?.bear_count || 0}
            suffix={data?.total ? `${(((data?.bear_count || 0) / Math.max(data.total - (data?.uncached_count || 0), 1)) * 100).toFixed(0)}%` : ''}
            icon={<FallOutlined />}
            color="#ef4444"
          />
        </Col>
        <Col xs={12} sm={6}>
          <MetricCard
            title="Avg Confidence"
            value={data?.average_confidence ? `${(data.average_confidence * 100).toFixed(0)}%` : '—'}
            subtitle={data?.market_mood ? `Mood: ${data.market_mood}` : ''}
            icon={<DashboardOutlined />}
            color="#f59e0b"
          />
        </Col>
      </Row>

      {/* Charts */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#9ca3af', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Signal Distribution
            </div>
            <ReactECharts option={distOption} style={{ height: 220 }} notMerge />
          </div>
        </Col>
        <Col xs={24} md={16}>
          <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#9ca3af', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Market Balance
            </div>
            <ReactECharts option={barOption} style={{ height: 220 }} notMerge />
          </div>
        </Col>
      </Row>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <Input
          placeholder="Search ticker…"
          prefix={<SearchOutlined style={{ color: '#6b7280' }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ width: 240, background: '#111827', borderColor: '#1f2937' }}
          allowClear
        />
        <Select
          placeholder="Company type"
          value={typeFilter}
          onChange={setTypeFilter}
          allowClear
          style={{ width: 160 }}
          options={[
            { value: 'bank', label: '🏦 Bank' },
            { value: 'insurance', label: '🛡️ Insurance' },
            { value: 'non_bank', label: '🏢 Non-Bank' },
          ]}
        />
        <Select
          placeholder="Direction"
          value={dirFilter}
          onChange={setDirFilter}
          allowClear
          style={{ width: 140 }}
          options={[
            { value: 'UP', label: '🟢 UP' },
            { value: 'DOWN', label: '🔴 DOWN' },
            { value: 'NEUTRAL', label: '🟡 HOLD' },
            { value: 'UNCACHED', label: '⏳ Pending' },
          ]}
        />
        <Button icon={<ReloadOutlined />} onClick={() => refetch()} style={{ borderColor: '#374151' }}>
          Refresh
        </Button>
      </div>

      {/* Table */}
      <Table
        dataSource={filtered}
        columns={columns}
        rowKey="ticker"
        loading={isLoading}
        pagination={{ pageSize: 20, showSizeChanger: true, pageSizeOptions: ['20', '50', '100'] }}
        onRow={(record) => ({
          onClick: () => navigate(`/stock/${encodeURIComponent(record.ticker)}`),
          style: {
            cursor: 'pointer',
            opacity: record.cached ? 1 : 0.5,
          },
        })}
        size="middle"
        style={{ background: '#111827', borderRadius: 12, overflow: 'hidden', border: '1px solid #1f2937' }}
      />
    </div>
  );
}
