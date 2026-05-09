import { useState, useMemo, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Slider, Select, Input, Drawer, Button } from 'antd';
import { SearchOutlined, SlidersOutlined } from '@ant-design/icons';
import { fetchGraphData } from '../api/client';
import StockGraph3D from '../components/StockGraph3D';

export default function GraphNetwork() {
  const navigate = useNavigate();
  const [threshold, setThreshold] = useState(0.4);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [focusNode, setFocusNode] = useState<string | undefined>(undefined);

  const { data, isLoading } = useQuery({
    queryKey: ['graph-data', threshold],
    queryFn: () => fetchGraphData(threshold),
    staleTime: 5 * 60 * 1000,
  });

  const handleNodeClick = (ticker: string) => {
    navigate(`/stock/${encodeURIComponent(ticker)}`);
  };

  // Focus node effect based on search
  useEffect(() => {
    if (!search || !data?.nodes) {
      setFocusNode(undefined);
      return;
    }
    const match = data.nodes.find(n => n.id.toLowerCase().includes(search.toLowerCase()));
    setFocusNode(match?.id);
  }, [search, data?.nodes]);

  // Filter nodes based on type only (edges will be automatically handled by the graph component)
  const filteredNodes = useMemo(() => {
    if (!data?.nodes) return [];
    return data.nodes.filter(n => {
      if (typeFilter && n.company_type !== typeFilter) return false;
      return true;
    });
  }, [data?.nodes, typeFilter]);

  const stats = data?.graph_metadata;

  return (
    <div style={{ position: 'relative', height: 'calc(100vh - 56px)', overflow: 'hidden' }}>
      {/* Absolute Header overlay */}
      <div style={{
        position: 'absolute',
        top: 24,
        left: 24,
        zIndex: 10,
        background: 'rgba(17, 24, 39, 0.75)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(31, 41, 55, 0.6)',
        borderRadius: 12,
        padding: '16px 20px',
        width: 320,
      }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: '#f9fafb', marginBottom: 4 }}>
          Market Graph Network
        </div>
        <div style={{ fontSize: 12, color: '#9ca3af', marginBottom: 16 }}>
          3D Interactive view of BVMT relationships
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Select
            showSearch
            placeholder="Search or select ticker…"
            value={search || undefined}
            onChange={(val) => setSearch(val || '')}
            onClear={() => setSearch('')}
            allowClear
            style={{ width: '100%' }}
            filterOption={(input, option) =>
              (option?.label ?? '').toString().toLowerCase().includes(input.toLowerCase())
            }
            options={
              data?.nodes
                ? data.nodes
                    .map((n) => ({ value: n.id, label: n.id }))
                    .sort((a, b) => a.label.localeCompare(b.label))
                : []
            }
          />
          <Select
            placeholder="Company type"
            value={typeFilter}
            onChange={setTypeFilter}
            allowClear
            style={{ width: '100%' }}
            options={[
              { value: 'bank', label: '🏦 Bank' },
              { value: 'insurance', label: '🛡️ Insurance' },
              { value: 'non_bank', label: '🏢 Non-Bank' },
            ]}
          />
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#9ca3af', marginBottom: 4 }}>
              <span>Correlation Threshold</span>
              <span>{threshold.toFixed(2)}</span>
            </div>
            <Slider
              min={0.1}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={setThreshold}
              tooltip={{ formatter: (val) => `${val}` }}
            />
          </div>
        </div>

        <Button
          type="text"
          icon={<SlidersOutlined />}
          style={{ width: '100%', marginTop: 12, color: '#60a5fa', background: 'rgba(59, 130, 246, 0.1)' }}
          onClick={() => setDrawerVisible(true)}
        >
          View Legend & Stats
        </Button>
      </div>

      {/* 3D Graph */}
      <div style={{ width: '100%', height: '100%' }}>
        {isLoading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#6b7280' }}>
            Loading network data…
          </div>
        ) : (
          <StockGraph3D
            nodes={filteredNodes}
            edges={data?.edges || []}
            height={window.innerHeight - 56}
            onNodeClick={handleNodeClick}
            focusTicker={focusNode}
            threshold={threshold}
          />
        )}
      </div>

      {/* Right Drawer */}
      <Drawer
        title="Network Insights"
        placement="right"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        mask={false}
        styles={{
          header: { background: '#0d1321', borderBottom: '1px solid #1f2937', color: '#f9fafb' },
          body: { background: '#0a0e1a', padding: 24 },
        }}
        width={320}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Stats */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#f9fafb', marginBottom: 12, textTransform: 'uppercase' }}>
              Graph Statistics
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div style={{ background: '#111827', border: '1px solid #1f2937', padding: 12, borderRadius: 8 }}>
                <div style={{ fontSize: 11, color: '#6b7280' }}>Nodes</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#3b82f6' }}>{stats?.total_nodes || 0}</div>
              </div>
              <div style={{ background: '#111827', border: '1px solid #1f2937', padding: 12, borderRadius: 8 }}>
                <div style={{ fontSize: 11, color: '#6b7280' }}>Edges</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#10b981' }}>{stats?.total_edges || 0}</div>
              </div>
            </div>
          </div>

          {/* Legend */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#f9fafb', marginBottom: 12, textTransform: 'uppercase' }}>
              Legend
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, background: '#111827', border: '1px solid #1f2937', padding: 16, borderRadius: 8 }}>
              <div>
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 6 }}>Node Size</div>
                <div style={{ fontSize: 12, color: '#d1d5db' }}>Proportional to composite confidence score.</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 6 }}>Node Color</div>
                <div style={{ display: 'flex', gap: 8, fontSize: 12 }}>
                  <span style={{ color: '#10b981' }}>● UP</span>
                  <span style={{ color: '#ef4444' }}>● DOWN</span>
                  <span style={{ color: '#f59e0b' }}>● HOLD</span>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 6 }}>Edge Thickness</div>
                <div style={{ fontSize: 12, color: '#d1d5db' }}>Proportional to correlation strength (threshold: {threshold}).</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 6 }}>Interaction</div>
                <div style={{ fontSize: 12, color: '#d1d5db' }}>
                  • Drag to rotate/pan<br/>
                  • Scroll to zoom<br/>
                  • Click node to analyze
                </div>
              </div>
            </div>
          </div>
        </div>
      </Drawer>
    </div>
  );
}
