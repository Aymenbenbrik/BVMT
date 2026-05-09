import { useEffect, useRef, useCallback } from 'react';
import type { GraphNode, GraphEdge } from '../types';

interface StockGraph3DProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  height?: number;
  focusTicker?: string;
  onNodeClick?: (ticker: string) => void;
  threshold?: number;
}

// Dynamically imported ForceGraph3D to avoid SSR issues
let ForceGraph3D: any = null;

export default function StockGraph3D({
  nodes,
  edges,
  height = 600,
  focusTicker,
  onNodeClick,
  threshold = 0.4,
}: StockGraph3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);

  // Lazy load the ForceGraph3D module
  useEffect(() => {
    if (!ForceGraph3D) {
      import('react-force-graph-3d').then((mod) => {
        ForceGraph3D = mod.default;
        // Force re-render
        if (containerRef.current) {
          containerRef.current.dispatchEvent(new Event('graphloaded'));
        }
      });
    }
  }, []);

  const filteredEdges = edges.filter((e) => e.weight >= threshold);

  const baseNodes = nodes.map((n) => ({
    id: n.id,
    label: n.label,
    company_type: n.company_type,
    direction: n.final_direction,
    confidence: n.final_confidence,
    score: n.final_score,
    isFocused: focusTicker ? n.id === focusTicker : false,
    isNeighbor: focusTicker
      ? filteredEdges.some(
          (e) =>
            (e.source === focusTicker && e.target === n.id) ||
            (e.target === focusTicker && e.source === n.id)
        )
      : false,
  }));

  const baseLinks = filteredEdges.map((e) => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
    correlation_type: e.correlation_type,
  }));

  const graphData = {
    nodes: focusTicker ? baseNodes.filter(n => n.isFocused || n.isNeighbor) : baseNodes,
    links: focusTicker ? baseLinks.filter(e => e.source === focusTicker || e.target === focusTicker) : baseLinks,
  };

  const getNodeColor = useCallback(
    (node: any) => {
      if (focusTicker) {
        if (node.isFocused) return '#ffffff';
        if (!node.isNeighbor) return '#374151';
      }
      switch (node.direction) {
        case 'UP': return '#10b981';
        case 'DOWN': return '#ef4444';
        case 'NEUTRAL':
        case 'HOLD': return '#f59e0b';
        default: return '#4b5563';
      }
    },
    [focusTicker]
  );

  const getNodeSize = useCallback(
    (node: any) => {
      if (node.isFocused) return 10;
      const base = node.score ? Math.max(3, node.score / 15) : 3;
      if (focusTicker && !node.isNeighbor) return 2;
      return base;
    },
    [focusTicker]
  );

  if (!ForceGraph3D) {
    return (
      <div
        ref={containerRef}
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#6b7280',
          background: '#0a0e1a',
          borderRadius: 12,
          border: '1px solid #1f2937',
        }}
      >
        Loading 3D Graph…
      </div>
    );
  }

  return (
    <div style={{ height, borderRadius: 12, overflow: 'hidden', border: '1px solid #1f2937' }}>
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        height={height}
        backgroundColor="#0a0e1a"
        nodeLabel={(node: any) => {
          const dir = node.direction || 'N/A';
          const conf = node.confidence != null ? `${(node.confidence * 100).toFixed(0)}%` : 'N/A';
          return `<div style="background:#1f2937;padding:8px 12px;border-radius:8px;font-size:12px;color:#f9fafb;border:1px solid #374151">
            <b>${node.label}</b><br/>Direction: ${dir}<br/>Confidence: ${conf}
          </div>`;
        }}
        nodeColor={getNodeColor}
        nodeVal={getNodeSize}
        nodeOpacity={0.9}
        linkColor={(link: any) => link.correlation_type === 'negative' ? '#6b728066' : '#3b82f644'}
        linkWidth={(link: any) => Math.max(0.3, link.weight * 2)}
        linkOpacity={0.4}
        onNodeClick={(node: any) => {
          if (onNodeClick && node.id) {
            onNodeClick(node.id);
          }
        }}
        enableNodeDrag={true}
        cooldownTicks={100}
        warmupTicks={50}
      />
    </div>
  );
}
