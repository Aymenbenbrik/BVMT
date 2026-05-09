import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Tabs, Button, Spin, Row, Col, Card, Tag, Collapse, message } from 'antd';
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  LineChartOutlined,
  BankOutlined,
  CommentOutlined,
  ShareAltOutlined,
  BulbOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { fetchPrediction, fetchStockHistory, refreshTicker, generateXaiNarratives } from '../api/client';
import DirectionBadge from '../components/DirectionBadge';
import PriceHorizonChart from '../components/PriceHorizonChart';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import FinancialRatioCard from '../components/FinancialRatioCard';
import SentimentGauge from '../components/SentimentGauge';
import ArticleCard from '../components/ArticleCard';
import NeighborCard from '../components/NeighborCard';

export default function StockDeepDive() {
  const { ticker } = useParams<{ ticker: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const decodedTicker = decodeURIComponent(ticker || '');

  const [xaiNarratives, setXaiNarratives] = useState<{ technical: string; fundamental: string; graph: string } | null>(() => {
    const cached = sessionStorage.getItem(`xai-narrative:${decodedTicker}:v1`);
    if (!cached) return null;
    try {
      return JSON.parse(cached) as { technical: string; fundamental: string; graph: string };
    } catch {
      return null;
    }
  });
  const [xaiNarrativeLoading, setXaiNarrativeLoading] = useState(false);
  const [xaiNarrativeError, setXaiNarrativeError] = useState<string | null>(null);

  const { data: prediction, isLoading, isError } = useQuery({
    queryKey: ['prediction', decodedTicker],
    queryFn: () => fetchPrediction(decodedTicker),
    enabled: !!decodedTicker,
  });

  const { data: historyData } = useQuery({
    queryKey: ['history', decodedTicker],
    queryFn: () => fetchStockHistory(decodedTicker),
    enabled: !!decodedTicker,
  });

  const handleRefresh = async () => {
    try {
      message.loading({ content: `Re-analyzing ${decodedTicker}…`, key: 'refresh', duration: 0 });
      await refreshTicker(decodedTicker);
      queryClient.invalidateQueries({ queryKey: ['prediction', decodedTicker] });
      queryClient.invalidateQueries({ queryKey: ['history', decodedTicker] });
      message.success({ content: 'Analysis refreshed!', key: 'refresh' });
    } catch {
      message.error({ content: 'Refresh failed', key: 'refresh' });
    }
  };

  const buildXaiPayload = () => {
    const techExp = xai.technical_explanation || {};
    const fundExp = xai.fundamental_explanation || {};
    const graphExp = xai.graph_explanation || {};

    const names = Array.isArray(fundExp.feature_names) ? (fundExp.feature_names as string[]) : [];
    const vals = Array.isArray(fundExp.shap_values) ? (fundExp.shap_values as number[]) : [];
    const fundPairs = names.map((name, i) => ({ name, value: vals[i] ?? 0 }));

    const explainVersion = (prediction?.explanation_facts?.explain_version as string) || 'v1';

    return {
      ticker: decodedTicker,
      explain_version: explainVersion,
      technical: {
        encoder_variable_importance: techExp.encoder_variable_importance || {},
        top_attention_timesteps: techExp.top_attention_timesteps || [],
        note: techExp.note || '',
      },
      fundamental: {
        positive_factors: fundPairs.filter((x) => x.value > 0),
        negative_factors: fundPairs.filter((x) => x.value < 0),
        neutral_factors: fundPairs.filter((x) => x.value === 0),
        base_value: fundExp.base_value ?? null,
        predicted_class: fundExp.predicted_class ?? null,
      },
      graph: {
        top_neighbors: graph.top_neighbors || [],
        interpretation: graphExp.interpretation || '',
        top_influencing_stocks: graphExp.top_influencing_stocks || [],
      },
    };
  };

  const handleGenerateXaiNarratives = async () => {
    if (xaiNarrativeLoading || xaiNarratives) return;

    const hasXaiData =
      (xai.technical_explanation && Object.keys(xai.technical_explanation).length > 0) ||
      (xai.fundamental_explanation && Object.keys(xai.fundamental_explanation).length > 0) ||
      (xai.graph_explanation && Object.keys(xai.graph_explanation).length > 0);

    if (!hasXaiData) {
      setXaiNarrativeError('XAI data is not available for this ticker.');
      return;
    }

    setXaiNarrativeLoading(true);
    setXaiNarrativeError(null);

    try {
      const payload = buildXaiPayload();
      const res = await generateXaiNarratives(payload);
      const result = {
        technical: res.technical,
        fundamental: res.fundamental,
        graph: res.graph,
      };
      setXaiNarratives(result);
      sessionStorage.setItem(`xai-narrative:${decodedTicker}:${payload.explain_version}`, JSON.stringify(result));
    } catch (err: unknown) {
      const status =
        typeof err === 'object' && err && 'response' in err
          ? (err as { response?: { status?: number } }).response?.status
          : undefined;
      if (status === 503) {
        setXaiNarrativeError('LLM not configured. Add EXPLAINER_LLM_API_KEY to your .env file.');
      } else if (status === 429) {
        setXaiNarrativeError('Rate limit reached. Please try again in a moment.');
      } else {
        setXaiNarrativeError('Failed to generate explanations. Please try again.');
      }
    } finally {
      setXaiNarrativeLoading(false);
    }
  };

  const renderXaiNarrative = (text?: string) => {
    if (xaiNarrativeLoading && !text) {
      return <div style={{ color: '#6b7280', marginBottom: 12 }}>Generating explanation...</div>;
    }
    if (text) {
      return <div style={{ marginBottom: 12, lineHeight: 1.7, color: '#d1d5db' }}>{text}</div>;
    }
    return <div style={{ color: '#6b7280', marginBottom: 12 }}>Explanation will appear after generation.</div>;
  };

  const handleTabChange = (key: string) => {
    if (key === 'xai') {
      handleGenerateXaiNarratives();
    }
  };

  if (isLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
        <div style={{ textAlign: 'center' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16, color: '#9ca3af' }}>
            Running AI agents for <b>{decodedTicker}</b>…
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
            First analysis takes 10-30 seconds
          </div>
        </div>
      </div>
    );
  }

  if (isError || !prediction) {
    return (
      <div className="page-container">
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/')} type="text" style={{ color: '#9ca3af', marginBottom: 16 }}>
          Back to Market
        </Button>
        <div style={{ textAlign: 'center', padding: 60, color: '#ef4444' }}>
          Failed to load analysis for {decodedTicker}
        </div>
      </div>
    );
  }

  const p = prediction;
  const orch = p.orchestration;
  const tech = p.technical;
  const fund = p.fundamental;
  const sent = p.sentiment;
  const graph = p.graph;
  const xai = p.xai;

  // Gauge chart option for final score
  const scoreGaugeOption = {
    backgroundColor: 'transparent',
    series: [{
      type: 'gauge',
      radius: '90%',
      center: ['50%', '55%'],
      startAngle: 200,
      endAngle: -20,
      min: 0,
      max: 100,
      progress: { show: true, width: 12, itemStyle: { color: '#3b82f6' } },
      axisLine: { lineStyle: { width: 12, color: [[1, '#1f2937']] } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      pointer: { show: false },
      detail: {
        valueAnimation: true,
        fontSize: 28,
        fontWeight: 700,
        color: '#f9fafb',
        offsetCenter: [0, '10%'],
        formatter: '{value}',
      },
      data: [{ value: orch.final_score.toFixed(0) }],
    }],
  };

  const tabItems = [
    {
      key: 'technical',
      label: <span><LineChartOutlined /> Technical</span>,
      children: (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Model info cards */}
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 12 }}>
              <ExperimentOutlined /> What the model used
            </div>
            <Row gutter={[12, 12]}>
              {[
                { label: 'Training Data', value: tech.model_info?.training_period || '2016–2023' },
                { label: 'Encoder Window', value: `${tech.model_info?.encoder_window_days || 60} days` },
                { label: 'Prediction Horizon', value: `${tech.model_info?.prediction_horizon_days || 7} days ahead` },
                { label: 'Directional Accuracy', value: `${tech.model_info?.model_accuracy_pct || 77.4}%` },
                { label: 'Expected Move', value: tech.expected_move_pct !== undefined && tech.expected_move_pct !== null ? `${tech.expected_move_pct > 0 ? '+' : ''}${tech.expected_move_pct}%` : 'N/A' },
                { label: 'Last Price', value: tech.current_close !== undefined && tech.current_close !== null ? `${tech.current_close.toFixed(3)} DT` : 'N/A' },
                { label: 'Mode', value: tech.model_info?.prediction_mode || 'price' },
              ].map((item, i) => (
                <Col xs={12} sm={8} md={3} key={i}>
                  <div style={{ background: '#0d1321', border: '1px solid #1f2937', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#f9fafb' }}>{item.value}</div>
                  </div>
                </Col>
              ))}
            </Row>
          </div>

          {/* Price Chart */}
          <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 12 }}>
              📈 Price Forecast — 7-Day Horizon
            </div>
            <PriceHorizonChart
              history={historyData?.history || []}
              predicted={historyData?.predicted || []}
              direction={tech.direction}
              height={380}
            />
          </div>

          {/* Predicted Days Table */}
          {tech.price_horizon.length > 0 && (
            <div style={{ background: '#0d1321', border: '1px solid #1f2937', borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#9ca3af', marginBottom: 10 }}>Predicted Price Horizon</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 8 }}>
                {tech.price_horizon.map((ph) => (
                  <div key={ph.day_number} style={{ textAlign: 'center', padding: '8px 4px', background: '#111827', borderRadius: 6 }}>
                    <div style={{ fontSize: 10, color: '#6b7280' }}>Day {ph.day_number}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: '#f9fafb' }}>{ph.predicted_price.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Feature Importance */}
          <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 12 }}>
              🎯 Top Driving Features
            </div>
            <FeatureImportanceChart features={tech.top_features} height={280} />
          </div>
        </div>
      ),
    },
    {
      key: 'fundamental',
      label: <span><BankOutlined /> Fundamental</span>,
      children: (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Model info */}
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 12 }}>
              <ExperimentOutlined /> What data was used
            </div>
            <Row gutter={[12, 12]}>
              {[
                { label: 'Filing Period', value: fund.period || 'N/A' },
                { label: 'Company Type', value: fund.company_type || 'N/A' },
                { label: 'Algorithm', value: fund.model_info?.algorithm || 'XGBoost' },
                { label: 'Cross Validation', value: fund.model_info?.cross_validation || 'LOTO' },
                { label: 'Features Used', value: fund.model_info?.features_used || 17 },
                { label: 'Health Classes', value: '4 classes' },
              ].map((item, i) => (
                <Col xs={12} sm={8} md={4} key={i}>
                  <div style={{ background: '#0d1321', border: '1px solid #1f2937', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#f9fafb' }}>{String(item.value)}</div>
                  </div>
                </Col>
              ))}
            </Row>
          </div>

          {/* Ratios */}
          <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 16 }}>
              📊 Financial Ratios
            </div>
            <FinancialRatioCard ratios={fund.ratios} companyType={fund.company_type} />
          </div>

          {/* XAI — Fundamental */}
          {xai.fundamental_explanation && Object.keys(xai.fundamental_explanation).length > 0 && (
            <div className="info-box">
              <div style={{ fontWeight: 600, marginBottom: 8, color: '#f9fafb' }}>🔍 XAI Explanation</div>
              {xai.fundamental_explanation.error ? (
                <div style={{ color: '#6b7280' }}>{xai.fundamental_explanation.error}</div>
              ) : (
                <div>
                  {xai.fundamental_explanation.feature_names && xai.fundamental_explanation.shap_values && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                      {(xai.fundamental_explanation.feature_names as string[]).map((name: string, i: number) => {
                        const sv = (xai.fundamental_explanation.shap_values as number[])[i] || 0;
                        return (
                          <Tag key={name} color={sv > 0 ? 'green' : sv < 0 ? 'red' : 'default'}>
                            {name}: {sv > 0 ? '+' : ''}{sv.toFixed(3)}
                          </Tag>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'sentiment',
      label: <span><CommentOutlined /> Sentiment</span>,
      children: (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Model info */}
          <Row gutter={[12, 12]}>
            {[
              { label: 'Articles', value: sent.article_count },
              { label: 'Model', value: sent.model_info?.model_name || 'FinBERT-FR' },
              { label: 'Language', value: sent.model_info?.language || 'French' },
              { label: 'Source', value: sent.model_info?.source || 'ilboursa.com' },
            ].map((item, i) => (
              <Col xs={12} sm={6} key={i}>
                <div style={{ background: '#0d1321', border: '1px solid #1f2937', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#f9fafb' }}>{String(item.value)}</div>
                </div>
              </Col>
            ))}
          </Row>

          {/* Gauge */}
          <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: 20 }}>
            <SentimentGauge
              score={sent.score}
              label={sent.label}
              confidence={sent.confidence}
              height={280}
            />
          </div>

          {/* Articles */}
          {sent.recent_articles.length > 0 && (
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 12 }}>📰 Recent Articles</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {sent.recent_articles.map((a, i) => (
                  <ArticleCard key={i} title={a.title} date={a.date} sentimentScore={a.sentiment_score} source={a.source} />
                ))}
              </div>
            </div>
          )}

          {sent.article_count === 0 && (
            <div style={{ textAlign: 'center', padding: 40, color: '#6b7280' }}>
              No news articles available for this stock
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'graph',
      label: <span><ShareAltOutlined /> Graph</span>,
      children: (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Info */}
          <Row gutter={[12, 12]}>
            {[
              { label: 'Signal', value: graph.signal },
              { label: 'Confidence', value: `${(graph.confidence * 100).toFixed(0)}%` },
              { label: 'Score', value: graph.neighborhood_score.toFixed(3) },
              { label: 'Gating', value: graph.confidence_gating_applied ? 'Applied' : 'Not applied' },
            ].map((item, i) => (
              <Col xs={12} sm={6} key={i}>
                <div style={{ background: '#0d1321', border: '1px solid #1f2937', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#f9fafb' }}>{item.value}</div>
                </div>
              </Col>
            ))}
          </Row>

          {/* Neighbors */}
          {graph.top_neighbors.length > 0 && (
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#f9fafb', marginBottom: 12 }}>
                🔗 Top Influencing Neighbors
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {graph.top_neighbors.map((n, i) => (
                  <NeighborCard key={n.ticker} neighbor={n} rank={i + 1} />
                ))}
              </div>
            </div>
          )}

          {/* Graph explanation */}
          {xai.graph_explanation && Object.keys(xai.graph_explanation).length > 0 && (
            <div className="info-box">
              <div style={{ fontWeight: 600, marginBottom: 8, color: '#f9fafb' }}>🔍 Graph Influence Narrative</div>
              <div>{xai.graph_explanation.interpretation || 'No graph narrative available.'}</div>
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'xai',
      label: <span><BulbOutlined /> Explanation</span>,
      children: (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* AI Article */}
          {p.ai_article.content && (
            <Card title="📝 AI-Generated Article" style={{ borderColor: '#1f2937' }}>
              <div style={{ lineHeight: 1.8, color: '#d1d5db' }}>{p.ai_article.content}</div>
            </Card>
          )}

          {/* Collapsible sections */}
          {xaiNarrativeError && (
            <div style={{
              background: '#1c1317',
              border: '1px solid #7f1d1d',
              borderRadius: 8,
              padding: '12px 16px',
              color: '#fca5a5',
              fontSize: 13,
            }}>
              {xaiNarrativeError}
            </div>
          )}

          <Collapse
            ghost
            items={[
              {
                key: 'tech-xai',
                label: <span style={{ color: '#60a5fa', fontWeight: 600 }}>Why the technical model predicted this direction</span>,
                children: (
                  <div style={{ color: '#d1d5db' }}>
                    {renderXaiNarrative(xaiNarratives?.technical)}
                    {xai.technical_explanation?.encoder_variable_importance && (
                      <div style={{ marginBottom: 16 }}>
                        <div style={{ fontWeight: 600, marginBottom: 8 }}>Variable Importance:</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                          {Object.entries(xai.technical_explanation.encoder_variable_importance as Record<string, number>)
                            .sort(([, a], [, b]) => (b as number) - (a as number))
                            .slice(0, 10)
                            .map(([name, val]) => (
                              <Tag key={name} color="blue">{name}: {(val as number).toFixed(3)}</Tag>
                            ))}
                        </div>
                      </div>
                    )}
                    {xai.technical_explanation?.top_attention_timesteps && (
                      <div>
                        <div style={{ fontWeight: 600, marginBottom: 8 }}>Top Attention Timesteps:</div>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {(xai.technical_explanation.top_attention_timesteps as number[]).map((ts, i) => (
                            <Tag key={i} color="purple">Day offset: {ts}</Tag>
                          ))}
                        </div>
                      </div>
                    )}
                    {xai.technical_explanation?.note && (
                      <div style={{ marginTop: 12, color: '#6b7280' }}>{xai.technical_explanation.note}</div>
                    )}
                  </div>
                ),
              },
              {
                key: 'fund-xai',
                label: <span style={{ color: '#10b981', fontWeight: 600 }}>Why the fundamental score is what it is</span>,
                children: (
                  <div>
                    {renderXaiNarrative(xaiNarratives?.fundamental)}
                    {xai.fundamental_explanation?.feature_names && xai.fundamental_explanation?.shap_values ? (
                      <div style={{ display: 'flex', gap: 24 }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 600, marginBottom: 8, color: '#10b981' }}>Positive Factors</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {(xai.fundamental_explanation.feature_names as string[])
                              .map((n: string, i: number) => ({ name: n, val: (xai.fundamental_explanation.shap_values as number[])[i] }))
                              .filter((x: { val: number }) => x.val > 0)
                              .sort((a: { val: number }, b: { val: number }) => b.val - a.val)
                              .map((x: { name: string; val: number }) => (
                                <Tag key={x.name} color="green">{x.name}: +{x.val.toFixed(3)}</Tag>
                              ))}
                          </div>
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 600, marginBottom: 8, color: '#ef4444' }}>Negative Factors</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {(xai.fundamental_explanation.feature_names as string[])
                              .map((n: string, i: number) => ({ name: n, val: (xai.fundamental_explanation.shap_values as number[])[i] }))
                              .filter((x: { val: number }) => x.val < 0)
                              .sort((a: { val: number }, b: { val: number }) => a.val - b.val)
                              .map((x: { name: string; val: number }) => (
                                <Tag key={x.name} color="red">{x.name}: {x.val.toFixed(3)}</Tag>
                              ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#6b7280' }}>No SHAP explanation available</div>
                    )}
                  </div>
                ),
              },
              {
                key: 'graph-xai',
                label: <span style={{ color: '#f59e0b', fontWeight: 600 }}>How neighboring stocks influenced this</span>,
                children: (
                  <div style={{ color: '#d1d5db' }}>
                    {renderXaiNarrative(xaiNarratives?.graph)}
                    {xai.graph_explanation?.interpretation || 'No graph influence narrative available.'}
                  </div>
                ),
              },
            ]}
          />

          {/* Decision explanation */}
          <div className="info-box" style={{ marginTop: 8 }}>
            <div style={{ fontWeight: 700, marginBottom: 8, color: '#f9fafb', fontSize: 15 }}>
              🎯 Final Decision Explanation
            </div>
            <div style={{ lineHeight: 1.8 }}>{orch.decision_explanation}</div>
          </div>

        </div>
      ),
    },
  ];

  return (
    <div className="page-container fade-in">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/')}
            type="text"
            style={{ color: '#9ca3af' }}
          />
          <div>
            <div style={{ fontSize: 24, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 12 }}>
              {decodedTicker}
              <span className={`company-badge ${p.metadata.company_type || 'non_bank'}`}>
                {p.metadata.company_type || 'non_bank'}
              </span>
            </div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>
              Last updated: {p.metadata.run_timestamp || 'N/A'}
              {p.metadata.cache_hit && <Tag color="blue" style={{ marginLeft: 8, fontSize: 10 }}>CACHED</Tag>}
            </div>
          </div>
        </div>
        <Button icon={<ReloadOutlined />} onClick={handleRefresh} style={{ borderColor: '#374151' }}>
          Refresh Analysis
        </Button>
      </div>

      {/* Hero Row */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={6}>
          <div className="metric-card" style={{ textAlign: 'center' }}>
            <div style={{ marginBottom: 8 }}>
              <DirectionBadge direction={orch.final_direction} size="large" />
            </div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>
              Confidence: {(orch.final_confidence * 100).toFixed(0)}%
            </div>
            <div className="metric-label" style={{ marginTop: 8 }}>Final Signal</div>
          </div>
        </Col>
        <Col xs={24} sm={6}>
          <div className="metric-card" style={{ textAlign: 'center' }}>
            <ReactECharts option={scoreGaugeOption} style={{ height: 120 }} notMerge />
            <div className="metric-label">Composite Score</div>
          </div>
        </Col>
        <Col xs={24} sm={6}>
          <div className="metric-card">
            <DirectionBadge direction={tech.direction} size="default" />
            <div style={{ marginTop: 8, fontSize: 13, color: '#9ca3af' }}>
              TFT: {(tech.confidence * 100).toFixed(0)}% confidence
            </div>
            {tech.expected_move_pct !== undefined && tech.expected_move_pct !== null && (
              <div style={{ marginTop: 4, fontSize: 12, color: tech.expected_move_pct >= 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                Move: {tech.expected_move_pct > 0 ? '+' : ''}{tech.expected_move_pct}%
              </div>
            )}
            {tech.current_close !== undefined && tech.current_close !== null && (
              <div style={{ marginTop: 2, fontSize: 11, color: '#6b7280' }}>
                Last Price: {tech.current_close.toFixed(2)} DT
              </div>
            )}
            <div className="metric-label" style={{ marginTop: 8 }}>Technical Model</div>
          </div>
        </Col>
        <Col xs={24} sm={6}>
          <div className="metric-card">
            <div style={{
              fontSize: 22,
              fontWeight: 700,
              color: fund.health_class === 'STRONG' ? '#10b981' : fund.health_class === 'WEAK' || fund.health_class === 'CRITICAL' ? '#ef4444' : '#f59e0b',
            }}>
              {fund.health_class || 'N/A'}
            </div>
            <div style={{ marginTop: 4, fontSize: 13, color: '#9ca3af' }}>
              Score: {fund.health_score.toFixed(1)} / 3.0
            </div>
            <div className="metric-label" style={{ marginTop: 8 }}>Company Health</div>
          </div>
        </Col>
      </Row>

      {/* Tabs */}
      <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '16px 24px' }}>
        <Tabs items={tabItems} defaultActiveKey="technical" size="large" onChange={handleTabChange} />
      </div>
    </div>
  );
}
