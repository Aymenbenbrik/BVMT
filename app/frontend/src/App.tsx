import { Routes, Route } from 'react-router-dom';
import { ConfigProvider, theme } from 'antd';
import { darkTheme } from './theme';
import Navigation from './components/Navigation';
import MarketOverview from './pages/MarketOverview';
import StockDeepDive from './pages/StockDeepDive';
import GraphNetwork from './pages/GraphNetwork';

function App() {
  return (
    <ConfigProvider theme={{ ...darkTheme, algorithm: theme.darkAlgorithm }}>
      <div style={{ minHeight: '100vh', background: '#0a0e1a' }}>
        <Navigation />
        <Routes>
          <Route path="/" element={<MarketOverview />} />
          <Route path="/stock/:ticker" element={<StockDeepDive />} />
          <Route path="/graph" element={<GraphNetwork />} />
        </Routes>
      </div>
    </ConfigProvider>
  );
}

export default App;
