# BVMT Multi-Agent AI Platform
> **Thesis Project**: Multi-agent stock analysis and explanation system for the Tunisian market  
> **Status**: Local FastAPI + React dashboard implemented, demo polish in progress  
> **Last Updated**: May 9, 2026

---

## 📋 Executive Summary

**BVMT** is a local multi-agent platform for BVMT stock analysis. It combines:
- **TechnicalAgent**: 7-day price forecasting and direction signal generation
- **FundamentalAgent**: XGBoost-based company health scoring
- **SentimentAgent**: French financial news sentiment analysis
- **GraphAgent**: Stock relationship modeling and 3D graph visualization
- **XAIAgent**: Explanation layer for technical, fundamental, and graph outputs
- **OrchestratorAgent**: Final decision engine with adaptive weights

The current user-facing application is a FastAPI backend plus a React dashboard that shows market overview, stock deep dives, and the graph network for the full BVMT universe.

---

## 📊 Project Status & Milestones

### ✅ Completed
- [x] **Data pipeline**: BVMT prices, financial ratios, and news ingestion are in place.
- [x] **Technical model**: 7-day forecasting pipeline with 77.4% directional accuracy on 2024 data.
- [x] **Fundamental model**: XGBoost health classifier with 85.8% Leave-One-Ticker-Out CV accuracy.
- [x] **Sentiment model**: French finance sentiment scoring for ilboursa.com articles.
- [x] **Graph model**: 3D network payload and confidence-gated graph output.
- [x] **Backend API**: FastAPI endpoints for stocks, predictions, history, graph data, and cache management.
- [x] **Frontend**: React dashboard with market overview, stock deep dive, and graph views.

### 🔄 In Progress
- [ ] **Demo packaging**: Final polish for thesis defense use.
- [ ] **Documentation**: Keep README, thesis notes, and report chapters aligned with the current stack.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                            │
│        (Final score + explanation synthesis)               │
└────────────────┬───────────────────┬──────────────────────┘
     │                   │
  ┌────────▼────────┐  ┌──────▼──────────┐
  │  TECHNICAL      │  │  FUNDAMENTAL    │
  │  AGENT          │  │  AGENT          │
  │  7-day forecast │  │  Health tiers   │
  └────────┬────────┘  └──────┬──────────┘
     │                   │
  ┌────────▼────────┐  ┌──────▼──────────┐
  │  SENTIMENT      │  │  GRAPH / XAI    │
  │  AGENT          │  │  AGENT          │
  │  French news    │  │  3D network     │
  └────────┬────────┘  └──────┬──────────┘
     │                   │
  ┌────────▼───────────────────▼────────┐
  │     FASTAPI BACKEND + REACT UI      │
  │  /stocks /prediction /graph-data    │
  │  /market-overview /stock-history    │
  └──────────────────────────────────────┘
```

### Core Components

| Module | Purpose | Status |
|--------|---------|--------|
| `agents/base_agent.py` | Abstract agent interface | ✅ Complete |
| `agents/technical_agent.py` | 7-day price forecasting | ✅ Complete |
| `agents/fundamental_agent.py` | XGBoost financial health | ✅ Complete |
| `agents/sentiment_agent.py` | French sentiment scoring | ✅ Complete |
| `agents/graph_agent.py` | Relationship graph and confidence gating | ✅ Complete |
| `agents/xai_agent.py` | Explanation synthesis | ✅ Complete |
| `agents/orchestrator.py` | Multi-agent decision engine | ✅ Complete |
| `app/backend/main.py` | FastAPI application | ✅ Complete |
| `app/frontend/src/pages/*` | React dashboard pages | ✅ Complete |
| `training/train_fundamental.py` | XGBoost training | ✅ Complete |
| `training/train_tft_v3.py` | TFT training / evaluation | ✅ Complete |
| `notebooks/*.ipynb` | Data exploration and validation | ✅ Multiple |

---

## 🚀 Quick Start

### Prerequisites
```bash
python 3.10+
node and npm
postgresql 16
```

### Start the backend
```bash
uvicorn app.backend.main:app --reload --port 8000
```

### Start the frontend
```bash
cd app/frontend
npm install
npm run dev
```

### One-command launch

- Windows: `start.bat`
- macOS/Linux: `start.sh`

### Optional cache warm-up

```bash
python -m uvicorn app.backend.main:app --port 8000
```

Then call `POST /cache/precompute` from the backend if you want the cache warmed before a demo.

### Enable On-Demand LLM Narratives
To enable the "Generate narrative" button on the UI, configure your `.env` file with an OpenAI-compatible API key (e.g., OpenAI, GitHub Models, Groq):
```env
EXPLAINER_LLM_API_KEY=your_api_key_here
# Example for GitHub Models:
EXPLAINER_LLM_BASE_URL=https://models.inference.ai.azure.com
EXPLAINER_LLM_MODEL=gpt-4o-mini
```
*Note: This feature is strictly on-demand. Bulk processing and background prediction jobs will never trigger the LLM to avoid excessive costs.*

---

## 📁 Project Structure

```
bvmt_project/
├── agents/
├── app/
│   ├── backend/
│   └── frontend/
├── data/
├── models/
├── notebooks/
├── reports/
├── results/
├── scripts/
├── training/
├── tests/
├── start.bat
├── start.sh
├── precompute.bat
├── precompute.sh
└── README.md
```

---

## 📊 Key Results

| Component | Result | Notes |
|---|---:|---|
| TechnicalAgent | 77.4% directional accuracy | Current best technical result on 2024 validation data |
| FundamentalAgent | 85.8% LOTO-CV accuracy | XGBoost on 37 tickers with 4-class health labels |
| SentimentAgent | French sentiment pre-scoring | Built on bardsai/finance-sentiment-fr-base |
| GraphAgent | 3D relationship payload | Supports confidence gating and lead/lag style influence |

**Technical takeaway**: the current technical model forecasts 7 daily prices and uses the final-day direction for the main signal.

**Fundamental takeaway**: company health is strongest when the model has stable ratio coverage, especially for banks.

---

## 🔍 Feature Engineering

### Current input contract

The dashboard and backend currently treat the technical model as a 15-feature, 60-day window forecaster.

**Price features**:
- `close_price`, `open_price`, `high_price`, `low_price`

**Volume**:
- `volume`

**Technical indicators**:
- `daily_return`
- `ma_5`
- `ma_20`
- `ma_50`
- `rsi_14`
- `volatility_20`
- `volume_ma_20`

**Relative features**:
- `price_to_ma20_ratio`
- `rsi_momentum`
- `volume_spike`

### Technical output

- 7 predicted prices for the next 7 trading days.
- 7 predicted return percentages.
- One main direction: `UP`, `DOWN`, or `HOLD`.
- One confidence score used by the orchestrator.

### Fundamental output

- 4-class health label: `CRITICAL`, `WEAK`, `MODERATE`, `STRONG`.
- Ratio groups for banks and non-banks.
- XGBoost confidence and normalized health score.

### Sentiment output

- French article score in `[-1, +1]`.
- Label in `NEGATIVE`, `NEUTRAL`, `POSITIVE`.
- Recent article list for the stock deep dive page.

### Graph output

- 3D stock network nodes and edges.
- Correlation-driven influence signals.
- Confidence gating when the graph signal is too weak.

---

## 🎯 Current Modeling Notes

- The earlier quantile-regression collapse is no longer the active user flow.
- The live application now focuses on the 7-day price forecasting contract used by the backend.
- The graph signal is only blended when confidence is meaningful.
- The frontend surfaces the technical, fundamental, sentiment, and graph outputs side by side.

---

## 🧪 Validation & Testing

### Unit Tests
```bash
pytest tests/ -v
```

**Coverage**:
- `test_sprint1.py`: Data pipeline validation
- `test_data_agent.py`: Agent interface compliance

### Integration Tests
```bash
python test_orchestrator.py
```

### Frontend Checks
```bash
cd app/frontend
npm run build
npm run lint
```

### Backend Health
```bash
curl http://127.0.0.1:8000/health
```

---

## 📚 Documentation

### Thesis Chapters
| Chapter | Title | Status |
|---------|-------|--------|
| **1** | Introduction & Literature Review | ✅ Draft |
| **2** | Problem Statement (BVMT context) | ✅ Draft |
| **3** | Data Pipeline & Feature Engineering | ✅ Complete |
| **4** | Model Architecture | 🔄 In Progress |
| **5** | Results & Evaluation | 🔄 In Progress |
| **6** | Contributions & Future Work | ⏳ Pending |

### Technical Notes
- `research_notes.md`: Development log, findings, decisions
- `shift.md`: Updated roadmap and sprint re-scope notes
- `PROJECT_EXPLANATION.md`: Thesis-oriented project explanation
- `reports/data_cleaning/PROJECT_ISSUES_AND_RESOLUTIONS.md`: data quality and fixes

---

## 🔧 Development Workflow

### Sprint Cadence
- **Sprint 1**: Infrastructure, data pipeline, feature engineering
- **Sprint 2**: Technical model training and evaluation
- **Sprint 3**: Fundamental and sentiment agents
- **Sprint 4**: Graph and explainability layer
- **Sprint 5**: Integration and evaluation
- **Sprint 6**: Local demo dashboard and packaging

### Current Focus
1. Keep the FastAPI backend and React frontend in sync.
2. Preserve the current technical and fundamental results.
3. Finish demo packaging for the defense machine.
4. Update thesis chapters with the current dashboard and API contract.

### Contribution Guidelines
- Create a feature branch: `git checkout -b feature/your-feature`
- Commit with clear messages: `git commit -m "Add: describe change"`
- Push & create PR with summary + test results
- Keep changes focused and avoid unrelated refactors

---

## 📞 Contact & Support

**Project Lead**: [Your Name]  
**Institution**: [Your University]  
**Advisor**: [Advisor Name]  
**GitHub Repo**: [Repository link]

**Questions?**
- 📧 Email: [your-email]
- 💬 Issues: GitHub Issues tracker
- 📖 Docs: README, `shift.md`, and `PROJECT_EXPLANATION.md`

---

## 📄 License

All code and data are provided for academic and non-commercial use.

---

## 🎉 Acknowledgments

- BVMT for historical price data
- PyTorch Lightning for training orchestration
- PyTorch Forecasting for the technical forecasting backbone
- scikit-learn and XGBoost for the fundamental model
- React, FastAPI, and Ant Design for the current dashboard

---

**Last Updated**: May 9, 2026  
**Next Milestone**: Final demo packaging and thesis chapter polish
