# Automated Risk Assessment Tool for Portfolio Optimization

## Overview
An AI-powered portfolio risk analysis system that allows users to upload portfolio allocations (stocks, bonds, crypto), calculates quantitative financial risk metrics, generates scenario-based market stress forecasts using LLMs, and recommends optimized portfolio allocations.

## Problem Statement
Portfolio managers and investors struggle to manually assess portfolio risk during volatile markets. This system automates risk calculation, stress testing, and optimization under hypothetical macroeconomic scenarios.

## Core Features
- Portfolio CSV upload and validation
- Historical market data fetching via Yahoo Finance
- Risk metrics: VaR, CVaR, Sharpe Ratio, Volatility, Correlation Matrix
- Monte Carlo simulation (10,000+ iterations)
- Portfolio optimization using constrained optimization (CVXPY)
- LLM-based scenario generation (e.g., "What if interest rates rise by 2%?")
- Asset reallocation recommendation engine
- Interactive Streamlit frontend dashboard
- Dockerized microservices
- Kubernetes deployment-ready

## Architecture

```
risk-tool/
├── backend/        # FastAPI REST API — routes, schemas, config
├── risk_engine/    # Core financial computation (VaR, CVaR, Sharpe, Monte Carlo)
├── optimizer/      # Portfolio optimization using CVXPY
├── llm_service/    # Hugging Face LLM integration for scenario analysis
├── frontend/       # Streamlit dashboard
├── data/           # Sample portfolios, raw/processed market data
├── tests/          # Unit and integration test suites
├── docker/         # Dockerfiles and docker-compose
├── kubernetes/     # K8s deployment manifests
└── notebooks/      # Exploratory analysis and prototyping
```

## Technology Stack
| Layer | Technology |
|---|---|
| API | FastAPI |
| Frontend | Streamlit |
| Data | Pandas, NumPy, yfinance |
| Risk Math | SciPy, NumPy |
| Optimization | CVXPY |
| LLM | Hugging Face Transformers |
| Containerization | Docker, Docker Compose |
| Orchestration | Kubernetes |

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- (Optional) kubectl for Kubernetes deployment

### Local Development
```bash
# Install dependencies for each service
pip install -r backend/requirements.txt
pip install -r risk_engine/requirements.txt
pip install -r optimizer/requirements.txt
pip install -r llm_service/requirements.txt
pip install -r frontend/requirements.txt

# Start backend
cd backend && uvicorn main:app --reload --port 8000

# Start frontend (separate terminal)
cd frontend && streamlit run app.py
```

### Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up --build
```

## Development Roadmap
- [x] Step 1: Project structure scaffolding
- [ ] Step 2: Risk engine (VaR, CVaR, Sharpe, Volatility)
- [ ] Step 3: Data fetching layer (yfinance integration)
- [ ] Step 4: Monte Carlo simulation engine
- [ ] Step 5: Portfolio optimizer (CVXPY)
- [ ] Step 6: LLM scenario service
- [ ] Step 7: FastAPI backend
- [ ] Step 8: Streamlit frontend
- [ ] Step 9: Docker containerization
- [ ] Step 10: Kubernetes deployment

## Contributing
Follow modular design principles. Each module must have its own requirements.txt and tests.

## License
MIT