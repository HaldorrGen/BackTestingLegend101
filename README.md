# Systematic Trading Strategy Backtesting Framework

## Overview

A quantitative trading system implementing trend-following strategies with dynamic risk management. The framework provides institutional-grade backtesting capabilities for multi-asset portfolios using technical indicators and volatility targeting.

## Key Features

- **Trend-Following Strategy**: SMA crossover with momentum confirmation
- **Risk Parity Allocation**: Inverse volatility weighting for optimal capital distribution
- **Dynamic Volatility Targeting**: Portfolio-level leverage adjustment to maintain target risk
- **Comprehensive Risk Metrics**: Sharpe ratio, maximum drawdown, CAGR calculations
- **Transaction Cost Modeling**: Explicit commission charges on position changes
- **Historical Stress Testing**: Built-in crisis period analysis (2000-2025)

## Project Structure

```
starts/
│
├── main.py                     # Main execution script and configuration
├── back_test.py                # Core backtesting engine
├── data_feed.py                # Data loading and indicator calculation
├── data_download.py            # Historical data acquisition
├── download_portfolio_data.py  # Batch data download utility
│
├── tickers/                    # Market data storage
│   ├── *.csv                   # Individual security data files
│   ├── tickers.py              # NASDAQ ticker list utility
│   └── universe.txt            # Complete ticker universe
│
├── DOCUMENTATION.md            # Comprehensive technical documentation
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib yfinance
```

### 2. Download Historical Data

```bash
python download_portfolio_data.py
```

### 3. Run Backtest

```bash
python main.py
```

## Configuration

Edit `main.py` to customize:

```python
# Portfolio composition
portfolio_tickers = ["SPY", "QQQ", "GLD", "USO"]

# Technical parameters
ws = 20              # Short SMA window
wl = 200             # Long SMA window
watr = 20            # ATR window
slope_lookback = 5   # Momentum period

# Risk management
goal_volatility = 0.10    # Target 10% annual volatility
commission = 0.002        # 0.2% transaction cost
```

## Strategy Logic

### Entry Conditions (ALL must be satisfied)

1. Price > Long-term SMA
2. Short-term SMA > Long-term SMA  
3. Short-term SMA exhibits positive momentum over `slope_lookback` period

### Position Management

- **Long Position**: 100% capital allocated when trend signal active
- **Cash Position**: Capital in T-bills (3% annual return) when no signal

### Risk Controls

- Volatility targeting with dynamic leverage (capped at 2.0x)
- Risk parity weighting across portfolio constituents
- Transaction cost integration on all position changes

## Output

### Performance Metrics

- Total Return & CAGR
- Maximum Drawdown
- Sharpe Ratio
- Market Exposure %
- Number of Trades

### Files Generated

- `metrics_*.txt`: Individual backtest results
- `all_crisis_results.txt`: Consolidated performance report

### Visualizations

- Price chart with SMA overlays and entry/exit signals
- Equity curve vs. buy-and-hold benchmark
- Individual asset performance breakdown

## Technical Documentation

For detailed technical specifications, mathematical formulations, and architecture details, refer to [DOCUMENTATION.md](DOCUMENTATION.md).

## Data Sources

- **Market Data**: Yahoo Finance via `yfinance` library
- **Coverage**: 1999-present (varies by security)
- **Update Frequency**: Manual execution of download scripts

## Default Test Periods

```python
crisis_times_dates = [
    ("1999-12-01", "2025-12-01"),  # Full period
    ("2000-01-01", "2002-12-01"),  # Dot-com crash
    ("2007-01-01", "2009-12-01"),  # Financial crisis
    ("2020-01-01", "2020-12-01"),  # COVID-19
    ("2022-01-01", "2022-12-01")   # 2022 bear market
]
```

## Limitations

- Assumes unlimited liquidity at closing prices
- Static risk parity weights (calculated at period start)
- Binary position sizing (100% or 0%)
- No explicit slippage modeling beyond commission parameter

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- yfinance

## License

This project is provided for research and educational purposes.

## Disclaimer

This backtesting framework is intended for research purposes only. Past performance does not guarantee future results. The system contains simplifying assumptions that may not reflect actual market conditions. Consult with qualified financial professionals before implementing any trading strategies.
