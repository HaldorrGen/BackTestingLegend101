# Technical Documentation: Systematic Trading Strategy Backtesting Framework

## Executive Summary

This document provides comprehensive technical documentation for a quantitative trading strategy backtesting system. The framework implements a trend-following methodology based on moving average crossovers with dynamic risk management through volatility targeting and risk parity allocation.

### System Overview

The backtesting framework consists of modular components designed for institutional-grade quantitative analysis:

- Multi-asset portfolio management with configurable allocation schemes
- Technical indicator-based signal generation utilizing Simple Moving Averages (SMA) and Average True Range (ATR)
- Dynamic volatility targeting at the portfolio level
- Risk parity allocation methodology for optimal capital distribution
- Comprehensive performance attribution and risk metrics calculation

---

## Architecture

### System Components

The system is structured into four primary modules:

1. **Data Management Layer** (`data_download.py`, `data_feed.py`)
2. **Strategy Execution Engine** (`back_test.py`)
3. **Configuration Management** (`main.py`)
4. **Auxiliary Utilities** (`tickers/tickers.py`)

### Data Flow Architecture

```
Data Source (Yahoo Finance)
    ↓
Data Download Module → CSV Storage (tickers/*.csv)
    ↓
Data Feed Module → Indicator Calculation
    ↓
Backtesting Engine → Signal Generation
    ↓
Portfolio Construction → Performance Metrics
    ↓
Visualization & Reporting
```

---

## Module Specifications

### 1. Data Management

#### 1.1 Data Download Module (`data_download.py`)

**Purpose**: Acquisition and storage of historical market data from external data providers.

**Key Function**: `download_data(ticker, start_date, end_date)`

**Parameters**:
- `ticker` (str): Security identifier conforming to Yahoo Finance convention
- `start_date` (str): Beginning of historical data period (ISO 8601 format)
- `end_date` (str): End of historical data period (ISO 8601 format)

**Output**: CSV files stored in `tickers/` directory with the following schema:
- Date (Index): Trading date in YYYY-MM-DD format
- Open, High, Low, Close: OHLC price data
- Volume: Trading volume

**Error Handling**: Validates data availability and manages multi-index column structures from data provider.

#### 1.2 Data Feed Module (`data_feed.py`)

**Purpose**: Data preprocessing, period filtering, and technical indicator calculation.

**Key Function**: `load_data(ticker, start_date, end_date, ws, wl, watr, warm_up_days=0)`

**Parameters**:
- `ticker` (str): Security identifier
- `start_date` (str): Analysis period start date
- `end_date` (str): Analysis period end date
- `ws` (int): Short-term SMA window size (in trading days)
- `wl` (int): Long-term SMA window size (in trading days)
- `watr` (int): ATR calculation window (in trading days)
- `warm_up_days` (int, optional): Number of trading days before `start_date` for indicator initialization

**Technical Indicators Calculated**:

1. **Simple Moving Average (Short-term)**:
   ```
   SMA_short[t] = (1/ws) * Σ(Close[t-i]) for i = 0 to ws-1
   ```

2. **Simple Moving Average (Long-term)**:
   ```
   SMA_long[t] = (1/wl) * Σ(Close[t-i]) for i = 0 to wl-1
   ```

3. **Average True Range**:
   ```
   TR[t] = max(High[t] - Low[t], |High[t] - Close[t-1]|, |Low[t] - Close[t-1]|)
   ATR[t] = (1/watr) * Σ(TR[t-i]) for i = 0 to watr-1
   ```

**Output**: DataFrame with original OHLC data, calculated indicators, and `is_analysis_period` flag.

---

### 2. Strategy Execution Engine (`back_test.py`)

#### 2.1 Single Asset Backtesting

**Function**: `_run_single_backtest(...)`

**Signal Generation Logic**:

The strategy employs a trend-following methodology with the following entry conditions:

1. **Price Above Long-term SMA**: `Close[t] > SMA_long[t]`
2. **Short-term SMA Above Long-term SMA**: `SMA_short[t] > SMA_long[t]`
3. **Positive Short-term SMA Momentum**: `SMA_short[t] > SMA_short[t - slope_lookback]`

**Position States**:
- Signal = 1: Long position (full capital allocation)
- Signal = 0: Cash position (T-bills equivalent return)

**Return Calculation**:

Daily portfolio return incorporates:

1. **Market Return** (when Signal = 1):
   ```
   R_market[t] = (Close[t] / Close[t-1] - 1) * Signal[t-1]
   ```

2. **Cash Return** (when Signal = 0):
   ```
   R_cash[t] = ((1 + r_annual)^(1/252) - 1) * (1 - Signal[t-1])
   ```
   where `r_annual` is the annual risk-free rate (default: 3%)

3. **Transaction Costs**:
   ```
   Commission[t] = c * |Signal[t] - Signal[t-1]|
   ```
   where `c` is the commission rate (default: 0.2%)

4. **Total Daily Return**:
   ```
   R_total[t] = R_market[t] + R_cash[t] - Commission[t]
   ```

**Lookahead Bias Prevention**:
- All signals are shifted by 1 period before execution
- Entry/exit occurs at the opening of day t+1 after signal generation at close of day t

#### 2.2 Portfolio Backtesting

**Function**: `run_portfolio_backtest(...)`

**Allocation Methodologies**:

##### 2.2.1 Risk Parity Allocation

The system implements static risk parity allocation using inverse volatility weighting:

**Volatility Calculation**:
```
σ_i = std(returns_i) * √252
```
where returns are calculated over `volatility_lookback` period (default: 252 trading days)

**Weight Calculation**:
```
w_i = (1/σ_i) / Σ(1/σ_j) for all j
```

**Portfolio Volatility Estimate** (assuming zero correlation):
```
σ_portfolio = √(Σ(w_i * σ_i)²)
```

**Theoretical Foundation**: Risk parity seeks to equalize risk contribution from each asset rather than capital allocation. Under the assumption of zero correlation between assets, each position contributes equally to portfolio variance.

##### 2.2.2 Dynamic Volatility Targeting

The framework implements dynamic leverage adjustment to maintain target portfolio volatility:

**Rolling Volatility Calculation**:
```
σ_realized[t] = std(returns[t-63:t]) * √252
```
using a 63-day (approximately quarterly) rolling window

**Leverage Coefficient**:
```
k[t] = min(σ_target / σ_realized[t], leverage_cap)
```
where:
- `σ_target` is the desired annual volatility (default: 10%)
- `leverage_cap` is the maximum allowable leverage (default: 2.0x)

**Adjusted Returns**:
```
R_adjusted[t] = R_portfolio[t] * k[t-1]
```

**Implementation Note**: Leverage coefficient k[t-1] is applied to returns at time t to avoid lookahead bias.

---

### 3. Performance Metrics

#### 3.1 Return Metrics

**Total Return**:
```
R_total = (Equity_final / Equity_initial) - 1
```

**Compound Annual Growth Rate (CAGR)**:
```
CAGR = (Equity_final / Equity_initial)^(1/T) - 1
```
where T is the investment period in years

#### 3.2 Risk Metrics

**Maximum Drawdown**:
```
DD[t] = Equity[t] / max(Equity[0:t]) - 1
MaxDD = min(DD[t]) for all t
```

**Sharpe Ratio** (annualized):
```
SR = (μ_daily / σ_daily) * √252
```
where μ_daily and σ_daily are mean and standard deviation of daily returns

**Realized Volatility**:
```
σ_realized = σ_daily * √252
```

#### 3.3 Trading Activity Metrics

**Market Exposure**:
```
Exposure = (Days_in_position / Total_days) * 100%
```

**Number of Trades**:
Count of transitions from Signal = 0 to Signal = 1 (entry events only)

---

## Configuration Parameters

### Global Parameters (`main.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portfolio_tickers` | list[str] | ["SPY", "QQQ", "GLD", "USO"] | Universe of tradable securities |
| `start_date` | str | "1999-01-01" | Historical analysis start date |
| `end_date` | str | "2025-01-01" | Historical analysis end date |
| `ws` | int | 20 | Short-term SMA window (days) |
| `wl` | int | 200 | Long-term SMA window (days) |
| `watr` | int | 20 | ATR calculation window (days) |
| `slope_lookback` | int | 5 | SMA momentum period (days) |
| `goal_volatility` | float | 0.10 | Target portfolio volatility (10%) |
| `commission` | float | 0.002 | Transaction cost (0.2% per trade) |
| `numeraire` | str | 'usd' | Display currency ('usd' or 'gold') |

### Backtesting Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capital` | float | 100000 | Initial portfolio value ($) |
| `allocation` | str/dict | 'risk_parity' | Capital allocation method |
| `volatility_lookback` | int | 252 | Volatility calculation window (days) |
| `leverage_cap` | float | 2.0 | Maximum leverage multiplier |
| `warm_up_days` | int | 0 | Pre-analysis indicator warm-up period |
| `cash_rate` | float | 0.03 | Annual risk-free rate (3%) |

---

## Risk Management Framework

### Position Sizing

The system implements a two-tier risk management approach:

1. **Asset-Level Risk Control**: Binary position sizing (100% allocation when signal active, 0% in cash)
2. **Portfolio-Level Risk Control**: Dynamic leverage adjustment based on realized volatility

### Capital Preservation Mechanisms

1. **Defensive Cash Allocation**: When no trend signal is present, capital is allocated to cash-equivalent instruments earning risk-free rate
2. **Volatility Targeting**: Reduces exposure during high-volatility regimes
3. **Transaction Cost Integration**: Explicit modeling of market impact through commission charges

---

## Validation and Testing

### Historical Crisis Period Analysis

The framework includes predefined stress test periods (`crisis_times_dates` in `main.py`):

- Dot-com Bubble (2000-2002)
- Global Financial Crisis (2007-2009)
- COVID-19 Market Disruption (2020)
- 2022 Bear Market

### Bias Prevention

1. **Lookahead Bias**: Signal execution delayed by 1 period
2. **Survivorship Bias**: System requires explicit data for analysis period
3. **Transaction Cost Reality**: Explicit commission modeling on all position changes

---

## Output and Reporting

### Console Output

The system generates formatted performance reports including:
- Individual asset performance metrics
- Portfolio-level aggregated statistics
- Risk parity weight allocation
- Volatility targeting effectiveness

### File Output

**Metrics Files**: `metrics_{counter}_{timestamp}.txt`
- Comprehensive performance attribution
- Individual ticker breakdown
- Risk-adjusted return statistics

**Consolidated Report**: `all_crisis_results.txt`
- Merged results from all test scenarios
- Comparative analysis across time periods

### Visualization

The framework generates matplotlib-based visualizations:

1. **Price and Signal Chart**:
   - OHLC price data
   - Short and long-term SMAs
   - Entry/exit signals (vertical lines)
   - Equity curve overlay

2. **Portfolio Performance Chart**:
   - Strategy equity curve vs. buy-and-hold benchmark
   - Individual asset contribution
   - Optional numeraire adjustment (USD or gold-denominated)

---

## System Requirements

### Dependencies

- Python 3.7+
- pandas: Data manipulation and time series analysis
- numpy: Numerical computations
- matplotlib: Visualization
- yfinance: Historical market data acquisition
- math: Statistical calculations

### Data Requirements

- Minimum: 400 trading days of historical data (for indicator warm-up with wl=200)
- Recommended: Full historical period from 1999 to present for robust backtesting

---

## Limitations and Assumptions

### Model Assumptions

1. **Liquidity Assumption**: Assumes unlimited liquidity at closing prices
2. **Slippage**: Not explicitly modeled (captured partially through commission parameter)
3. **Market Impact**: Assumes trades do not affect market prices
4. **Dividend Treatment**: Returns calculated on price appreciation only
5. **Correlation Structure**: Risk parity calculations assume zero correlation between assets
6. **Leverage Availability**: Assumes frictionless access to leverage up to `leverage_cap`

### Known Limitations

1. **Static Risk Parity**: Weights calculated once at period start, not dynamically rebalanced
2. **Volatility Targeting**: Implemented at portfolio level but not per-asset
3. **Signal Complexity**: Binary entry logic may miss nuanced market conditions
4. **Benchmark Selection**: Buy-and-hold comparison may not represent optimal passive strategy

---

## Future Enhancement Considerations

### Potential Extensions

1. **Dynamic Rebalancing**: Periodic weight recalculation based on realized volatilities
2. **Transaction Cost Optimization**: Intelligent trade execution to minimize costs
3. **Multi-Factor Signals**: Integration of momentum, value, and quality factors
4. **Risk Model Enhancement**: Full covariance matrix estimation for risk parity
5. **Machine Learning Integration**: Adaptive parameter optimization
6. **Options Overlay**: Tail risk hedging through derivative strategies

---

## Usage Guidelines

### Basic Workflow

1. **Data Acquisition**: Execute `download_portfolio_data.py` to populate historical data
2. **Parameter Configuration**: Modify `main.py` configuration section
3. **Backtest Execution**: Run `main.py` to perform analysis
4. **Results Review**: Examine console output and generated metric files
5. **Visualization Analysis**: Review matplotlib charts for qualitative assessment

### Customization Points

- **Universe Selection**: Modify `portfolio_tickers` list in `main.py`
- **Signal Logic**: Edit entry conditions in `back_test.py:_run_single_backtest()`
- **Allocation Method**: Implement custom allocation dictionary or modify risk parity logic
- **Performance Metrics**: Add custom calculations in metrics section

---

## Technical Notes

### Performance Considerations

- Vectorized operations using pandas for computational efficiency
- Minimal iterative loops in return calculation
- CSV-based data storage for rapid prototyping

### Code Quality

- Modular architecture for maintainability
- Comprehensive inline documentation in Russian (original development language)
- Defensive programming with error handling for missing data

---

## Appendix: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| t | Time index (trading days) |
| Close[t] | Closing price at time t |
| SMA_short[t] | Short-term moving average at time t |
| SMA_long[t] | Long-term moving average at time t |
| ATR[t] | Average True Range at time t |
| Signal[t] | Position signal (1 = long, 0 = cash) |
| R[t] | Return at time t |
| σ | Volatility (standard deviation) |
| w_i | Weight of asset i |
| k[t] | Leverage coefficient at time t |
| T | Time period in years |

---

## Document Version

**Version**: 1.0  
**Date**: January 2026  
**Author**: Institutional Quantitative Analysis Team  
**Classification**: Internal Technical Documentation

---

## Disclaimer

This documentation describes a backtesting framework for research purposes. Past performance does not guarantee future results. The system contains simplifying assumptions that may not reflect actual market conditions. Any implementation for live trading requires extensive additional validation, risk management, and compliance review.
