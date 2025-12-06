# 🤖 Risk Regulated PPO Agent for Regime Adaptive SPY Trading

A Reinforcement Learning trading agent that learns to trade the S&P 500 (SPY) using Proximal Policy Optimization (PPO). The agent is trained to protect capital during bear markets while maintaining positive returns across different market conditions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📊 Performance Summary

| Period | Agent Return | Buy & Hold | Outperformance | Max Drawdown |
|--------|--------------|------------|----------------|--------------|
| **2022 Validation** (Bear Market) | +5.63% | -18.65% | +24.28% | 6.41% |
| **2023-2024 Test** (Bull Market) | +10.06% | +58.82% | -48.76% | 5.06% |

### Key Metrics
- **Sharpe Ratio**: 0.604 (validation), 0.972 (test)
- **Sortino Ratio**: 0.817 (validation), 1.241 (test)
- **Trading Activity**: 35 trades (validation), 82 trades (test)
- **Risk Control**: Maximum drawdown stayed below 7% in both periods

## 🎯 What Makes This Special?

This agent demonstrates **defensive trading**—it sacrifices bull market gains for downside protection:
- ✅ **Protected capital** during 2022 bear market (+5.6% vs -18.7% B&H)
- ✅ **Controlled risk** with max drawdowns under 7%
- ✅ **Consistent performance** with strong risk-adjusted returns
- ⚠️ **Conservative** approach underperforms in strong bull markets

## 🏗️ Architecture

### Reinforcement Learning Setup
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: Stable-Baselines3 + Gymnasium
- **Training Data**: 2014-2021 (1,927 trading days)
- **Validation**: 2022 (251 trading days)
- **Test**: 2023-2024 (501 trading days)

### Market Features (8 total)
The agent observes these features to make decisions:

1. **Efficiency Ratio (10-day)** - Short-term trend strength
2. **Efficiency Ratio (30-day)** - Long-term trend strength  
3. **Hurst Exponent** - Trending vs. mean-reverting detection
4. **Shannon Entropy** - Market uncertainty measurement
5. **Risk-Adjusted Momentum** - Momentum relative to volatility
6. **VIX** - Fear/volatility index
7. **Exposure** - Current position size
8. **Cash Ratio** - Available capital

### Action Space
- **0**: HOLD (do nothing)
- **1**: BUY (enter long position)
- **2**: SELL (exit position)

### Risk Management
The environment enforces these rules automatically:
- **Stop Loss**: Sell if position drops 5%
- **Trailing Stop**: Sell if price drops 7% from peak
- **Circuit Breaker**: Stop trading if portfolio drops 15%
- **Position Limit**: Maximum 50% of portfolio per trade
- **Transaction Cost**: 0.01% per trade

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
GPU recommended (training takes ~1 hour on A100, longer on CPU)
```

### Installation
```bash
pip install stable-baselines3[extra] gymnasium optuna yfinance
```

### Run the Notebook
1. Open `SPY_PPO_Trading_Agent_CLEAN.ipynb` in Google Colab or Jupyter
2. Select runtime mode:
   - `fast`: ~1 hour (10 trials, good for testing)
   - `medium`: ~3 hours (20 trials, balanced)
   - `production`: ~10 hours (50 trials, best results)
3. Run all cells
4. Results are automatically saved to Google Drive or local directory

### Configuration Options
```python
# Quick configuration at the top of notebook
MODE = 'fast'  # Change to 'medium' or 'production'

# Data periods
TRAIN_START = '2014-01-01'
TRAIN_END = '2021-12-31'
VAL_START = '2022-01-01'
VAL_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2024-12-31'

# Risk settings
STOP_LOSS_PCT = 0.05
TRAILING_STOP_PCT = 0.07
MAX_DRAWDOWN_PCT = 0.15
```

## 📁 Project Structure

```
spy-ppo-trading-agent/
├── SPY_PPO_Trading_Agent_CLEAN.ipynb  # Main notebook
├── README.md                           # This file
└── outputs/                            # Generated files
    ├── ppo_spy_final.zip              # Trained model
    ├── best_params.json               # Optimal hyperparameters
    ├── final_results.json             # Performance metrics
    ├── data_split.png                 # Data visualization
    └── equity_curves.png              # Performance charts
```

## 🔬 How It Works

### 1. Data Preparation
- Downloads SPY and VIX data from Yahoo Finance
- Engineers 6 advanced market features
- Splits into train/validation/test sets

### 2. Environment Creation
- Custom Gymnasium environment simulates trading
- Implements realistic transaction costs
- Enforces risk management rules automatically

### 3. Hyperparameter Optimization
- Optuna searches for optimal hyperparameters
- Optimizes on 2022 bear market (validation set)
- Maximizes Sortino Ratio with constraints

### 4. Training
- PPO agent learns from 2014-2021 data
- Neural network: 256→128→64 architecture (default)
- ~200k timesteps for final model

### 5. Evaluation
- Tests on unseen 2023-2024 data
- Compares against buy-and-hold baseline
- Calculates comprehensive performance metrics

## 📈 Understanding the Results

### Why Underperformance in Bull Markets?
The agent was optimized on the 2022 bear market, teaching it to be **defensive**:
- Quickly exits positions when risk increases
- Maintains larger cash positions for safety
- Prioritizes capital preservation over maximum gains

This is a **feature, not a bug**—ideal for risk-averse investors.

### Risk-Adjusted Performance
Despite lower absolute returns in bull markets, the agent shows strong risk-adjusted performance:
- Higher Sharpe and Sortino ratios than many active strategies
- Significantly lower drawdowns (5-6% vs 15-20% for B&H)
- More consistent returns across market conditions

## 🛠️ Customization Ideas

### Modify Trading Strategy
```python
# More aggressive position sizing
MAX_POSITION_PCT = 0.80  # From 0.50

# Tighter risk controls
STOP_LOSS_PCT = 0.03  # From 0.05

# Different optimization target
# In objective function, change:
return metrics['sharpe']  # Instead of sortino
```



## 📊 Visualizations

The notebook generates:
- **Data Split Chart**: Shows train/validation/test periods on price chart
- **Equity Curves**: Agent vs. Buy & Hold performance over time
- **Trade Analysis**: Entry/exit points and profitability

## ⚠️ Disclaimer

**This project is for educational purposes only. Not financial advice.**

- Past performance does not guarantee future results
- Trading involves significant risk of loss
- The model may not perform well in all market conditions
- Always do your own research and consult financial advisors



## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- **Stable-Baselines3**: Clean RL implementations
- **Optuna**: Hyperparameter optimization framework
- **yfinance**: Free market data access
- **Gymnasium**: RL environment standard

## 📚 References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)


