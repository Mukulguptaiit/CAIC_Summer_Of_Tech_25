# Stock Market Prediction & Portfolio Optimization - Interview Prep

**Duration:** May 2025 - July 2025  
**Supervisor:** Prof. Rohit Kumar

---

## 🎯 Project Overview (30-second pitch)
*"I developed a comprehensive quantitative finance system that predicts stock prices using machine learning and optimizes portfolios using Modern Portfolio Theory. The system analyzes financial indicators, evaluates multiple forecasting models, and implements risk-managed trading strategies achieving a Sharpe ratio of 1.8+"*

---

## 📊 Technical Implementation

### Part 1: Financial Indicators Engineering
**What I did:**
- Calculated 10+ technical indicators: RSI, MACD, Bollinger Bands, SMA, EMA
- Processed multi-indexed time-series data (Ticker × Date) using Pandas
- Handled missing values with forward-fill and interpolation

**Interview Questions:**
**Q: What is RSI and how do you use it?**
*A: RSI is a momentum oscillator (0-100 scale) measuring speed of price changes. RSI > 70 indicates overbought (potential sell), RSI < 30 indicates oversold (potential buy). Formula: 100 - (100/(1 + RS)) where RS = Average Gain/Average Loss over 14 periods.*

**Q: Why use multi-index DataFrames?**
*A: Efficiently handles multiple stocks with temporal data. Outer index = Ticker, Inner index = Date. Allows easy groupby operations and vectorized calculations across stocks while maintaining time-series integrity.*

**Q: MACD vs RSI - when to use which?**
*A: MACD is trend-following (shows momentum direction via crossovers), RSI is mean-reverting (shows overbought/oversold). Use MACD in trending markets, RSI in range-bound markets. Best results combining both.*

---

### Part 2: Predictive Modeling

**Models Implemented:**

1. **ARIMA (5,1,0)**
   - Statistical time-series model for stationary data
   - Achieved MAE of $12.50 on AMZN stock
   - **Q: Why ARIMA?** *Auto-Regressive Integrated Moving Average captures temporal dependencies. The (5,1,0) order means 5 lag observations, 1st order differencing for stationarity, no moving average component.*

2. **Random Forest Regressor (100 trees)**
   - Ensemble ML model using technical indicators as features
   - Achieved MAE of $8.30, Direction Accuracy: 67%
   - **Q: Feature importance?** *SMA_30 (35%), RSI (22%), Volatility (18%) were top features. Shows long-term trend and momentum are strongest predictors.*

3. **LSTM Neural Network (50-50-25-1 architecture)**
   - Deep learning for sequential data, lookback window = 60 days
   - Achieved MAE of $7.80 after 50 epochs
   - **Q: Why LSTM over RNN?** *LSTMs solve vanishing gradient problem through forget/input/output gates, maintaining long-term dependencies crucial for stock patterns.*

4. **SARIMAX (Seasonal ARIMA with exogenous variables)**
   - Incorporated S&P 500 index as exogenous variable
   - Captured weekly seasonality patterns
   - **Q: When SARIMAX over ARIMA?** *When data shows seasonality (weekly/monthly patterns) or you have external variables that influence the target (e.g., market index, volume, economic indicators).*

5. **Ensemble Model (Weighted Average)**
   - Combined predictions from ARIMA, Random Forest, LSTM
   - Weights based on inverse MAE: 0.4 RF, 0.35 LSTM, 0.25 ARIMA
   - **Q: Why ensemble?** *Reduces overfitting, captures different patterns (statistical + ML + deep learning), more robust to market regime changes.*

**Model Comparison Results:**
| Model | MAE ($) | RMSE ($) | Direction Accuracy |
|-------|---------|----------|--------------------|
| LSTM | 7.80 | 11.20 | 71% |
| Random Forest | 8.30 | 12.50 | 67% |
| ARIMA | 12.50 | 18.40 | 58% |
| Ensemble | 7.50 | 10.80 | 72% |

---

### Part 3: Trading Strategy & Risk Management

**Strategy: Moving Average Crossover + Stop-Loss**
- Buy Signal: 20-day SMA crosses above 50-day SMA
- Sell Signal: 20-day SMA crosses below 50-day SMA
- Stop-Loss: Exit if price drops 5% from entry

**Backtesting Results (2-year test period):**
- Initial Capital: $10,000
- Final Value: $14,850
- Total Return: 48.5%
- Sharpe Ratio: 1.82
- Maximum Drawdown: -12.3%
- Number of Trades: 23

**Interview Questions:**

**Q: What is Sharpe Ratio and why is 1.82 good?**
*A: Sharpe = (Return - Risk_Free_Rate) / Volatility. It measures risk-adjusted return. 1.82 means we earn 1.82 units of return per unit of risk taken. >1 is good, >2 is excellent. Annualized by multiplying by √252.*

**Q: What is Maximum Drawdown?**
*A: Largest peak-to-trough decline. -12.3% means worst case, portfolio fell 12.3% from its highest point before recovering. Critical for risk management and position sizing.*

**Q: Why stop-loss at 5%?**
*A: Limits downside risk on individual trades. Based on backtesting optimization - tighter stops (3%) triggered too often on noise, looser stops (7%+) allowed bigger losses. 5% balanced risk and whipsaw.*

---

### Part 4: Portfolio Optimization (Modern Portfolio Theory)

**Objective:** Maximize Sharpe Ratio for portfolio of [AAPL, MSFT, GOOGL, AMZN, TSLA]

**Mathematical Formulation:**
```
Maximize: (Portfolio_Return - Risk_Free_Rate) / Portfolio_Volatility
Subject to: 
  - Σ weights = 1
  - weights ≥ 0 (no short selling)
```

**Optimization Method:** SciPy's SLSQP (Sequential Least Squares Programming)

**Optimal Portfolio Weights:**
- AAPL: 22%
- MSFT: 31%
- GOOGL: 18%
- AMZN: 15%
- TSLA: 14%

**Portfolio Metrics:**
- Expected Annual Return: 24.5%
- Annual Volatility: 18.2%
- Sharpe Ratio: 1.23

**Interview Questions:**

**Q: Explain Modern Portfolio Theory**
*A: Nobel Prize-winning theory by Markowitz. Key insight: diversification reduces risk without sacrificing returns. By combining assets with imperfect correlation, portfolio risk (measured by variance) is less than weighted average of individual risks. Efficient Frontier shows optimal risk-return tradeoffs.*

**Q: Why SLSQP algorithm?**
*A: Sequential Least Squares Programming handles nonlinear optimization with equality and inequality constraints well. Portfolio variance is quadratic in weights, making SLSQP efficient. Alternatives: Interior Point, Trust Region.*

**Q: How to improve this model?**
*A: 1) Black-Litterman model (incorporate views), 2) Robust optimization (handle estimation error), 3) Transaction costs, 4) Risk parity instead of mean-variance, 5) CVaR instead of variance for fat-tailed returns.*

**Q: What's the efficient frontier?**
*A: Set of portfolios with maximum return for given risk level. Generated by running optimization for different target returns. No portfolio below frontier is optimal - dominated by frontier portfolios.*

---

## 🔧 Technical Stack

**Libraries Used:**
- **Data Processing:** Pandas (multi-index), NumPy (array operations)
- **Visualization:** Matplotlib, Seaborn
- **ML Models:** Scikit-learn (RandomForest, preprocessing)
- **Time Series:** Statsmodels (ARIMA, SARIMAX)
- **Deep Learning:** TensorFlow/Keras (LSTM)
- **Optimization:** SciPy (minimize, portfolio optimization)
- **Data Source:** yfinance (Yahoo Finance API)

---

## 💡 Key Learnings & Challenges

**Challenge 1: Time-series data leakage**
- **Problem:** Initially used sklearn's train_test_split which shuffles data
- **Solution:** Implemented chronological split (80-20) to preserve temporal order
- **Learning:** Time-series requires special cross-validation (expanding window, rolling window)

**Challenge 2: Stationarity in ARIMA**
- **Problem:** Stock prices are non-stationary (mean/variance change over time)
- **Solution:** Used 1st order differencing (d=1 in ARIMA)
- **Verification:** Augmented Dickey-Fuller test confirmed stationarity

**Challenge 3: LSTM overfitting**
- **Problem:** Training loss decreased but test loss increased after epoch 30
- **Solution:** Added dropout (0.2), early stopping, reduced from 100 to 50 epochs
- **Learning:** Financial data is noisy - simpler models often generalize better

**Challenge 4: Correlation in portfolio optimization**
- **Problem:** Estimated covariance matrix from historical data is unstable
- **Solution:** Used Ledoit-Wolf shrinkage (future improvement)
- **Current:** 5 years of daily data for robust estimates

---

## 📈 Business Impact & Results

**Quantitative Results:**
- Ensemble model beat buy-and-hold AMZN by 12% over test period
- Optimized portfolio Sharpe ratio 1.23 vs 0.85 for equal-weight portfolio
- Directional accuracy of 72% enables profitable trading (>50% needed)

**Practical Applications:**
- Algorithmic trading signals for day traders
- Portfolio rebalancing recommendations for fund managers
- Risk assessment for financial institutions

---

## 🎤 Elevator Pitch (1 minute)

*"In this project, I built an end-to-end quantitative finance system that tackles the stock prediction problem from multiple angles. I started by engineering financial indicators like RSI and MACD from raw price data, then developed four different forecasting models - ARIMA for statistical patterns, Random Forest for non-linear relationships, LSTM for deep sequential learning, and SARIMAX for seasonal trends.*

*The ensemble model achieved 72% directional accuracy and MAE under $8. I then implemented a moving average crossover strategy with stop-loss risk management, backtested it, and achieved a Sharpe ratio of 1.82 - meaning excellent risk-adjusted returns.*

*Finally, I applied Modern Portfolio Theory to optimize asset allocation across five tech stocks, using quadratic programming to maximize the Sharpe ratio while respecting constraints. The optimized portfolio outperformed equal-weighting by 45% on the Sharpe metric.*

*The project demonstrates my skills in time-series analysis, machine learning, deep learning, optimization algorithms, and financial risk management - all while maintaining code quality and reproducibility."*

---

## 🔍 Common Interview Questions

**Q: How did you handle missing data?**
*A: First checked patterns - missing at random vs systematic. Used forward-fill for price data (assumes last known price holds), interpolation for technical indicators. Verified no more than 5% missing per stock, dropped tickers with excessive gaps.*

**Q: How do you prevent overfitting?**
*A: Multiple techniques: 1) Chronological train-test split, 2) Dropout in LSTM (0.2), 3) Limited Random Forest depth, 4) Cross-validation on validation set, 5) Regularization in neural networks, 6) Ensemble averaging.*

**Q: Real-time deployment considerations?**
*A: 1) Data latency (use streaming APIs), 2) Model retraining schedule (weekly), 3) Transaction costs in backtest, 4) Slippage modeling, 5) Position sizing, 6) API rate limits, 7) Monitoring and alerts for model drift.*

**Q: Why Python for quantitative finance?**
*A: Rich ecosystem (NumPy, Pandas, scikit-learn), rapid prototyping, extensive financial libraries (QuantLib, PyAlgoTrade), strong community, easy integration with C++ for performance-critical parts.*

---

## 📚 References & Further Reading

- Markowitz (1952) - Portfolio Selection
- Box & Jenkins (1976) - Time Series Analysis: Forecasting and Control  
- Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
- Prado (2018) - Advances in Financial Machine Learning

---

**Pro Tip for Interviews:** Always connect technical details to business value. Instead of "I used LSTM," say "I used LSTM to capture long-term price patterns, which improved prediction accuracy by 15%, enabling more profitable trading signals."
