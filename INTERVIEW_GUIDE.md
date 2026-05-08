# Interview Preparation Guide - CAIC Summer of Tech Projects

## 📚 Overview
This guide highlights key concepts, algorithms, and interview questions covered in your projects.

---

## 1. Stock Market Prediction & Portfolio Optimization

### Key Algorithms Implemented

#### Technical Indicators
```python
# RSI (Relative Strength Index)
- Momentum oscillator (0-100 scale)
- Formula: RSI = 100 - (100 / (1 + RS))
- RS = Average Gain / Average Loss
- Interview Tip: RSI > 70 = overbought, RSI < 30 = oversold

# MACD (Moving Average Convergence Divergence)
- Trend-following momentum indicator
- Components: MACD Line, Signal Line, Histogram
- Buy signal: MACD crosses above signal line
- Interview Question: "Why use EMA instead of SMA for MACD?"
  Answer: EMA is more responsive to recent price changes

# Bollinger Bands
- Volatility indicator with 3 bands
- Middle: 20-day SMA
- Upper/Lower: SMA ± (2 × std dev)
- Interpretation: Bands widen = high volatility, narrow = low volatility
```

#### Machine Learning Models

**Random Forest for Stock Prediction**
```
Time Complexity: O(n_trees × n_samples × log(n_samples) × n_features)
Space Complexity: O(n_trees × tree_size)

Interview Questions:
Q: Why Random Forest over Linear Regression?
A: Handles non-linear relationships, resistant to overfitting through ensemble averaging

Q: How to prevent overfitting?
A: Increase n_estimators, limit max_depth, use min_samples_split

Q: Feature importance - how is it calculated?
A: Based on Gini impurity decrease or mean decrease in accuracy
```

**ARIMA Time Series Model**
```
Model: ARIMA(p, d, q)
- p: autoregressive order
- d: differencing order (stationarity)
- q: moving average order

Interview Deep Dive:
Q: What is stationarity and why does it matter?
A: Mean, variance, autocorrelation constant over time. ARIMA requires it.

Q: How to check stationarity?
A: Augmented Dickey-Fuller test, plot ACF/PACF

Q: When to use SARIMAX over ARIMA?
A: When data has seasonality (monthly/quarterly patterns) or need exogenous variables
```

**LSTM Neural Networks**
```python
Architecture:
- Input: Sequence of past N days
- Hidden: LSTM layers with dropout
- Output: Next day's prediction

Interview Concepts:
- Vanishing gradient problem (why LSTM > RNN)
- Forget gate, input gate, output gate
- Why MinMaxScaler for financial data?
  Answer: Keeps values in [0,1] for sigmoid activation

Complexity:
- Training: O(epochs × samples × sequence_length × hidden_units²)
- Inference: O(sequence_length × hidden_units²)
```

#### Portfolio Optimization

**Modern Portfolio Theory (Markowitz)**
```
Objective: Maximize Sharpe Ratio or Minimize Variance
Constraints:
1. Σ wi = 1 (weights sum to 100%)
2. wi ≥ 0 (no short selling)
3. Optional: wi ≤ max_allocation

Mathematical Formulation:
Portfolio Return: E[Rp] = Σ(wi × E[Ri])
Portfolio Variance: σp² = w^T × Cov × w
Sharpe Ratio: (E[Rp] - Rf) / σp

Interview Questions:
Q: What is the efficient frontier?
A: Set of optimal portfolios offering highest return for given risk level

Q: Why use quadratic programming?
A: Portfolio variance is quadratic function of weights

Q: Limitations of Markowitz model?
A: Assumes normal returns, static correlations, no transaction costs
```

#### Risk Metrics

**Sharpe Ratio**
```
Formula: (Return - Risk_Free_Rate) / Volatility × √252

Interpretation:
< 1.0: Poor risk-adjusted returns
1.0-2.0: Good
> 2.0: Excellent

Interview Note: Penalizes upside and downside volatility equally.
Sortino ratio uses only downside deviation (better for asymmetric risk).
```

**Maximum Drawdown**
```
Definition: Largest peak-to-trough decline

Calculation:
1. Track running maximum (cummax)
2. Calculate drawdown at each point
3. Take minimum (most negative) value

Why it matters:
- Worst-case loss scenario
- Psychological impact on investors
- Risk management tool
```

---

## 2. Supply Chain Optimization

### Computational Geometry

**Convex Hull (Graham Scan)**
```
Time Complexity: O(n log n)
Space Complexity: O(n)

Applications:
- Define minimum bounding region for delivery zones
- Identify outlier locations
- Facility placement optimization

Interview Algorithm Walkthrough:
1. Find anchor point (lowest y, leftmost x if tie)
2. Sort by polar angle
3. Process points maintaining left turns only
```

**Polygon Intersection**
```
Use Case: Overlapping delivery zones
Algorithm: Sutherland-Hodgman (for convex polygons)
Time Complexity: O(n × m) where n, m = polygon vertices

Interview Tip: Explain Shapely library does this efficiently
using sweep line algorithms for complex polygons.
```

### Graph Algorithms

**Dijkstra's Shortest Path**
```
Time Complexity: O((V + E) log V) with min-heap
Space Complexity: O(V)

Application: Find fastest delivery route

Interview Questions:
Q: Why not use BFS?
A: BFS works for unweighted graphs. Dijkstra handles weighted edges.

Q: Negative weights?
A: Use Bellman-Ford (O(VE)) or detect negative cycles

Q: Bidirectional search optimization?
A: Search from both start and end simultaneously, reduces search space
```

**Traveling Salesman Problem (TSP)**
```
Complexity: NP-Hard, O(n!) exact solution
Heuristics:
1. Nearest Neighbor: O(n²) - greedy approach
2. 2-opt improvement: O(n²) per iteration
3. Genetic algorithms for large instances

Interview Discussion:
Q: Why is TSP NP-hard?
A: No known polynomial-time algorithm, grows factorially

Q: Real-world approximations?
A: Christofides algorithm (1.5-approximation for metric TSP)
```

### Linear Programming (PuLP)

**Inventory Optimization**
```python
Minimize: Σ(holding_cost × inventory + stockout_cost × shortfall)
Subject to:
- inventory + stockout = demand
- Σ(holding_cost × inventory) ≤ budget

Interview Concepts:
- Decision variables vs parameters
- Objective function formulation
- Constraint types (equality, inequality)
- Simplex method basics
```

**EOQ (Economic Order Quantity)**
```
Formula: EOQ = √(2 × Annual_Demand × Order_Cost / Holding_Cost)

Assumptions:
1. Constant demand rate
2. Fixed ordering cost
3. Known holding cost
4. Instantaneous replenishment

Interview Extension:
Q: What if demand is uncertain?
A: Use (Q, r) model with safety stock: r = μL + z × σL
```

### Demand Forecasting

**ARIMA vs Prophet vs Exponential Smoothing**
```
ARIMA:
- Best for: Stationary data, short-term forecasts
- Limitations: Requires stationarity, manual parameter tuning

Prophet (Facebook):
- Best for: Seasonal data, missing values, outliers
- Components: Trend + Seasonality + Holidays
- Automatic parameter selection

Exponential Smoothing:
- Best for: Short-term forecasts, weighted recent data
- Types: Simple, Double (Holt), Triple (Holt-Winters)

Interview Question:
"When would you choose ARIMA over Prophet?"
Answer: ARIMA for well-behaved stationary series with clear ACF/PACF patterns.
Prophet for business time series with strong seasonality and holidays.
```

---

## 3. Urban Mobility Planning

### Multi-Objective Optimization

**Problem Formulation**
```
Objectives:
1. Minimize emissions: Σ(emission_factor × vehicle_km)
2. Maximize equity: Prioritize underserved communities
3. Minimize travel time: Optimize routes and frequencies

Constraints:
- Budget limitations
- Infrastructure capacity
- Political/social feasibility

Solution Methods:
1. Weighted sum: Σ(wi × fi(x))
2. Pareto frontier: Set of non-dominated solutions
3. ε-constraint: Optimize one objective, constrain others
```

### Emission Calculations

**Bottom-Up Approach**
```python
Emissions = Σ(Vehicle_km × Emission_Factor × Modal_Share)

Emission Factors (example):
- Private car: 120 g CO2/km
- Bus: 30 g CO2/passenger-km
- Metro: 20 g CO2/passenger-km
- Bicycle: 0 g CO2/km

Interview Discussion:
- Well-to-wheel vs tank-to-wheel emissions
- Load factors and occupancy rates
- Electric vehicle grid carbon intensity
```

### Transit Route Optimization

**Bus Route Design**
```
Formulation (Set Cover variant):
Maximize: Σ(population_served × need_weight)
Subject to:
- Each zone served by ≤1 route
- Route capacity constraints
- Budget limitations
- Connectivity requirements (min 3 zones per route)

Interview Concept: This is NP-hard (similar to Set Cover)
Approach: Linear Programming relaxation + rounding
```

### Accessibility Metrics

**15-Minute City Concept**
```
Metric: Percentage of population within 15-min walk/cycle to transit

Implementation:
1. Generate isochrones (reachable areas)
2. Overlay with population density
3. Calculate coverage statistics

Interview Note: Explain difference between 
- Euclidean distance (straight line)
- Manhattan distance (grid)
- Network distance (actual paths)
```

---

## 🎯 Common Interview Questions Across Projects

### Data Science Fundamentals

**Train-Test Split for Time Series**
```
Q: Why not use sklearn's train_test_split for time series?
A: It shuffles data, breaking temporal dependencies. Must use chronological split.

Q: What's a good train-test ratio?
A: Typically 80-20 or 70-30, but depends on:
   - Data frequency (daily vs monthly)
   - Forecast horizon
   - Stationarity of the series
```

**Handling Missing Data**
```
Methods:
1. Forward fill: Use last known value
2. Backward fill: Use next known value
3. Interpolation: Linear, polynomial, spline
4. Mean/median imputation
5. Model-based: KNN, MICE

Interview Decision Tree:
- Missing at random? → Interpolation
- Time series? → Forward/backward fill
- Categorical? → Mode or create 'Unknown' category
```

### Optimization

**Local vs Global Optima**
```
Q: How do you avoid local optima?
A: Multiple strategies:
1. Multiple random starts
2. Simulated annealing
3. Genetic algorithms
4. Convex optimization (guaranteed global optimum)
```

**Computational Complexity**
```
Be prepared to analyze:
- Time complexity: How runtime scales with input
- Space complexity: Memory requirements
- Trade-offs: Accuracy vs speed

Example responses:
"This algorithm is O(n²) which is acceptable for n<1000,
but for larger datasets I would use a spatial index structure
like R-tree to reduce to O(n log n)."
```

### Domain Knowledge

**Finance**
- Understand risk-return tradeoff
- Know basic portfolio theory
- Explain behavioral finance concepts

**Supply Chain**
- Just-in-time vs safety stock
- Bullwhip effect
- Network flow optimization

**Urban Planning**
- Environmental justice
- Transportation equity
- Land use integration

---

## 💡 Project Presentation Tips

1. **Start with Problem Statement**
   "This project optimizes stock portfolios using Modern Portfolio Theory..."

2. **Explain Methodology**
   "I implemented Markowitz optimization using quadratic programming..."

3. **Show Results**
   "The optimized portfolio achieved a Sharpe ratio of 1.8, 
    representing a 40% improvement over equal-weight allocation..."

4. **Discuss Limitations**
   "Current model assumes normal returns and static correlations.
    Future improvements could include robust optimization for fat tails..."

5. **Demonstrate Learning**
   "Through this project, I learned how to handle time-series cross-validation
    and the importance of parameter tuning in ARIMA models..."

---

## 📖 Key Papers & Resources

### Stock Prediction
- Markowitz (1952) - Portfolio Selection
- Fama & French (1993) - Common risk factors
- LSTMs: Hochreiter & Schmidhuber (1997)

### Supply Chain
- Dijkstra (1959) - Shortest path algorithm
- Christofides (1976) - TSP approximation
- Prophet: Taylor & Letham (2018)

### Urban Mobility
- Vuchic (2005) - Urban Transit Systems
- IPCC Guidelines - Emission factors
- 15-Minute City: Moreno et al. (2021)

---

## 🔍 Technical Terminology Checklist

Make sure you can explain these terms:
- [ ] Stationarity (time series)
- [ ] Autocorrelation / Partial Autocorrelation
- [ ] Sharpe Ratio vs Sortino Ratio
- [ ] Efficient Frontier
- [ ] Maximum Drawdown
- [ ] Overfitting vs Underfitting
- [ ] Cross-validation (especially for time series)
- [ ] Feature engineering
- [ ] Regularization (L1/L2)
- [ ] Ensemble methods
- [ ] Gradient descent
- [ ] Backpropagation
- [ ] Linear Programming
- [ ] Integer Programming
- [ ] NP-hard problems
- [ ] Greedy algorithms
- [ ] Dynamic programming
- [ ] Graph traversal (BFS/DFS)
- [ ] Complexity analysis (Big-O notation)

---

Good luck with your interviews! Remember: Understand the **why** behind each algorithm, not just the **how**.
