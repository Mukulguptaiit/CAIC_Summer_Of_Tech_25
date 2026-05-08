"""
Stock Market Prediction & Portfolio Optimization
==================================================
This module implements a comprehensive stock market analysis system including:
1. Technical Indicators (RSI, MACD, Bollinger Bands, Moving Averages)
2. Predictive Models (ARIMA, SARIMAX, LSTM, Random Forest)
3. Trading Strategies with Risk Management (Sharpe Ratio, Drawdown, Stop-Loss)
4. Portfolio Optimization using Modern Portfolio Theory (MVO)

Interview Topics Covered:
- Time-series analysis and forecasting
- Machine learning for regression
- Financial risk metrics
- Optimization algorithms
- Feature engineering for financial data
"""

import numpy as np  # Numerical computing
import pandas as pd  # Data manipulation and analysis
import yfinance as yf  # Yahoo Finance API for stock data
import matplotlib.pyplot as plt  # Data visualization
from datetime import datetime, timedelta  # Date/time handling
from sklearn.ensemble import RandomForestRegressor  # ML model for prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Model evaluation
from sklearn.preprocessing import MinMaxScaler  # Feature scaling for neural networks
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Try importing optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. ARIMA/SARIMAX models will not be available.")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("Warning: TensorFlow not installed. LSTM models will not be available.")

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not installed. Portfolio optimization will not be available.")


class FinancialIndicators:
    """
    Technical Indicators Calculator
    ================================
    Implements common financial technical indicators used in trading strategies.
    
    Interview Note: These indicators help identify trends, momentum, and volatility.
    They're the foundation of many quantitative trading strategies.
    """
    
    @staticmethod
    def calculate_sma(data, window=20):
        """
        Simple Moving Average (SMA)
        ----------------------------
        Calculates the arithmetic mean of closing prices over a specified window.
        
        Use Case: Identifies trends and support/resistance levels
        Formula: SMA = (Sum of closing prices over N periods) / N
        
        Parameters:
        -----------
        data : DataFrame - Stock price data with 'Close' column
        window : int - Number of periods for averaging (default: 20 days)
        
        Returns:
        --------
        Series - SMA values
        
        Interview Tip: SMA is lagging indicator - it smooths out price action
        but responds slower to recent price changes than EMA.
        """
        return data['Close'].rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data, window=20):
        """
        Exponential Moving Average (EMA)
        ---------------------------------
        Calculates weighted average giving more importance to recent prices.
        
        Use Case: More responsive to recent price changes than SMA
        Formula: EMA_today = (Price_today × α) + (EMA_yesterday × (1-α))
                 where α = 2/(window+1)
        
        Parameters:
        -----------
        data : DataFrame - Stock price data
        window : int - Period for calculation (default: 20)
        
        Returns:
        --------
        Series - EMA values
        
        Interview Tip: EMA reacts faster to price changes, making it better
        for short-term trading signals. The 'adjust=False' ensures proper
        recursive calculation.
        """
        return data['Close'].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, window=14):
        """
        Relative Strength Index (RSI)
        ------------------------------
        Momentum oscillator measuring speed and magnitude of price changes.
        
        Use Case: Identifies overbought (>70) and oversold (<30) conditions
        Formula: RSI = 100 - (100 / (1 + RS))
                 where RS = Average Gain / Average Loss
        
        Parameters:
        -----------
        data : DataFrame - Stock price data
        window : int - Period for calculation (default: 14 days)
        
        Returns:
        --------
        Series - RSI values (0-100 scale)
        
        Interview Tip: RSI > 70 suggests overbought (potential sell signal)
                       RSI < 30 suggests oversold (potential buy signal)
        This is a mean-reversion indicator.
        """
        delta = data['Close'].diff()  # Calculate daily price changes
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Calculate relative strength and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """
        Moving Average Convergence Divergence (MACD)
        ---------------------------------------------
        Trend-following momentum indicator showing relationship between two EMAs.
        
        Use Case: Identifies trend changes and momentum shifts
        Components:
        - MACD Line: Difference between fast and slow EMA
        - Signal Line: 9-period EMA of MACD line
        - Histogram: MACD - Signal (shows convergence/divergence)
        
        Parameters:
        -----------
        data : DataFrame - Stock price data
        fast : int - Fast EMA period (default: 12)
        slow : int - Slow EMA period (default: 26)
        signal : int - Signal line period (default: 9)
        
        Returns:
        --------
        tuple - (macd, signal_line, histogram)
        
        Interview Tip: Buy signal when MACD crosses above signal line
                       Sell signal when MACD crosses below signal line
        Histogram shows strength of momentum.
        """
        # Calculate fast and slow EMAs
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        
        # MACD line is difference between EMAs
        macd = ema_fast - ema_slow
        
        # Signal line is EMA of MACD
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # Histogram shows MACD-Signal divergence
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """
        Bollinger Bands
        ---------------
        Volatility bands placed above and below a moving average.
        
        Use Case: Identifies overbought/oversold conditions and volatility
        Components:
        - Middle Band: 20-period SMA
        - Upper Band: SMA + (2 × Standard Deviation)
        - Lower Band: SMA - (2 × Standard Deviation)
        
        Parameters:
        -----------
        data : DataFrame - Stock price data
        window : int - Period for calculation (default: 20)
        num_std : int - Number of standard deviations (default: 2)
        
        Returns:
        --------
        tuple - (upper_band, middle_band, lower_band)
        
        Interview Tip: Bands widen during high volatility, narrow during low volatility.
        Price touching upper band may indicate overbought, lower band oversold.
        Bollinger Squeeze (narrow bands) often precedes volatility expansion.
        """
        # Calculate middle band (SMA)
        sma = data['Close'].rolling(window=window).mean()
        
        # Calculate standard deviation for volatility
        std = data['Close'].rolling(window=window).std()
        
        # Upper and lower bands are SMA ± (k × std)
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def add_all_indicators(data):
        """Add all technical indicators to dataframe"""
        df = data.copy()
        
        # Moving Averages
        df['SMA_20'] = FinancialIndicators.calculate_sma(data, 20)
        df['SMA_50'] = FinancialIndicators.calculate_sma(data, 50)
        df['EMA_12'] = FinancialIndicators.calculate_ema(data, 12)
        df['EMA_26'] = FinancialIndicators.calculate_ema(data, 26)
        
        # RSI
        df['RSI'] = FinancialIndicators.calculate_rsi(data)
        
        # MACD
        macd, signal, histogram = FinancialIndicators.calculate_macd(data)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = FinancialIndicators.calculate_bollinger_bands(data)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # Price changes
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        return df


class StockPredictor:
    """
    Stock Price Prediction Engine
    ==============================
    Implements multiple forecasting models for stock price prediction:
    - Random Forest (ML approach)
    - ARIMA (Statistical time-series)
    - SARIMAX (Seasonal ARIMA with exogenous variables)
    - LSTM (Deep learning sequential model)
    
    Interview Topics:
    - Feature engineering for time-series
    - Train-test split for temporal data (no shuffling!)
    - Model evaluation metrics (RMSE, MAE)
    - Handling missing values in financial data
    """
    
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize predictor with stock ticker and date range
        
        Parameters:
        -----------
        ticker : str - Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date : str - Start date in 'YYYY-MM-DD' format
        end_date : str - End date in 'YYYY-MM-DD' format
        
        Interview Note: We store processed data separately to avoid
        repeated API calls and maintain data consistency.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None  # Raw stock data
        self.processed_data = None  # Data with indicators
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.processed_data = FinancialIndicators.add_all_indicators(self.data)
        print(f"Data fetched: {len(self.data)} records")
        return self.data
    
    def prepare_data_for_ml(self, target_col='Close', features=None, test_size=0.2):
        """
        Prepare Data for Machine Learning Models
        -----------------------------------------
        Converts time-series data into supervised learning format.
        
        CRITICAL for Interview: Time-series data requires special handling:
        1. NO random shuffling (preserves temporal order)
        2. Train-test split is chronological (earlier data = train, later = test)
        3. Features must not include future information (no data leakage)
        
        Parameters:
        -----------
        target_col : str - Column to predict (default: 'Close')
        features : list - Feature columns to use (default: technical indicators)
        test_size : float - Proportion of data for testing (default: 0.2 = 20%)
        
        Returns:
        --------
        tuple - (X_train, X_test, y_train, y_test, test_dates)
        
        Interview Tip: Explain why we DON'T use sklearn's train_test_split here -
        it shuffles data which breaks temporal dependencies!
        """
        # Drop rows with NaN (from rolling calculations)
        df = self.processed_data.dropna()
        
        # Default feature set includes technical indicators
        if features is None:
            features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 
                       'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'Volatility']
        
        # Prepare feature matrix X and target vector y
        X = df[features].values
        y = df[target_col].values
        
        # Chronological train-test split (IMPORTANT: no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, df.index[split_idx:]
    
    def train_random_forest(self, n_estimators=100):
        """
        Train Random Forest Regression Model
        -------------------------------------
        Ensemble learning method using multiple decision trees.
        
        How it works:
        1. Creates n_estimators decision trees with random subsets of data
        2. Each tree makes a prediction
        3. Final prediction = average of all tree predictions
        
        Advantages for Finance:
        - Handles non-linear relationships well
        - Resistant to overfitting (with enough trees)
        - Provides feature importance scores
        - No need for feature scaling
        
        Parameters:
        -----------
        n_estimators : int - Number of trees in the forest (default: 100)
        
        Returns:
        --------
        tuple - (trained_model, predictions, actual_values, test_dates)
        
        Interview Questions to Prepare:
        Q: Why Random Forest over single decision tree?
        A: Reduces overfitting through averaging, more robust predictions
        
        Q: How do you prevent overfitting?
        A: Increase n_estimators, use max_depth, min_samples_split parameters
        
        Q: What's the time complexity?
        A: Training O(n_trees × n_samples × log(n_samples) × n_features)
        """
        print("\nTraining Random Forest model...")
        
        # Prepare data with technical indicators as features
        X_train, X_test, y_train, y_test, test_dates = self.prepare_data_for_ml()
        
        # Initialize Random Forest with key parameters:
        # - n_estimators: number of trees (more = better but slower)
        # - random_state: ensures reproducibility
        # - n_jobs=-1: uses all CPU cores for parallel processing
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        
        # Fit model to training data
        model.fit(X_train, y_train)
        
        # Generate predictions on test set
        predictions = model.predict(X_test)
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))  # Root Mean Squared Error
        mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
        
        print(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return model, predictions, y_test, test_dates
    
    def train_arima(self, order=(5, 1, 0)):
        """Train ARIMA model"""
        if not HAS_STATSMODELS:
            print("ARIMA not available. Install statsmodels: pip install statsmodels")
            return None, None, None, None
        
        print(f"\nTraining ARIMA model with order {order}...")
        
        # Use closing prices
        prices = self.data['Close'].dropna()
        split_idx = int(len(prices) * 0.8)
        train, test = prices[:split_idx], prices[split_idx:]
        
        model = ARIMA(train, order=order)
        fitted_model = model.fit()
        
        # Forecast
        predictions = fitted_model.forecast(steps=len(test))
        
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        
        print(f"ARIMA - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return fitted_model, predictions.values, test.values, test.index
    
    def train_sarimax(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Train SARIMAX model"""
        if not HAS_STATSMODELS:
            print("SARIMAX not available. Install statsmodels: pip install statsmodels")
            return None, None, None, None
        
        print(f"\nTraining SARIMAX model...")
        
        prices = self.data['Close'].dropna()
        split_idx = int(len(prices) * 0.8)
        train, test = prices[:split_idx], prices[split_idx:]
        
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        predictions = fitted_model.forecast(steps=len(test))
        
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        
        print(f"SARIMAX - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return fitted_model, predictions.values, test.values, test.index
    
    def train_lstm(self, lookback=60, epochs=50, batch_size=32):
        """Train LSTM neural network"""
        if not HAS_KERAS:
            print("LSTM not available. Install tensorflow: pip install tensorflow")
            return None, None, None, None
        
        print(f"\nTraining LSTM model...")
        
        # Prepare data
        prices = self.data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            X.append(scaled_prices[i-lookback:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                 validation_data=(X_test, y_test), verbose=0)
        
        # Predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)
        
        print(f"LSTM - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        test_dates = self.data.index[split_idx + lookback:]
        
        return model, predictions.flatten(), y_test_actual.flatten(), test_dates
    
    def plot_predictions(self, actual, predicted, dates, model_name, save_path=None):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual, label='Actual Price', color='blue', linewidth=2)
        plt.plot(dates, predicted, label='Predicted Price', color='red', linewidth=2, linestyle='--')
        plt.title(f'{self.ticker} Stock Price Prediction - {model_name}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class TradingStrategy:
    """
    Trading Strategy Backtesting Framework
    =======================================
    Implements and evaluates trading strategies using historical data.
    
    Interview Concepts:
    - Moving Average Crossover (momentum strategy)
    - Risk-adjusted returns (Sharpe Ratio)
    - Drawdown analysis (risk measurement)
    - Stop-loss implementation (risk management)
    
    Key Metrics Explained:
    ----------------------
    1. Sharpe Ratio: (Return - Risk-free rate) / Volatility
       - Measures risk-adjusted return
       - Higher is better (>1 is good, >2 is excellent)
    
    2. Maximum Drawdown: Largest peak-to-trough decline
       - Measures worst-case loss
       - Critical for risk management
    
    3. Win Rate: Percentage of profitable trades
       - Useful but can be misleading (small wins, big losses)
    """
    
    @staticmethod
    def simple_moving_average_strategy(data, short_window=20, long_window=50):
        """
        Simple Moving Average (SMA) Crossover Strategy
        -----------------------------------------------
        Classic momentum strategy based on two moving averages.
        
        Trading Rules:
        - BUY Signal: When short MA crosses ABOVE long MA (golden cross)
        - SELL Signal: When short MA crosses BELOW long MA (death cross)
        
        Logic: Short MA responds faster to price changes. When it crosses
        above slow MA, indicates upward momentum. Vice versa for downtrend.
        
        Parameters:
        -----------
        data : DataFrame - Price data with 'Close' column
        short_window : int - Fast MA period (default: 20 days)
        long_window : int - Slow MA period (default: 50 days)
        
        Returns:
        --------
        DataFrame with columns:
        - price: Close prices
        - short_mavg: Short-term MA
        - long_mavg: Long-term MA  
        - signal: 1 = long position, 0 = no position
        - positions: Change in signal (marks entry/exit)
        
        Interview Tip: This is a trend-following strategy. Works well in
        trending markets but generates false signals in sideways markets.
        Consider adding filters (volume, RSI) to reduce whipsaws.
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        
        # Calculate short and long moving averages
        signals['short_mavg'] = data['Close'].rolling(window=short_window).mean()
        signals['long_mavg'] = data['Close'].rolling(window=long_window).mean()
        
        # Generate signals: 1 = long, 0 = neutral
        signals['signal'] = 0.0
        
        # Create buy signal when short MA > long MA
        signals['signal'][short_window:] = np.where(
            signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0
        )
        
        # Track position changes (1 = buy, -1 = sell)
        signals['positions'] = signals['signal'].diff()
        
        return signals
    
    @staticmethod
    def calculate_returns(signals, initial_capital=10000):
        """Calculate strategy returns"""
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions['stock'] = 100 * signals['signal']
        
        portfolio = positions.multiply(signals['price'], axis=0)
        pos_diff = positions.diff()
        
        portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """
        Calculate Sharpe Ratio
        ----------------------
        Measures risk-adjusted return of an investment strategy.
        
        Formula: Sharpe = (E[R] - Rf) / σ(R) × √252
        where:
        - E[R] = Expected return
        - Rf = Risk-free rate (default: 2% annually)
        - σ(R) = Standard deviation of returns
        - √252 = Annualization factor (252 trading days/year)
        
        Interpretation:
        - Sharpe < 1.0: Poor risk-adjusted returns
        - Sharpe 1.0-2.0: Good returns for the risk taken
        - Sharpe > 2.0: Excellent risk-adjusted returns
        
        Parameters:
        -----------
        returns : Series - Daily returns
        risk_free_rate : float - Annual risk-free rate (default: 0.02)
        
        Returns:
        --------
        float - Annualized Sharpe Ratio
        
        Interview Note: Sharpe ratio penalizes both upside and downside
        volatility equally. Sortino ratio (uses only downside deviation)
        is sometimes preferred for asymmetric risk.
        """
        # Convert annual risk-free rate to daily
        daily_rf = risk_free_rate / 252
        
        # Calculate excess returns over risk-free rate
        excess_returns = returns - daily_rf
        
        # Sharpe = mean(excess returns) / std(returns) × √252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        """
        Calculate Maximum Drawdown
        ---------------------------
        Measures the largest peak-to-trough decline in portfolio value.
        
        Definition: Maximum observed loss from a peak to a trough before
        a new peak is achieved.
        
        Formula: MDD = (Trough Value - Peak Value) / Peak Value
        
        Why it matters:
        - Indicates worst-case scenario loss
        - Key metric for risk management
        - Helps size positions appropriately
        - Important for psychological/behavioral finance
        
        Parameters:
        -----------
        portfolio_values : Series - Time series of portfolio values
        
        Returns:
        --------
        float - Maximum drawdown (negative value, e.g., -0.25 = -25%)
        
        Interview Example:
        If portfolio goes from $100k → $80k → $120k:
        - Peak: $100k
        - Trough: $80k  
        - MDD: ($80k - $100k) / $100k = -20%
        
        Even though portfolio recovered to $120k, the MDD is still -20%
        because that was the maximum decline from peak.
        """
        # Calculate running maximum (highest value so far)
        cumulative_max = portfolio_values.cummax()
        
        # Calculate drawdown at each point: (current - peak) / peak
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        
        # Maximum drawdown is the worst (most negative) value
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    @staticmethod
    def apply_stop_loss(signals, stop_loss_pct=0.05):
        """Apply stop-loss strategy"""
        positions = signals.copy()
        positions['stop_loss'] = 0
        
        in_position = False
        entry_price = 0
        
        for i in range(len(positions)):
            if positions['signal'].iloc[i] == 1 and not in_position:
                in_position = True
                entry_price = positions['price'].iloc[i]
            elif in_position:
                current_price = positions['price'].iloc[i]
                if (entry_price - current_price) / entry_price >= stop_loss_pct:
                    positions['signal'].iloc[i] = 0
                    positions['stop_loss'].iloc[i] = 1
                    in_position = False
        
        return positions


class PortfolioOptimization:
    """
    Modern Portfolio Theory (MPT) Implementation
    =============================================
    Implements Markowitz Portfolio Optimization for finding optimal asset weights.
    
    Core Concept (Nobel Prize-winning theory):
    - Diversification reduces risk without sacrificing returns
    - Optimal portfolio maximizes return for given risk level
    - Efficient Frontier: Set of optimal portfolios
    
    Mathematical Framework:
    -----------------------
    Portfolio Return: E[Rp] = Σ(wi × E[Ri])
    Portfolio Variance: σp² = w^T × Σ × w
    Sharpe Ratio: (E[Rp] - Rf) / σp
    
    where:
    - wi = weight of asset i
    - E[Ri] = expected return of asset i
    - Σ = covariance matrix
    - w = vector of weights
    
    Interview Topics:
    - Quadratic optimization (minimizing variance)
    - Constraint handling (weights sum to 1, no shorting)
    - Risk-return tradeoff
    - Correlation and diversification benefits
    """
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize portfolio optimizer
        
        Parameters:
        -----------
        tickers : list - Stock symbols to include in portfolio
        start_date : str - Start date for historical data
        end_date : str - End date for historical data
        
        Interview Note: We need sufficient historical data to estimate
        returns and covariance matrix reliably. Typical: 3-5 years minimum.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None  # Price data
        self.returns = None  # Daily returns
        
    def fetch_portfolio_data(self):
        """Fetch data for multiple stocks"""
        print(f"Fetching portfolio data for {len(self.tickers)} stocks...")
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        self.returns = self.data.pct_change().dropna()
        print(f"Data fetched successfully")
        return self.data
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return and risk"""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def optimize_portfolio(self, target='sharpe'):
        """
        Optimize Portfolio Using Scipy Optimization
        --------------------------------------------
        Finds optimal asset weights to maximize Sharpe ratio or minimize variance.
        
        Optimization Problem:
        ---------------------
        If target='sharpe':
            Maximize: (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
            Subject to: Σwi = 1, wi ≥ 0 (no short selling)
        
        If target='variance':
            Minimize: w^T × Covariance Matrix × w
            Subject to: Σwi = 1, wi ≥ 0
        
        Parameters:
        -----------
        target : str - 'sharpe' for maximum Sharpe, 'variance' for min variance
        
        Returns:
        --------
        tuple - (optimal_weights, return, risk, sharpe_ratio)
        
        Interview Deep Dive:
        --------------------
        Q: Why use SLSQP method?
        A: Sequential Least Squares Programming handles constraints well
           (sum to 1, non-negative weights). Good for portfolio optimization.
        
        Q: What if we allow short selling?
        A: Change bounds to (-1, 1) or remove lower bound constraint.
           Short selling can improve returns but adds significant risk.
        
        Q: How to handle transaction costs?
        A: Add penalty term to objective function proportional to portfolio
           turnover: cost × Σ|wi_new - wi_old|
        """
        if not HAS_SCIPY:
            print("Portfolio optimization not available. Install scipy: pip install scipy")
            return None
        
        num_assets = len(self.tickers)
        
        # Objective function: negative Sharpe (minimize instead of maximize)
        def negative_sharpe(weights):
            return -self.calculate_portfolio_metrics(weights)[2]
        
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return self.calculate_portfolio_metrics(weights)[1]
        
        # Constraints: weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: each weight between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Run optimization
        if target == 'sharpe':
            result = minimize(negative_sharpe, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        else:  # Minimum variance
            result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        
        # Extract optimal weights
        optimal_weights = result.x
        ret, risk, sharpe = self.calculate_portfolio_metrics(optimal_weights)
        
        print(f"\nOptimal Portfolio ({target}):")
        for ticker, weight in zip(self.tickers, optimal_weights):
            print(f"{ticker}: {weight*100:.2f}%")
        print(f"\nExpected Annual Return: {ret*100:.2f}%")
        print(f"Annual Volatility: {risk*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        return optimal_weights, ret, risk, sharpe
    
    def efficient_frontier(self, num_portfolios=5000):
        """Generate efficient frontier"""
        if not HAS_SCIPY:
            print("Efficient frontier not available. Install scipy: pip install scipy")
            return None, None, None
        
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            
            ret, risk, sharpe = self.calculate_portfolio_metrics(weights)
            results[0, i] = risk
            results[1, i] = ret
            results[2, i] = sharpe
        
        return results[0], results[1], results[2]
    
    def plot_efficient_frontier(self, save_path=None):
        """Plot the efficient frontier"""
        risks, returns, sharpes = self.efficient_frontier()
        
        if risks is None:
            return
        
        plt.figure(figsize=(12, 8))
        plt.scatter(risks, returns, c=sharpes, cmap='viridis', marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Mark optimal portfolio
        optimal_weights, opt_ret, opt_risk, opt_sharpe = self.optimize_portfolio('sharpe')
        plt.scatter(opt_risk, opt_ret, marker='*', color='red', s=500, 
                   label=f'Optimal Portfolio (Sharpe: {opt_sharpe:.2f})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("STOCK MARKET PREDICTION & PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    # Define parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-12-01"
    
    # 1. Stock Prediction
    print("\n" + "=" * 80)
    print("PART 1: STOCK PRICE PREDICTION")
    print("=" * 80)
    
    predictor = StockPredictor(ticker, start_date, end_date)
    data = predictor.fetch_data()
    
    # Display indicators
    print("\nTechnical Indicators Sample:")
    print(predictor.processed_data[['Close', 'SMA_20', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].tail())
    
    # Train models
    rf_model, rf_pred, rf_actual, rf_dates = predictor.train_random_forest()
    
    if HAS_STATSMODELS:
        arima_model, arima_pred, arima_actual, arima_dates = predictor.train_arima()
        sarimax_model, sarimax_pred, sarimax_actual, sarimax_dates = predictor.train_sarimax()
    
    if HAS_KERAS:
        lstm_model, lstm_pred, lstm_actual, lstm_dates = predictor.train_lstm(epochs=20)
    
    # 2. Trading Strategy
    print("\n" + "=" * 80)
    print("PART 2: TRADING STRATEGY BACKTESTING")
    print("=" * 80)
    
    signals = TradingStrategy.simple_moving_average_strategy(data)
    portfolio = TradingStrategy.calculate_returns(signals)
    
    sharpe = TradingStrategy.calculate_sharpe_ratio(portfolio['returns'].dropna())
    max_dd = TradingStrategy.calculate_max_drawdown(portfolio['total'])
    
    print(f"\nStrategy Performance:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    print(f"Total Return: {(portfolio['total'].iloc[-1]/portfolio['total'].iloc[0] - 1)*100:.2f}%")
    
    # Apply stop-loss
    signals_sl = TradingStrategy.apply_stop_loss(signals, stop_loss_pct=0.05)
    portfolio_sl = TradingStrategy.calculate_returns(signals_sl)
    
    print(f"\nWith 5% Stop-Loss:")
    print(f"Total Return: {(portfolio_sl['total'].iloc[-1]/portfolio_sl['total'].iloc[0] - 1)*100:.2f}%")
    
    # 3. Portfolio Optimization
    print("\n" + "=" * 80)
    print("PART 3: PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    portfolio_opt = PortfolioOptimization(tickers, start_date, end_date)
    portfolio_opt.fetch_portfolio_data()
    
    # Optimize for maximum Sharpe ratio
    optimal_weights, ret, risk, sharpe = portfolio_opt.optimize_portfolio('sharpe')
    
    # Plot efficient frontier
    print("\nGenerating Efficient Frontier...")
    portfolio_opt.plot_efficient_frontier()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
