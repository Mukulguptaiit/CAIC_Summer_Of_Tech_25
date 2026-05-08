"""
Week 2: Stock Market Prediction - Model Implementation and Evaluation
Implement Linear Regression, ARIMA, and Random Forest models for stock price forecasting
Backtest trading strategy based on predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    print("Warning: statsmodels not installed. ARIMA will not be available.")


class Week2StockPredictor:
    """Stock price prediction using multiple models"""
    
    def __init__(self, ticker='AMZN', start_date='2014-01-01'):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_data(self):
        """Load stock data"""
        print(f"Loading data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, progress=False)
        print(f"Data loaded: {len(self.data)} records")
        return self.data
    
    def prepare_data(self, train_split=0.8):
        """Split data into training and testing sets"""
        split_idx = int(len(self.data) * train_split)
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        print(f"\nData split:")
        print(f"  Train: {len(self.train_data)} records ({self.train_data.index[0].date()} to {self.train_data.index[-1].date()})")
        print(f"  Test:  {len(self.test_data)} records ({self.test_data.index[0].date()} to {self.test_data.index[-1].date()})")
        
        return self.train_data, self.test_data
    
    def create_features_linear_regression(self, data, lookback=5):
        """Create features for Linear Regression (past N days)"""
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data['Close'].iloc[i-lookback:i].values)
            y.append(data['Close'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_linear_regression(self, lookback=5):
        """Train Linear Regression model"""
        print(f"\n{'='*60}")
        print("TRAINING LINEAR REGRESSION")
        print('='*60)
        
        # Prepare features
        X_train, y_train = self.create_features_linear_regression(self.train_data, lookback)
        X_test, y_test = self.create_features_linear_regression(self.test_data, lookback)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Direction accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        self.models['Linear Regression'] = model
        self.predictions['Linear Regression'] = predictions
        self.metrics['Linear Regression'] = {
            'MAE': mae,
            'RMSE': rmse,
            'Direction_Accuracy': direction_accuracy,
            'actual': y_test
        }
        
        return model, predictions
    
    def train_arima(self, order=(5, 1, 0)):
        """Train ARIMA model"""
        if not HAS_ARIMA:
            print("\nARIMA not available. Install statsmodels: pip install statsmodels")
            return None, None
        
        print(f"\n{'='*60}")
        print("TRAINING ARIMA")
        print('='*60)
        
        # Train on training data
        train_prices = self.train_data['Close']
        model = ARIMA(train_prices, order=order)
        fitted_model = model.fit()
        
        # Forecast test period
        forecast_steps = len(self.test_data)
        predictions = fitted_model.forecast(steps=forecast_steps)
        
        # Evaluate
        actual = self.test_data['Close'].values
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        # Direction accuracy
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        print(f"Order: {order}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        self.models['ARIMA'] = fitted_model
        self.predictions['ARIMA'] = predictions.values
        self.metrics['ARIMA'] = {
            'MAE': mae,
            'RMSE': rmse,
            'Direction_Accuracy': direction_accuracy,
            'actual': actual
        }
        
        return fitted_model, predictions
    
    def add_technical_indicators(self, data):
        """Add technical indicators for Random Forest"""
        df = data.copy()
        
        # Moving averages
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Lag features
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Close_Lag2'] = df['Close'].shift(2)
        
        return df
    
    def train_random_forest(self, n_estimators=100):
        """Train Random Forest model"""
        print(f"\n{'='*60}")
        print("TRAINING RANDOM FOREST")
        print('='*60)
        
        # Add technical indicators
        train_with_indicators = self.add_technical_indicators(self.train_data)
        test_with_indicators = self.add_technical_indicators(self.test_data)
        
        # Features
        feature_cols = ['SMA_7', 'SMA_30', 'RSI', 'Volatility', 'Close_Lag1', 'Close_Lag2']
        
        # Remove NaN rows
        train_clean = train_with_indicators.dropna()
        test_clean = test_with_indicators.dropna()
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Close']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Close']
        
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Direction accuracy
        actual_direction = np.diff(y_test.values) > 0
        pred_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        print(f"\nFeature Importance:")
        print(feature_importance.to_string(index=False))
        
        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = predictions
        self.metrics['Random Forest'] = {
            'MAE': mae,
            'RMSE': rmse,
            'Direction_Accuracy': direction_accuracy,
            'actual': y_test.values,
            'feature_importance': feature_importance
        }
        
        return model, predictions
    
    def compare_models(self):
        """Compare all models"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print('='*60)
        
        comparison = pd.DataFrame({
            'Model': list(self.metrics.keys()),
            'MAE': [self.metrics[m]['MAE'] for m in self.metrics.keys()],
            'RMSE': [self.metrics[m]['RMSE'] for m in self.metrics.keys()],
            'Direction_Accuracy': [self.metrics[m]['Direction_Accuracy'] for m in self.metrics.keys()]
        })
        
        comparison = comparison.sort_values('MAE')
        
        print("\n" + comparison.to_string(index=False))
        
        best_model = comparison.iloc[0]['Model']
        print(f"\n🏆 Best Model (by MAE): {best_model}")
        
        return comparison, best_model
    
    def backtest_strategy(self, model_name, initial_capital=10000):
        """Backtest trading strategy"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {model_name}")
        print('='*60)
        
        predictions = self.predictions[model_name]
        actual = self.metrics[model_name]['actual']
        
        # Get corresponding dates
        if model_name == 'Linear Regression':
            dates = self.test_data.index[5:]  # Skip lookback period
        elif model_name == 'ARIMA':
            dates = self.test_data.index
        else:  # Random Forest
            dates = self.test_data.index[len(self.test_data) - len(predictions):]
        
        # Align predictions with actual prices
        actual_prices = actual[:len(predictions)]
        
        # Simulate trading
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long
        trades = []
        portfolio_values = [initial_capital]
        
        for i in range(len(predictions) - 1):
            current_price = actual_prices[i]
            predicted_next = predictions[i]
            
            # Trading signal: Buy if predicted > current, Sell if predicted < current
            if predicted_next > current_price and position == 0:
                # Buy signal
                shares = capital / current_price
                position = 1
                trades.append({
                    'date': dates[i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares
                })
            elif predicted_next < current_price and position == 1:
                # Sell signal
                capital = shares * current_price
                position = 0
                trades.append({
                    'date': dates[i],
                    'action': 'SELL',
                    'price': current_price,
                    'profit': capital - initial_capital
                })
            
            # Calculate portfolio value
            if position == 1:
                portfolio_value = shares * current_price
            else:
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
        
        # Close position if still open
        if position == 1:
            final_price = actual_prices[-1]
            capital = shares * final_price
            trades.append({
                'date': dates[-1],
                'action': 'SELL',
                'price': final_price,
                'profit': capital - initial_capital
            })
        
        # Calculate metrics
        final_value = portfolio_values[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Sharpe Ratio (simplified)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        print(f"\nInitial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print(f"\nTrades Summary:")
        if trades:
            trades_df = pd.DataFrame(trades)
            print(trades_df.to_string(index=False))
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'sharpe_ratio': sharpe_ratio
        }
    
    def visualize_results(self):
        """Visualize predictions and backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.ticker} - Model Predictions & Backtesting', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Actual
        ax = axes[0, 0]
        for model_name in self.predictions.keys():
            predictions = self.predictions[model_name]
            actual = self.metrics[model_name]['actual']
            
            # Get dates
            if model_name == 'Linear Regression':
                dates = self.test_data.index[5:5+len(predictions)]
            elif model_name == 'ARIMA':
                dates = self.test_data.index[:len(predictions)]
            else:
                dates = self.test_data.index[len(self.test_data)-len(predictions):]
            
            ax.plot(dates, predictions, label=f'{model_name} Predicted', linestyle='--', alpha=0.7)
        
        ax.plot(self.test_data.index, self.test_data['Close'], label='Actual', color='black', linewidth=2)
        ax.set_title('Predictions vs Actual Prices', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Model Comparison
        ax = axes[0, 1]
        models = list(self.metrics.keys())
        maes = [self.metrics[m]['MAE'] for m in models]
        
        ax.bar(models, maes, color=['#3498db', '#e74c3c', '#2ecc71'][:len(models)], alpha=0.7)
        ax.set_title('Model Comparison - MAE', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error ($)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Direction Accuracy
        ax = axes[1, 0]
        dir_acc = [self.metrics[m]['Direction_Accuracy'] for m in models]
        
        ax.bar(models, dir_acc, color=['#9b59b6', '#f39c12', '#1abc9c'][:len(models)], alpha=0.7)
        ax.set_title('Direction Prediction Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.axhline(y=50, color='red', linestyle='--', label='Random Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Feature Importance (if Random Forest exists)
        ax = axes[1, 1]
        if 'Random Forest' in self.metrics and 'feature_importance' in self.metrics['Random Forest']:
            feat_imp = self.metrics['Random Forest']['feature_importance']
            ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue', alpha=0.7)
            ax.set_title('Random Forest - Feature Importance', fontsize=12, fontweight='bold')
            ax.set_xlabel('Importance')
        else:
            ax.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('week2_predictions.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved as 'week2_predictions.png'")
        plt.show()


def main():
    """Main execution"""
    print("="*60)
    print("WEEK 2: STOCK MARKET PREDICTION")
    print("="*60)
    
    # Initialize predictor
    predictor = Week2StockPredictor(ticker='AMZN')
    
    # Load and prepare data
    predictor.load_data()
    predictor.prepare_data(train_split=0.8)
    
    # Train models
    predictor.train_linear_regression(lookback=5)
    predictor.train_arima(order=(5, 1, 0))
    predictor.train_random_forest(n_estimators=100)
    
    # Compare models
    comparison, best_model = predictor.compare_models()
    
    # Backtest best model
    backtest_results = predictor.backtest_strategy(best_model)
    
    # Visualize
    predictor.visualize_results()
    
    print("\n" + "="*60)
    print("WEEK 2 COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
