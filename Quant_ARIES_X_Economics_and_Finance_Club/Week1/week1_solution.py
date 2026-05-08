"""
Week 1: Getting Started with Tools & Data
Stock Market Analytics Challenge

Objectives:
- Work with multi-indexed DataFrames
- Clean and preprocess financial time-series data
- Calculate technical indicators (moving averages, returns, volatility)
- Perform exploratory analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)


class StockDataProcessor:
    """Process and analyze stock market data for multiple tickers"""
    
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize with list of stock tickers
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.multi_index_df = None
        
    def download_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Downloading data for {len(self.tickers)} tickers...")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        all_data = []
        
        for ticker in self.tickers:
            try:
                print(f"  Fetching {ticker}...")
                stock_data = yf.download(ticker, start=self.start_date, 
                                        end=self.end_date, progress=False)
                
                if not stock_data.empty:
                    # Add ticker column
                    stock_data['Ticker'] = ticker
                    all_data.append(stock_data)
                else:
                    print(f"  Warning: No data for {ticker}")
                    
            except Exception as e:
                print(f"  Error downloading {ticker}: {str(e)}")
        
        if not all_data:
            raise ValueError("No data downloaded for any ticker")
        
        # Combine all data
        self.data = pd.concat(all_data)
        print(f"\nTotal records downloaded: {len(self.data)}")
        
        return self.data
    
    def create_multiindex_dataframe(self):
        """Create MultiIndexed DataFrame with Ticker and Date as indices"""
        if self.data is None:
            raise ValueError("No data available. Call download_data() first")
        
        # Reset index to make Date a column
        df = self.data.reset_index()
        
        # Set MultiIndex
        df = df.set_index(['Ticker', 'Date'])
        df = df.sort_index()
        
        self.multi_index_df = df
        
        print("\nMultiIndex DataFrame created:")
        print(f"Index levels: {self.multi_index_df.index.names}")
        print(f"Shape: {self.multi_index_df.shape}")
        print(f"\nSample data:")
        print(self.multi_index_df.head(10))
        
        return self.multi_index_df
    
    def clean_data(self):
        """Clean the data: handle missing values, filter date range"""
        if self.multi_index_df is None:
            raise ValueError("MultiIndex DataFrame not created yet")
        
        df = self.multi_index_df.copy()
        
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        # 1. Check missing values per ticker
        print("\nMissing values per ticker:")
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                missing = ticker_data.isnull().sum()
                if missing.sum() > 0:
                    print(f"\n{ticker}:")
                    print(missing[missing > 0])
                else:
                    print(f"{ticker}: No missing values")
        
        # 2. Handle missing values
        print("\nHandling missing values...")
        
        # Group by ticker and forward fill, then backward fill
        df_cleaned = df.groupby(level='Ticker').apply(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        # Drop any remaining NaN rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        dropped_rows = initial_rows - len(df_cleaned)
        
        print(f"Rows dropped due to NaN: {dropped_rows}")
        
        # 3. Filter to last 10 years
        ten_years_ago = datetime.now() - timedelta(days=365*10)
        
        # Filter by date (second level of MultiIndex)
        df_cleaned = df_cleaned[
            df_cleaned.index.get_level_values('Date') >= ten_years_ago
        ]
        
        print(f"\nData filtered to last 10 years (from {ten_years_ago.date()})")
        print(f"Final shape: {df_cleaned.shape}")
        
        self.multi_index_df = df_cleaned
        
        return self.multi_index_df
    
    def add_technical_indicators(self):
        """Add technical indicators: returns, moving averages, volatility"""
        if self.multi_index_df is None:
            raise ValueError("No data available")
        
        print("\n" + "="*60)
        print("ADDING TECHNICAL INDICATORS")
        print("="*60)
        
        df = self.multi_index_df.copy()
        
        # Process each ticker separately
        processed_dfs = []
        
        for ticker in self.tickers:
            if ticker not in df.index.get_level_values('Ticker'):
                continue
                
            ticker_data = df.loc[ticker].copy()
            
            # 1. Daily Return (% change)
            ticker_data['Daily_Return'] = ticker_data['Close'].pct_change() * 100
            
            # 2. 7-day Moving Average
            ticker_data['SMA_7'] = ticker_data['Close'].rolling(window=7).mean()
            
            # 3. 30-day Moving Average
            ticker_data['SMA_30'] = ticker_data['Close'].rolling(window=30).mean()
            
            # 4. Rolling Volatility (30-day)
            ticker_data['Volatility_30'] = ticker_data['Daily_Return'].rolling(window=30).std()
            
            # Add ticker back as index
            ticker_data['Ticker'] = ticker
            ticker_data = ticker_data.reset_index().set_index(['Ticker', 'Date'])
            
            processed_dfs.append(ticker_data)
        
        # Combine all tickers
        self.multi_index_df = pd.concat(processed_dfs)
        self.multi_index_df = self.multi_index_df.sort_index()
        
        print("\nTechnical indicators added:")
        print("  - Daily_Return (% change in closing price)")
        print("  - SMA_7 (7-day Simple Moving Average)")
        print("  - SMA_30 (30-day Simple Moving Average)")
        print("  - Volatility_30 (30-day rolling volatility)")
        
        print("\nSample data with indicators:")
        print(self.multi_index_df[['Close', 'Daily_Return', 'SMA_7', 'SMA_30', 'Volatility_30']].tail(10))
        
        return self.multi_index_df
    
    def exploratory_analysis(self):
        """Perform exploratory analysis to answer key questions"""
        if self.multi_index_df is None:
            raise ValueError("No data available")
        
        print("\n" + "="*60)
        print("EXPLORATORY ANALYSIS")
        print("="*60)
        
        df = self.multi_index_df.copy()
        
        # Question 1: Which stock had the highest average return?
        print("\n1. HIGHEST AVERAGE RETURN OVER 10 YEARS")
        print("-" * 60)
        
        avg_returns = {}
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                avg_return = ticker_data['Daily_Return'].mean()
                avg_returns[ticker] = avg_return
        
        avg_returns_sorted = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
        
        print("\nAverage Daily Returns (%):")
        for ticker, ret in avg_returns_sorted:
            print(f"  {ticker}: {ret:.4f}%")
        
        best_ticker = avg_returns_sorted[0][0]
        best_return = avg_returns_sorted[0][1]
        print(f"\n🏆 Winner: {best_ticker} with {best_return:.4f}% average daily return")
        
        # Annualized return approximation
        annualized = ((1 + best_return/100) ** 252 - 1) * 100
        print(f"   Approximate annualized return: {annualized:.2f}%")
        
        # Question 2: Which stock had the most volatile month?
        print("\n2. MOST VOLATILE MONTH")
        print("-" * 60)
        
        monthly_volatilities = []
        
        for ticker in self.tickers:
            if ticker not in df.index.get_level_values('Ticker'):
                continue
                
            ticker_data = df.loc[ticker].copy()
            ticker_data = ticker_data.reset_index()
            
            # Group by year-month
            ticker_data['YearMonth'] = ticker_data['Date'].dt.to_period('M')
            monthly_vol = ticker_data.groupby('YearMonth')['Daily_Return'].std()
            
            for period, vol in monthly_vol.items():
                monthly_volatilities.append({
                    'Ticker': ticker,
                    'Month': period,
                    'Volatility': vol
                })
        
        vol_df = pd.DataFrame(monthly_volatilities)
        vol_df = vol_df.sort_values('Volatility', ascending=False)
        
        print("\nTop 10 Most Volatile Months:")
        print(vol_df.head(10).to_string(index=False))
        
        most_volatile = vol_df.iloc[0]
        print(f"\n🎢 Most Volatile: {most_volatile['Ticker']} in {most_volatile['Month']}")
        print(f"   Volatility: {most_volatile['Volatility']:.2f}%")
        
        return {
            'avg_returns': avg_returns_sorted,
            'most_volatile': most_volatile,
            'monthly_volatilities': vol_df
        }
    
    def visualize_analysis(self):
        """Create visualizations for the analysis"""
        if self.multi_index_df is None:
            raise ValueError("No data available")
        
        df = self.multi_index_df.copy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Stock Market Analysis - Last 10 Years', fontsize=16, fontweight='bold')
        
        # 1. Price Evolution
        ax = axes[0, 0]
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                dates = ticker_data.index.get_level_values('Date')
                ax.plot(dates, ticker_data['Close'], label=ticker, linewidth=2)
        ax.set_title('Stock Prices Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Normalized Returns (starting from 100)
        ax = axes[0, 1]
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                normalized = (1 + ticker_data['Daily_Return']/100).cumprod() * 100
                dates = ticker_data.index.get_level_values('Date')
                ax.plot(dates, normalized, label=ticker, linewidth=2)
        ax.set_title('Cumulative Returns (Base=100)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        
        # 3. Average Daily Returns
        ax = axes[1, 0]
        avg_returns = []
        labels = []
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                avg_returns.append(ticker_data['Daily_Return'].mean())
                labels.append(ticker)
        
        colors = ['green' if x > 0 else 'red' for x in avg_returns]
        ax.bar(labels, avg_returns, color=colors, alpha=0.7)
        ax.set_title('Average Daily Returns (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Volatility Comparison
        ax = axes[1, 1]
        volatilities = []
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                volatilities.append(ticker_data['Daily_Return'].std())
        
        ax.bar(labels, volatilities, color='orange', alpha=0.7)
        ax.set_title('Overall Volatility (Std Dev of Returns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Moving Averages Example (First Ticker)
        ax = axes[2, 0]
        first_ticker = self.tickers[0]
        if first_ticker in df.index.get_level_values('Ticker'):
            ticker_data = df.loc[first_ticker]
            dates = ticker_data.index.get_level_values('Date')
            
            # Plot last 2 years
            mask = dates >= (datetime.now() - timedelta(days=730))
            recent_data = ticker_data[mask]
            recent_dates = dates[mask]
            
            ax.plot(recent_dates, recent_data['Close'], label='Close', linewidth=2)
            ax.plot(recent_dates, recent_data['SMA_7'], label='SMA-7', linestyle='--', alpha=0.7)
            ax.plot(recent_dates, recent_data['SMA_30'], label='SMA-30', linestyle='--', alpha=0.7)
        
        ax.set_title(f'{first_ticker} - Moving Averages (Last 2 Years)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Returns Distribution
        ax = axes[2, 1]
        for ticker in self.tickers:
            if ticker in df.index.get_level_values('Ticker'):
                ticker_data = df.loc[ticker]
                returns = ticker_data['Daily_Return'].dropna()
                ax.hist(returns, bins=50, alpha=0.5, label=ticker, edgecolor='black')
        
        ax.set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig('week1_stock_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved as 'week1_stock_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("WEEK 1: STOCK MARKET ANALYTICS CHALLENGE")
    print("="*60)
    
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL']
    
    print(f"\nSelected Tickers: {', '.join(tickers)}")
    
    # Initialize processor
    processor = StockDataProcessor(tickers)
    
    # Step 1: Download data
    processor.download_data()
    
    # Step 2: Create MultiIndex DataFrame
    processor.create_multiindex_dataframe()
    
    # Step 3: Clean data
    processor.clean_data()
    
    # Step 4: Add technical indicators
    processor.add_technical_indicators()
    
    # Step 5: Exploratory analysis
    analysis_results = processor.exploratory_analysis()
    
    # Step 6: Visualizations
    processor.visualize_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\n✅ All tasks completed successfully")
    print("✅ MultiIndexed DataFrame created")
    print("✅ Data cleaned and preprocessed")
    print("✅ Technical indicators calculated")
    print("✅ Exploratory analysis performed")
    print("✅ Visualizations generated")


if __name__ == "__main__":
    main()
