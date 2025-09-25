The Refactored Python Script (`financial_analysis_toolkit.py`)

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# --- Task 1: Fund Performance Analysis ---

def analyze_fund_performance(fund_filepath, index_filepath):
    """
    Analyzes fund performance by calculating cumulative returns and comparing one fund
    against the HS300 index.
    """
    print("\n--- Running Task 1: Fund Performance Analysis ---")
    
    # 1.1: Read stock and fund data
    funds_df = pd.read_excel(fund_filepath, index_col=0)
    selected_funds = funds_df.iloc[:, :5].sort_index(ascending=True)

    # 1.2: Calculate cumulative return for 5 funds
    start_date = 20200102
    for col in selected_funds.columns:
        start_value = selected_funds.loc[start_date, col]
        selected_funds[f'{col}_cum_return'] = (selected_funds[col] - start_value) / start_value
    
    print("Cumulative Returns for 5 Funds:")
    print(selected_funds.iloc[:, 5:])

    # 1.3: Plot one fund's cumulative return vs. HS300 index
    hs300_df = pd.read_excel(index_filepath, index_col='trade_date').sort_index(ascending=True)
    hs_close = hs300_df[['close']]
    hs_start_date = 20200106 # Match the earliest available date for comparison
    hs_start_value = hs300_df.loc[hs_start_date, 'close']
    hs_close['hs_cum_return'] = (hs_close['close'] - hs_start_value) / hs_start_value

    # Combine data for plotting
    comparison_df = pd.concat([selected_funds.iloc[:, -1], hs_close['hs_cum_return']], axis=1)
    comparison_df.index = pd.to_datetime(comparison_df.index.astype(str))
    
    comparison_df.plot.line(title="Fund Cumulative Return vs. HS300 Index")
    plt.ylabel("Cumulative Return")
    plt.show()


# --- Task 2: Volatility and Return ---

def calculate_volatility_and_returns(stock_filepath, fund_filepath):
    """
    Calculates volatility for a stock and annualized return for a fund.
    """
    print("\n--- Running Task 2: Volatility and Return Calculation ---")
    
    # 2.1: Convert price series to numpy arrays
    stock_df = pd.read_excel(stock_filepath)
    fund_df = pd.read_excel(fund_filepath, index_col=0).iloc[:, 0]
    
    stock_np = np.array(stock_df['close'])
    fund_np = np.array(fund_df)

    # 2.2: Calculate volatility (variance) for the stock
    stock_variance = stock_np.var()
    print(f"Stock Volatility (Variance of close price): {stock_variance:.6f}")

    # 2.3: Calculate annualized return for the fund
    # Note: The original formula seems incorrect. A standard approach is used here.
    total_return = (fund_np[-1] - fund_np[0]) / fund_np[0]
    num_days = len(fund_np)
    annualized_return = ((1 + total_return) ** (365.0 / num_days)) - 1
    print(f"Fund Annualized Return: {annualized_return:.4%}")


# --- Task 3: Data Visualization ---

def create_visualizations(fund_filepath, stock_dir_path):
    """
    Creates various financial plots: fund net value and stock correlation heatmap.
    """
    print("\n--- Running Task 3: Data Visualization ---")

    # 3.1: Create 2x2 subplot of 4 fund net values
    funds_df = pd.read_excel(fund_filepath, index_col=0)
    fund_bundle = funds_df.iloc[:100, :4].sort_index(ascending=True)
    fund_bundle.index = pd.to_datetime(fund_bundle.index.astype(str))
    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))
    fig.suptitle('Net Asset Value Trends for Four Funds')
    
    axes[0, 0].plot(fund_bundle.iloc[:, 0])
    axes[0, 0].set_title(fund_bundle.columns[0])
    
    axes[0, 1].plot(fund_bundle.iloc[:, 1])
    axes[0, 1].set_title(fund_bundle.columns[1])
    
    axes[1, 0].plot(fund_bundle.iloc[:, 2])
    axes[1, 0].set_title(fund_bundle.columns[2])
    
    axes[1, 1].plot(fund_bundle.iloc[:, 3])
    axes[1, 1].set_title(fund_bundle.columns[3])

    plt.show()

    # 3.2: Create a heatmap of the stock covariance matrix
    stock_files = os.listdir(stock_dir_path)[:5]
    close_prices = []
    for file in stock_files:
        stock_df = pd.read_excel(os.path.join(stock_dir_path, file))
        # Ensure all series have the same length by trimming to the shortest
        close_prices.append(stock_df['close'].values[:471])

    close_prices_np = np.vstack(close_prices)
    covariance_matrix = np.cov(close_prices_np)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(covariance_matrix, annot=True, fmt='.4f', 
                xticklabels=[f'Stock {i+1}' for i in range(5)], 
                yticklabels=[f'Stock {i+1}' for i in range(5)])
    plt.title("Covariance Matrix of Stock Close Prices")
    plt.show()


# --- Task 4: Stock Screening and Statistical Tests ---

def perform_screening_and_tests(stock_dir_path, fund_filepath):
    """
    Performs advanced analysis: Sharpe ratio screening, moving averages, and t-test.
    """
    print("\n--- Running Task 4: Stock Screening and Statistical Tests ---")
    
    # 4.1: Find the top 5 stocks of 2020 by Sharpe Ratio
    rf = 0.01  # Risk-free rate
    sharpe_ratios = []
    stock_files = os.listdir(stock_dir_path)

    for file in stock_files:
        df_tmp = pd.read_excel(os.path.join(stock_dir_path, file))
        df_2020 = df_tmp[(df_tmp['trade_date'] >= 20200101) & (df_tmp['trade_date'] < 20210101)]
        
        if not df_2020.empty:
            daily_returns = df_2020['pct_chg'] / 100 # Assuming pct_chg is in percent
            excess_returns = daily_returns - (rf / 252) # Daily risk-free rate
            
            # Calculate annualized Sharpe Ratio
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            sharpe_ratios.append((file[:-4], sharpe_ratio))

    # Sort stocks by Sharpe ratio and get the top 5
    top_5_stocks = sorted(sharpe_ratios, key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 Stocks of 2020 by Sharpe Ratio:")
    for stock, sr in top_5_stocks:
        print(f"- {stock} (Sharpe Ratio: {sr:.4f})")

    # 4.2: Plot moving averages for one stock
    stock1_df = pd.read_excel(os.path.join(stock_dir_path, stock_files[0]))
    stock1_df['trade_date'] = pd.to_datetime(stock1_df['trade_date'].astype(str))
    stock1_df = stock1_df.set_index('trade_date').sort_index()
    
    stock1_df['5-day MA'] = stock1_df['close'].rolling(5).mean()
    stock1_df['10-day MA'] = stock1_df['close'].rolling(10).mean()
    stock1_df['20-day MA'] = stock1_df['close'].rolling(20).mean()
    
    stock1_df[['close', '5-day MA', '10-day MA', '20-day MA']].plot(figsize=(12, 6))
    plt.title(f'Closing Price and Moving Averages for {stock_files[0][:-4]}')
    plt.ylabel('Price')
    plt.show()

    # 4.3: Perform t-test on two funds' daily returns
    funds_df = pd.read_excel(fund_filepath)
    fund1_returns = funds_df.iloc[:, 2].pct_change().dropna()
    fund2_returns = funds_df.iloc[:, 5].pct_change().dropna()
    
    # Check for variance equality (Levene's test)
    levene_stat, levene_p = stats.levene(fund1_returns, fund2_returns)
    print(f"\nLevene's test for equal variances: p-value = {levene_p:.4f}")
    
    # Perform independent t-test
    ttest_stat, ttest_p = stats.ttest_ind(fund1_returns, fund2_returns)
    print(f"Independent t-test for daily returns: p-value = {ttest_p:.4f}")
    if ttest_p < 0.05:
        print("Conclusion: The daily returns of the two funds are significantly different (p < 0.05).")
    else:
        print("Conclusion: There is no significant difference in the daily returns of the two funds (p >= 0.05).")


# --- Main execution block ---

if __name__ == '__main__':
    # Define file paths and directories
    # Note: Ensure these files and directories are set up as described in the README.
    FUND_FILE = "fund_info.xls"
    INDEX_FILE = "index_info.xls"
    STOCK_DIR = "stock_info_xls/"
    
    # Run all analysis tasks
    analyze_fund_performance(FUND_FILE, INDEX_FILE)
    calculate_volatility_and_returns(os.path.join(STOCK_DIR, os.listdir(STOCK_DIR)[0]), FUND_FILE)
    create_visualizations(FUND_FILE, STOCK_DIR)
    perform_screening_and_tests(STOCK_DIR, FUND_FILE)
