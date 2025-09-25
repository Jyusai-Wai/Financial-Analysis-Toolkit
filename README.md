# Python Financial Analysis Toolkit

This project is a collection of Python scripts designed to perform various financial analyses on stock and fund data. It leverages popular data science libraries like Pandas, NumPy, and Matplotlib to calculate key metrics, visualize data, and perform statistical tests. This toolkit serves as a practical example of applying Python programming to solve real-world financial problems.

## Features

This toolkit is organized into four main analysis tasks:

### 1. Fund Performance Analysis
-   Calculates the cumulative return of selected funds over a specific period.
-   Plots a comparative chart of a fund's cumulative return against a benchmark index (e.g., CSI 300).

### 2. Volatility and Return Calculation
-   Computes the volatility (variance) of a stock's closing price.
-   Calculates the annualized return of a fund.

### 3. Data Visualization
-   Generates a 2x2 subplot grid displaying the net value trends of four different funds.
-   Calculates the daily returns of five stocks and visualizes their covariance matrix as a heatmap, showing the correlation between them.

### 4. Stock Screening and Statistical Testing
-   **Sharpe Ratio Calculation**: Screens a portfolio of 50 stocks to find the top 5 performers in a given year based on the highest Sharpe Ratio.
-   **Moving Averages**: Plots the closing price of a stock along with its 5-day, 10-day, and 20-day moving averages.
-   **Hypothesis Testing**: Performs an independent two-sample t-test to determine if the daily returns of two funds are significantly different from each other.

## Required Libraries

Ensure you have the following Python libraries installed. You can install them using pip:
```bash
pip install pandas numpy scipy matplotlib seaborn openpyxl
