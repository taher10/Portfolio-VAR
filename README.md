# Introduction
Value at Risk (VaR) is a widely used risk measure that estimates the maximum potential loss of a portfolio with a specified level of confidence over a defined period. The Conditional Value at Risk (CVaR), also known as Expected Shortfall, complements VaR by estimating the expected loss given that the portfolio’s loss exceeds the VaR threshold.

The project employs three distinct approaches to calculate both VaR and CVaR:

Historical Method: Based on past returns.
Parametric Method: Assumes a normal distribution of returns.
Monte Carlo Simulation: Uses random simulations to estimate risk.

# Data Source
The data for this project is obtained from Yahoo Finance using the yfinance Python package. We use daily adjusted closing prices for the three chosen ETFs to represent equity (SPY), fixed income (AGG), and gold (IAUM).

#Portfolio Overview
The portfolio is composed of three ETFs:

SPY: Represents the equity asset class.
AGG: Represents fixed income.
IAUM: Represents gold.
These asset classes are weighted equally in the portfolio to calculate the aggregate risk metrics. However, the project is flexible enough to accommodate different weightings for more realistic portfolio modeling.

# Risk Calculation Methods
1. Historical Method
In the Historical Method, we assume that future portfolio returns will follow the same distribution as past returns. The VaR is calculated by simply taking the 5th percentile of historical returns. The CVaR is then computed as the average loss beyond the VaR threshold.

Steps:

Calculate historical returns for the portfolio.
Find the 5th percentile of the historical returns as the VaR.
Calculate CVaR as the average of the worst losses beyond the VaR.
2. Parametric Method
The Parametric Method assumes that returns follow a normal distribution. VaR is determined by subtracting the product of the z-value corresponding to the confidence level and the standard deviation from the mean return. The CVaR is then calculated as the expected loss beyond the VaR level.

Steps:

Compute the mean and standard deviation of portfolio returns.
Use the z-score for the desired confidence interval (e.g., 95%) to calculate VaR.
Compute CVaR as the expected loss when losses exceed the VaR threshold.
3. Monte Carlo Simulation
In the Monte Carlo Simulation, random paths are generated using the Cholesky decomposition to maintain the correlation between the assets in the portfolio. The resulting paths simulate potential future returns, from which the VaR and CVaR are calculated.

Steps:

Generate random normal returns for each asset over a given period.
Use the Cholesky decomposition of the covariance matrix to simulate correlated returns.
Run multiple simulations to compute the VaR and CVaR based on simulated portfolio losses.

# Results
The project calculates the following metrics for the portfolio:

Value at Risk (VaR): The maximum potential loss with a given level of confidence.
Conditional Value at Risk (CVaR): The expected loss assuming the VaR threshold is breached.
Results from each method are compared to assess the consistency and robustness of each approach.

