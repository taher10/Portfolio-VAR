import datetime
import numpy as np

from data_ingestion import DataIngestion
from model import VaRModel
from results import Results


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "AGG", "IAUM"]

# yfinance returns columns in alphabetical order: AGG, IAUM, SPY
# Weights must align with that order: AGG=30%, IAUM=20%, SPY=50%
WEIGHTS = np.array([0.30, 0.20, 0.50])

START_DATE = datetime.datetime(2021, 6, 30)
END_DATE = datetime.datetime(2024, 6, 30)

INITIAL_PORTFOLIO = 100_000
CONFIDENCE_LEVEL = 0.95
OUTPUT_DIR = "output"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Data Ingestion
    ingestion = DataIngestion(TICKERS, WEIGHTS, START_DATE, END_DATE)
    prices = ingestion.fetch_data()
    returns, port_returns = ingestion.compute_returns()

    # 2. Visualise raw data
    vis = Results(CONFIDENCE_LEVEL, OUTPUT_DIR)
    vis.plot_prices(prices)
    vis.plot_returns_distribution(port_returns)

    # 3. VaR Models
    model = VaRModel(returns, port_returns, WEIGHTS, INITIAL_PORTFOLIO, CONFIDENCE_LEVEL)

    # Historical
    hist_metrics = model.historical_var()
    vis.print_var_cvar("Historical Method", hist_metrics)

    # Parametric
    param_metrics = model.parametric_var()
    vis.print_var_cvar("Parametric Method", param_metrics)

    # Monte Carlo
    mc_metrics, sim_pct_change = model.monte_carlo_var(n_simulation=1000, T=252)
    vis.print_var_cvar("Monte Carlo Method", mc_metrics)
    vis.plot_mc_pct_change(sim_pct_change)
    vis.plot_mc_portfolio_values(sim_pct_change, INITIAL_PORTFOLIO)

    print(f"\nResults saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
