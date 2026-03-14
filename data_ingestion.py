import numpy as np
import pandas as pd
import yfinance as yf
import datetime


class DataIngestion:
    """Handles downloading price data and computing portfolio returns."""

    def __init__(
        self,
        tickers: list[str],
        weights: np.ndarray,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
    ):
        # yfinance returns columns in alphabetical order regardless of the input order.
        # Weights must correspond to the alphabetically-sorted ticker list.
        # Default sorted order for ['SPY', 'AGG', 'IAUM'] -> ['AGG', 'IAUM', 'SPY']
        # So weights = [AGG_weight, IAUM_weight, SPY_weight]
        self.tickers = tickers
        self.weights = weights
        self.start_date = start_date
        self.end_date = end_date

        self.prices: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.port_returns: pd.Series | None = None

    def fetch_data(self) -> pd.DataFrame:
        """Download adjusted close prices for all tickers."""
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date)["Close"]
        df.index = pd.to_datetime(df.index)
        self.prices = df.dropna()
        return self.prices

    def compute_returns(self) -> tuple[pd.DataFrame, pd.Series]:
        """Compute individual and weighted portfolio daily percentage returns."""
        if self.prices is None:
            raise RuntimeError("Call fetch_data() before compute_returns().")
        self.returns = self.prices.pct_change().dropna()
        self.port_returns = self.returns.dot(self.weights)
        return self.returns, self.port_returns
