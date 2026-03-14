import numpy as np
import pandas as pd
from scipy.stats import norm


class VaRModel:
    """
    Value-at-Risk and Conditional Value-at-Risk model.

    Supports three estimation methods:
      - Historical simulation
      - Parametric (variance-covariance / normal distribution)
      - Monte Carlo simulation
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        port_returns: pd.Series,
        weights: np.ndarray,
        initial_portfolio: float = 100_000,
        confidence_level: float = 0.95,
    ):
        self.returns = returns
        self.port_returns = port_returns
        self.weights = weights
        self.initial_portfolio = initial_portfolio
        self.confidence_level = confidence_level

        self.cov_matrix: pd.DataFrame = returns.cov()
        self.mean_returns: pd.Series = returns.mean()

    # ------------------------------------------------------------------
    # Public estimation methods
    # ------------------------------------------------------------------

    def historical_var(self) -> dict:
        """Estimate VaR/CVaR using the historical simulation method."""
        var = self.port_returns.quantile(q=1 - self.confidence_level)
        cvar = self.port_returns[self.port_returns <= var].mean()
        return self._scale(var, cvar)

    def parametric_var(self) -> dict:
        """Estimate VaR/CVaR using the parametric (normal) method."""
        mean = self.port_returns.mean()
        port_std_dev = np.sqrt(self.weights.T @ self.cov_matrix @ self.weights)
        z_value = norm.ppf(q=1 - self.confidence_level)
        var = -(mean - port_std_dev * z_value)
        cvar = self.port_returns[self.port_returns <= var].mean()
        return self._scale(var, cvar)

    def monte_carlo_var(
        self,
        n_simulation: int = 1000,
        T: int = 252,
        seed: int = 123,
    ) -> tuple[dict, np.ndarray]:
        """
        Estimate VaR/CVaR using Monte Carlo simulation with Cholesky decomposition.

        Returns
        -------
        metrics : dict
            Scaled VaR/CVaR values (daily, monthly, annual).
        sim_pct_change : np.ndarray, shape (T, n_simulation)
            Simulated daily portfolio percentage changes across all paths.
        """
        np.random.seed(seed)

        # Mean matrix: shape (n_assets, T)
        mean_matrix = np.full(
            shape=(T, len(self.weights)), fill_value=self.mean_returns
        ).T

        L = np.linalg.cholesky(self.cov_matrix)
        sim_pct_change = np.zeros((T, n_simulation))

        for m in range(n_simulation):
            Z = np.random.normal(size=(T, len(self.weights)))
            daily_pct_change = mean_matrix + L @ Z.T  # shape: (n_assets, T)
            sim_pct_change[:, m] = self.weights @ daily_pct_change

        # VaR/CVaR from the final-day distribution
        final_day_returns = pd.Series(sim_pct_change[-1, :])
        var = final_day_returns.quantile(1 - self.confidence_level)
        cvar = final_day_returns[final_day_returns <= var].mean()

        return self._scale(var, cvar), sim_pct_change

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale(self, var: float, cvar: float) -> dict:
        """Convert daily VaR/CVaR to monetary terms and scale to monthly/annual."""
        var_value = var * self.initial_portfolio
        cvar_value = cvar * self.initial_portfolio
        return {
            "daily_var": var_value,
            "monthly_var": var_value * np.sqrt(21),
            "annual_var": var_value * np.sqrt(252),
            "daily_cvar": cvar_value,
            "monthly_cvar": cvar_value * np.sqrt(21),
            "annual_cvar": cvar_value * np.sqrt(252),
        }
