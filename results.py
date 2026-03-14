import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Results:
    """Handles printing VaR/CVaR metrics and rendering all plots."""

    def __init__(self, confidence_level: float = 0.95, output_dir: str = "output"):
        self.confidence_level = confidence_level
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._report_path = os.path.join(output_dir, "var_report.txt")
        # Clear/create the report file at the start of each run
        open(self._report_path, "w").close()

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_var_cvar(self, method_name: str, metrics: dict) -> None:
        """Print daily, monthly, and annual VaR/CVaR and append to the report file."""
        cl = self.confidence_level
        lines = [
            f"\n{'=' * 50}",
            f" {method_name}",
            f"{'=' * 50}",
            f"Daily   VaR  at {cl:.0%} confidence: ${metrics['daily_var']:>10.2f}",
            f"Monthly VaR  at {cl:.0%} confidence: ${metrics['monthly_var']:>10.2f}",
            f"Annual  VaR  at {cl:.0%} confidence: ${metrics['annual_var']:>10.2f}",
            f"Daily   CVaR at {cl:.0%} confidence: ${metrics['daily_cvar']:>10.2f}",
            f"Monthly CVaR at {cl:.0%} confidence: ${metrics['monthly_cvar']:>10.2f}",
            f"Annual  CVaR at {cl:.0%} confidence: ${metrics['annual_cvar']:>10.2f}",
        ]
        for line in lines:
            print(line)
        with open(self._report_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _save(self, filename: str) -> None:
        """Save the current figure to output_dir and close it."""
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()

    def plot_prices(self, prices: pd.DataFrame) -> None:
        """Line chart of adjusted close prices for each asset."""
        ax = prices.plot.line(
            xlabel="Time",
            ylabel="Adjusted Close Price",
            title="Asset Prices",
            figsize=(10, 6),
        )
        ax.grid(True)
        plt.tight_layout()
        self._save("prices.png")

    def plot_returns_distribution(self, port_returns: pd.Series) -> None:
        """Histogram of daily portfolio percentage returns."""
        plt.figure(figsize=(8, 5))
        plt.hist(port_returns, bins=20, density=True, alpha=0.6, color="steelblue")
        plt.xlabel("Portfolio Returns")
        plt.ylabel("Density")
        plt.title("Portfolio Daily Percentage Returns")
        plt.tight_layout()
        self._save("returns_distribution.png")

    def plot_mc_pct_change(self, sim_pct_change: np.ndarray) -> None:
        """Plot all Monte Carlo simulation paths (percentage change)."""
        plt.figure(figsize=(10, 6))
        plt.plot(sim_pct_change, alpha=0.3, linewidth=0.5)
        plt.axhline(
            np.percentile(sim_pct_change, 5),
            color="r", linestyle="dashed", linewidth=1.5,
            label="5th Percentile",
        )
        plt.axhline(
            np.percentile(sim_pct_change, 95),
            color="g", linestyle="dashed", linewidth=1.5,
            label="95th Percentile",
        )
        plt.axhline(
            np.mean(sim_pct_change),
            color="b", linestyle="solid", linewidth=1.5,
            label="Mean",
        )
        plt.legend()
        plt.ylabel("Portfolio Percentage Change")
        plt.xlabel("Days")
        plt.title("Monte Carlo Simulation — Portfolio Percentage Change")
        plt.tight_layout()
        self._save("mc_pct_change.png")

    def plot_mc_portfolio_values(
        self,
        sim_pct_change: np.ndarray,
        initial_portfolio: float,
    ) -> None:
        """
        Convert simulated percentage-change paths to portfolio values and plot.
        Marks the VaR and CVaR of the final-day distribution.
        """
        T, n_simulation = sim_pct_change.shape
        portfolio_values = np.zeros((T, n_simulation))
        for m in range(n_simulation):
            portfolio_values[:, m] = (
                np.cumprod(sim_pct_change[:, m] + 1) * initial_portfolio
            )

        final_values = portfolio_values[-1, :]
        mc_var = np.percentile(final_values, 100 * (1 - self.confidence_level))
        mc_cvar = final_values[final_values <= mc_var].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values, alpha=0.3, linewidth=0.5)
        plt.axhline(mc_var, color="r", linewidth=1.5, label=f"VaR: ${mc_var:,.2f}")
        plt.axhline(mc_cvar, color="g", linewidth=1.5, label=f"CVaR: ${mc_cvar:,.2f}")
        plt.legend(loc="upper left")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Days")
        plt.title("Monte Carlo Simulation — Portfolio Value")
        plt.tight_layout()
        self._save("mc_portfolio_values.png")
