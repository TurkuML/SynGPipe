# Gaussian copula synthesizer for tabular data.
# Independent implementation.
# Use of AI tools: this implementation was developed with the
# assistance of a generative AI tool.

"""
Gaussian copula synthesizer.

The model works in three steps:
  1. Learn the distribution of each column separately (its "marginal").
  2. Map every column to a standard normal scale and learn how the columns
     move together (a correlation matrix).
  3. To generate data, sample correlated normal values and map each one
     back to its original scale using the learned marginals.

"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, gamma, beta, gaussian_kde, rankdata
from sklearn.covariance import LedoitWolf

EPS = 1e-6  # keeps probabilities away from exactly 0 and 1

_PARAMETRIC_CANDIDATES = {
    "normal": norm,
    "lognormal": lognorm,
    "gamma": gamma,
    "beta": beta,
}


def _sorted_unique(values):
    """Unique values in sorted order (first-seen order if values are not sortable)."""
    uniques = pd.unique(pd.Series(values))
    try:
        return np.asarray(sorted(uniques), dtype=object)
    except TypeError:
        return np.asarray(uniques, dtype=object)


# ---------------------------------------------------------------------------
# Marginal distributions
# ---------------------------------------------------------------------------

class ContinuousMarginal:
    """
    Marginal distribution of a continuous variable.

    method:
      - "auto":       best parametric fit (by AIC); if it fits poorly
                      (Kolmogorov-Smirnov test), a kernel density estimate is used.
      - "parametric": best parametric fit; KDE only if no candidate can be fitted.
      - "kde":        always use a kernel density estimate.
    """

    def __init__(self, method="auto", ks_alpha=0.01, grid_size=1000):
        if method not in ("auto", "parametric", "kde"):
            raise ValueError("method must be 'auto', 'parametric' or 'kde'.")
        self.method = method
        self.ks_alpha = ks_alpha
        self.grid_size = grid_size

        self.kind = None               # "constant", "parametric" or "kde"
        self.distribution_name = None
        self.distribution = None
        self.params = None
        self.constant = None
        self.lower = None
        self.upper = None
        self.integer = False
        self._grid = None
        self._grid_cdf = None

    def fit(self, data, bounds=None):
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            raise ValueError("Cannot fit a continuous marginal on empty data.")

        if bounds is None:
            self.lower, self.upper = float(np.min(data)), float(np.max(data))
        else:
            self.lower, self.upper = float(bounds[0]), float(bounds[1])
            if self.lower > self.upper:
                raise ValueError(f"Invalid bounds {bounds}: lower > upper.")

        # If all training values are whole numbers, synthetic values are rounded too
        self.integer = bool(np.all(np.mod(data, 1) == 0))

        # Constant column: nothing to fit
        if np.ptp(data) == 0:
            self.kind = "constant"
            self.distribution_name = "constant"
            self.constant = data[0]
            return self

        if self.method in ("auto", "parametric"):
            self._fit_parametric(data)

            if self.kind == "parametric" and self.method == "auto":
                pvalue = stats.kstest(data, self.distribution.cdf, args=self.params).pvalue
                if pvalue < self.ks_alpha:
                    self.kind = None   # poor fit -> fall back to KDE

        if self.kind is None:
            self._fit_kde(data)

        return self

    def _fit_parametric(self, data):
        best_aic, best = np.inf, None

        for name, dist in _PARAMETRIC_CANDIDATES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    params = dist.fit(data)
                    loglik = np.sum(dist.logpdf(data, *params))
                except Exception:
                    continue

            # Skip degenerate fits (infinite or undefined likelihood)
            if not np.isfinite(loglik) or not np.all(np.isfinite(params)):
                continue

            aic = 2 * len(params) - 2 * loglik
            if aic < best_aic:
                best_aic, best = aic, (name, dist, params)

        if best is not None:
            self.kind = "parametric"
            self.distribution_name, self.distribution, self.params = best

    def _fit_kde(self, data):
        kde = gaussian_kde(data)
        bandwidth = float(np.sqrt(kde.covariance[0, 0]))

        # Tabulate the KDE's cumulative distribution on a fine grid
        grid = np.linspace(data.min() - 4 * bandwidth,
                           data.max() + 4 * bandwidth,
                           self.grid_size)
        cdf = np.array([kde.integrate_box_1d(-np.inf, g) for g in grid])
        cdf = np.maximum.accumulate(cdf)              # enforce monotonicity
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])     # rescale to [0, 1]

        self._grid, self._grid_cdf = grid, cdf
        self.kind = "kde"
        self.distribution_name = "kde"

    def cdf(self, x):
        """Value -> probability (cumulative distribution function)."""
        x = np.asarray(x, dtype=float)
        if self.kind == "constant":
            return np.full(x.shape, 0.5)
        if self.kind == "parametric":
            return self.distribution.cdf(x, *self.params)
        return np.interp(x, self._grid, self._grid_cdf)

    def ppf(self, u):
        """Probability -> value (inverse CDF). Clips to bounds; rounds integer columns."""
        u = np.clip(np.asarray(u, dtype=float), EPS, 1 - EPS)

        if self.kind == "constant":
            x = np.full(u.shape, self.constant, dtype=float)
        elif self.kind == "parametric":
            x = self.distribution.ppf(u, *self.params)
        else:
            x = np.interp(u, self._grid_cdf, self._grid)

        x = np.clip(x, self.lower, self.upper)
        if self.integer:
            x = np.round(x)
        return x


class DiscreteMarginal:
    """
    Empirical marginal for categorical (nominal) and ordinal variables.

    Categories are kept in sorted order, so for ordinal variables
    (e.g., 1 < 2 < 3 < 4 < 5) the natural order is preserved.
    """

    def __init__(self):
        self.categories = None
        self.probs = None
        self.cum_probs = None
        self._index = None

    def fit(self, data):
        data = pd.Series(data)
        self.categories = _sorted_unique(data)

        counts = data.value_counts(normalize=True)
        self.probs = np.array([counts.get(c, 0.0) for c in self.categories], dtype=float)
        self.cum_probs = np.cumsum(self.probs)
        self.cum_probs[-1] = 1.0                      # guard against rounding error
        self._index = {c: i for i, c in enumerate(self.categories)}
        return self

    def cdf(self, x, rng):
        """
        Map each category to a random point inside its own probability interval.
        This spreads tied records out instead of stacking them on one value.
        """
        idx = np.array([self._index[v] for v in x])
        upper = self.cum_probs[idx]
        lower = upper - self.probs[idx]
        return rng.uniform(lower, upper)

    def ppf(self, u):
        """Probability -> category."""
        idx = np.searchsorted(self.cum_probs, u, side="right")
        idx = np.minimum(idx, len(self.categories) - 1)
        return self.categories[idx]


# ---------------------------------------------------------------------------
# Gaussian Copula Synthesizer
# ---------------------------------------------------------------------------

class GaussianCopulaSynthesizer:
    """
    Gaussian copula synthesizer.

    Parameters
    ----------
    continuous_columns, categorical_columns, ordinal_columns : list of str
        Columns to model. Every column must be declared in exactly one list.
        Undeclared columns (e.g., IDs, dates) are NOT modeled and NOT returned,
        so real identifiers are never copied into the synthetic data.
    bounds : dict, optional
        {column: (lower, upper)} limits for continuous columns.
        Default: the training minimum and maximum.
    continuous_marginal : {"auto", "parametric", "kde"}
        How continuous marginals are fitted (see ContinuousMarginal).
    random_state : int, optional
        Seed for reproducible results.
    """

    def __init__(self,
                 continuous_columns=None,
                 categorical_columns=None,
                 ordinal_columns=None,
                 bounds=None,
                 continuous_marginal="auto",
                 random_state=None):

        self.continuous_columns = list(continuous_columns or [])
        self.categorical_columns = list(categorical_columns or [])
        self.ordinal_columns = list(ordinal_columns or [])
        self.bounds = dict(bounds or {})
        self.continuous_marginal = continuous_marginal
        self.random_state = random_state

        self.marginals = {}
        self.columns = None
        self.dropped_columns = []
        self.correlation = None
        self.rng = None
        self._dtypes = {}

    # ----- Validation -----

    def _select_columns(self, df):
        declared = self.continuous_columns + self.categorical_columns + self.ordinal_columns

        if not declared:
            raise ValueError("No columns declared. Provide continuous, categorical "
                             "and/or ordinal column lists.")

        duplicates = sorted({c for c in declared if declared.count(c) > 1})
        if duplicates:
            raise ValueError(f"Columns declared in more than one list: {duplicates}")

        missing = [c for c in declared if c not in df.columns]
        if missing:
            raise ValueError(f"Declared columns not found in the data: {missing}")

        self.dropped_columns = [c for c in df.columns if c not in declared]
        if self.dropped_columns:
            warnings.warn(
                f"Columns not declared as continuous, categorical or ordinal are "
                f"excluded from the synthetic data: {self.dropped_columns}"
            )

        self.columns = [c for c in df.columns if c in declared]

        if df[self.columns].isnull().any().any():
            raise ValueError("The data contains missing values. Remove or impute them before fitting.")
        if len(df) < 2:
            raise ValueError("At least 2 rows are required to fit the model.")

    # ----- Fitting -----

    def fit(self, df):
        self.rng = np.random.default_rng(self.random_state)
        self._select_columns(df)
        self.marginals = {}

        n_rows = len(df)
        latent = np.empty((n_rows, len(self.columns)))

        for j, col in enumerate(self.columns):
            data = df[col].to_numpy()
            self._dtypes[col] = df[col].dtype

            if col in self.continuous_columns:
                marginal = ContinuousMarginal(method=self.continuous_marginal)
                marginal.fit(data, bounds=self.bounds.get(col))
                # Rank-based probabilities (pseudo-observations): robust to marginal misfit
                u = rankdata(data.astype(float)) / (n_rows + 1)
            else:
                # Ordinal and categorical: empirical distribution of the observed values
                marginal = DiscreteMarginal().fit(data)
                u = marginal.cdf(data, self.rng)

            self.marginals[col] = marginal
            latent[:, j] = norm.ppf(np.clip(u, EPS, 1 - EPS))

        self.correlation = self._estimate_correlation(latent)
        return self

    @staticmethod
    def _estimate_correlation(latent):
        if latent.shape[1] == 1:
            return np.ones((1, 1))

        cov = LedoitWolf().fit(latent).covariance_

        # Convert covariance to correlation (diagonal = 1)
        sd = np.sqrt(np.diag(cov))
        sd[sd == 0] = 1.0
        corr = cov / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)
        return corr

    # ----- Sampling -----

    def sample(self, n_samples):
        if self.correlation is None:
            raise RuntimeError("The model is not fitted. Call fit() before sample().")

        n_samples = int(n_samples)
        dim = len(self.columns)

        z = self.rng.multivariate_normal(np.zeros(dim), self.correlation,
                                         size=n_samples, method="eigh")
        u = norm.cdf(z)

        synthetic = {}
        for j, col in enumerate(self.columns):
            values = self.marginals[col].ppf(u[:, j])
            synthetic[col] = self._restore_dtype(values, col)

        return pd.DataFrame(synthetic, columns=self.columns)

    def _restore_dtype(self, values, col):
        series = pd.Series(values)
        try:
            return series.astype(self._dtypes[col])
        except (TypeError, ValueError):
            return series

    # ----- Inspection -----

    def describe_marginals(self):
        """Summary of the distribution fitted to each column."""
        rows = []
        for col in self.columns:
            m = self.marginals[col]
            if isinstance(m, ContinuousMarginal):
                rows.append({"column": col, "type": "continuous",
                             "distribution": m.distribution_name,
                             "range": f"[{m.lower:g}, {m.upper:g}]"})
            else:
                kind = "ordinal" if col in self.ordinal_columns else "categorical"
                rows.append({"column": col, "type": kind,
                             "distribution": "empirical",
                             "range": f"{len(m.categories)} categories"})
        return pd.DataFrame(rows)
