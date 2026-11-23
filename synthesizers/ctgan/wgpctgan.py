# WGP: CTGAN — CTGAN synthesizer trained with the WGAN-GP loss.
# Implements the CTGAN architecture (Xu et al., 2019) and follows the WGAN-GP
# training schedule (Gulrajani et al., 2017) of several critic updates per generator update.
# Use of AI tools: this implementation was developed with the
# assistance of a generative AI tool. 
#
# References:
#   - Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K. (2019).
#     Modeling Tabular Data using Conditional GAN. NeurIPS 2019.
#   - Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs. NeurIPS 2017.
#   - Lin, Z. et al. (2018). PacGAN: The power of two samples in generative
#     adversarial networks. NeurIPS 2018.

"""
CTGAN synthesizer with WGAN-GP loss.

Notes and limitations:
  - This synthesizer is NOT differentially private and can memorize records
    from small training sets. Use the Privacy report (DCR) to check. 
  - Continuous values are clipped to the given bounds (default: the training
    minimum and maximum). Training values outside the bounds are clipped before
    fitting. Using the training min/max reveals the real extreme values.
  - Ordinal columns are modeled as discrete categories (as in CTGAN).
  - Columns not declared as continuous, categorical or ordinal (e.g., IDs,
    dates) are excluded, so real identifiers are never copied.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_unique(values):
    """Unique values in sorted order (first-seen order if not sortable)."""
    uniques = pd.unique(pd.Series(values))
    try:
        return np.asarray(sorted(uniques), dtype=object)
    except TypeError:
        return np.asarray(uniques, dtype=object)


def _restore_dtype(values, dtype):
    series = pd.Series(values)
    try:
        return series.astype(dtype)
    except (TypeError, ValueError):
        return series


# ---------------------------------------------------------------------------
# 1. Data transformer (mode-specific normalization + one-hot encoding)
# ---------------------------------------------------------------------------

class DataTransformer:
    """
    Converts a table into the numeric format used by CTGAN.

    Continuous column -> [alpha, one-hot mode]
        A Bayesian Gaussian mixture finds the "modes" (peaks) of the column.
        Each value is represented by the mode it belongs to and its scaled
        position inside that mode (alpha, in [-1, 1]).
        Values are clipped to the column bounds (user bounds or training min/max).
    Discrete column   -> one-hot vector of its categories.
    """

    def __init__(self, max_modes=10, weight_threshold=0.005, random_state=None, bounds=None):
        self.max_modes = max_modes
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        # {column: (lower, upper)}; columns not listed use the training min/max
        self.bounds = dict(bounds or {})

    # ----- fitting -----

    def fit(self, df, continuous_columns):
        self.columns = list(df.columns)
        self.meta = []

        for col in self.columns:
            x = df[col].to_numpy()
            if col in continuous_columns:
                m = self._fit_continuous(col, x.astype(float))
            else:
                m = self._fit_discrete(col, x)
            m["dtype"] = df[col].dtype
            self.meta.append(m)

        # (width, activation) of every block in the transformed data
        self.output_info = [span for m in self.meta for span in m["spans"]]
        self.output_dim = sum(w for w, _ in self.output_info)

        # (start, width) of every discrete column block, used by the sampler
        self.discrete_spans = []
        start = 0
        for m in self.meta:
            if m["type"] == "discrete":
                self.discrete_spans.append((start, m["spans"][0][0]))
            start += sum(w for w, _ in m["spans"])
        return self

    def _resolve_bounds(self, col, x, integer):
        """(lower, upper) for a continuous column: user bounds if given, else training min/max.
        For integer columns the bounds are shrunk to whole numbers (e.g. 39.5-85.5 -> 40-85),
        so a rounded synthetic value can never fall outside them."""
        if col in self.bounds:
            lower, upper = (float(v) for v in self.bounds[col])
        else:
            lower, upper = float(x.min()), float(x.max())

        if integer:
            lower, upper = float(np.ceil(lower)), float(np.floor(upper))
            if lower > upper:
                raise ValueError(f"Bounds for '{col}' contain no whole number, "
                                 f"but the column only has whole numbers.")
        return lower, upper

    def _fit_continuous(self, col, x):
        # Decide integer-ness on the original values, before clipping
        integer = bool(np.all(np.mod(x, 1) == 0))
        lower, upper = self._resolve_bounds(col, x, integer)

        # Training values outside the bounds are clipped (same rule as the DP bounds)
        n_clipped = int(np.sum((x < lower) | (x > upper)))
        if n_clipped:
            warnings.warn(f"{n_clipped} training value(s) of '{col}' are outside the bounds "
                          f"[{lower:g}, {upper:g}] and were clipped before fitting.")
        x = np.clip(x, lower, upper)

        n_modes = int(min(self.max_modes, len(np.unique(x))))
        gm = BayesianGaussianMixture(
            n_components=n_modes,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=200,
            n_init=1,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # convergence warnings on small data
            gm.fit(x.reshape(-1, 1))

        # Keep only modes with a meaningful weight
        valid = gm.weights_ > self.weight_threshold
        if not valid.any():
            valid[np.argmax(gm.weights_)] = True

        return {
            "name": col, "type": "continuous", "gm": gm, "valid": valid,
            "means": gm.means_.reshape(-1)[valid],
            "stds": np.sqrt(gm.covariances_.reshape(-1))[valid],
            "lower": lower, "upper": upper,
            "integer": integer,
            "spans": [(1, "tanh"), (int(valid.sum()), "softmax")],
        }

    def _fit_discrete(self, col, x):
        categories = _sorted_unique(x)
        return {
            "name": col, "type": "discrete", "categories": categories,
            "index": {c: i for i, c in enumerate(categories)},
            "spans": [(len(categories), "softmax")],
        }

    # ----- forward transform -----

    def transform(self, df):
        parts = []
        for m in self.meta:
            x = df[m["name"]].to_numpy()
            if m["type"] == "continuous":
                parts.append(self._transform_continuous(m, x.astype(float)))
            else:
                parts.append(self._transform_discrete(m, x))
        return np.column_stack(parts).astype(np.float32)

    def _transform_continuous(self, m, x):
        x = np.clip(x, m["lower"], m["upper"])       # same clipping as in fit
        n = len(x)
        probs = m["gm"].predict_proba(x.reshape(-1, 1))[:, m["valid"]] + 1e-6
        probs /= probs.sum(axis=1, keepdims=True)

        # Sample one mode per value, in proportion to how likely it belongs to it
        cum = probs.cumsum(axis=1)
        modes = (self.rng.random((n, 1)) > cum).sum(axis=1)
        modes = np.minimum(modes, probs.shape[1] - 1)

        alpha = (x - m["means"][modes]) / (4 * m["stds"][modes])
        alpha = np.clip(alpha, -0.99, 0.99)

        onehot = np.zeros((n, probs.shape[1]))
        onehot[np.arange(n), modes] = 1
        return np.column_stack([alpha, onehot])

    def _transform_discrete(self, m, x):
        codes = np.array([m["index"][v] for v in x])
        onehot = np.zeros((len(x), len(m["categories"])))
        onehot[np.arange(len(x)), codes] = 1
        return onehot

    # ----- inverse transform -----

    def inverse_transform(self, data):
        out, st = {}, 0
        for m in self.meta:
            if m["type"] == "continuous":
                k = m["spans"][1][0]
                alpha = np.clip(data[:, st], -1, 1)
                modes = np.argmax(data[:, st + 1:st + 1 + k], axis=1)
                x = alpha * 4 * m["stds"][modes] + m["means"][modes]
                x = np.clip(x, m["lower"], m["upper"])
                if m["integer"]:
                    x = np.round(x)
                st += 1 + k
            else:
                k = len(m["categories"])
                x = m["categories"][np.argmax(data[:, st:st + k], axis=1)]
                st += k
            out[m["name"]] = _restore_dtype(x, m["dtype"])
        return pd.DataFrame(out, columns=self.columns)


# ---------------------------------------------------------------------------
# 2. Conditional sampler (conditional vector + training-by-sampling)
# ---------------------------------------------------------------------------

class ConditionalSampler:
    """
    During training, picks a discrete column and one of its categories,
    builds the matching conditional vector, and returns real rows that
    have that category. Rare categories are sampled more often
    (log-frequency), so the generator learns them too.
    """

    def __init__(self, data, discrete_spans, rng, log_frequency=True):
        self.data = data
        self.rng = rng
        self.n_discrete = len(discrete_spans)
        widths = [w for _, w in discrete_spans]
        self.cond_dim = int(sum(widths))
        self.cond_offsets = np.cumsum([0] + widths)[:-1].astype(int)

        self.rows_by_category, self.p_train, self.p_original = [], [], []
        for st, w in discrete_spans:
            codes = data[:, st:st + w].argmax(axis=1)
            self.rows_by_category.append([np.flatnonzero(codes == c) for c in range(w)])
            counts = np.bincount(codes, minlength=w).astype(float)
            freq = np.log(counts + 1) if log_frequency else counts
            self.p_train.append(freq / freq.sum())
            self.p_original.append(counts / counts.sum())

    def _build(self, batch, probabilities):
        cols = self.rng.integers(0, self.n_discrete, size=batch)
        cats = np.array([self.rng.choice(len(probabilities[c]), p=probabilities[c])
                         for c in cols])
        cond = np.zeros((batch, self.cond_dim), dtype=np.float32)
        cond[np.arange(batch), self.cond_offsets[cols] + cats] = 1
        return cond, cols, cats

    def sample_condvec(self, batch):
        """Conditional vector for training (log-frequency sampling)."""
        if self.n_discrete == 0:
            return None
        cond, cols, cats = self._build(batch, self.p_train)
        mask = np.zeros((batch, self.n_discrete), dtype=np.float32)
        mask[np.arange(batch), cols] = 1
        return cond, mask, cols, cats

    def sample_original_condvec(self, batch):
        """Conditional vector for generation (real frequencies)."""
        if self.n_discrete == 0:
            return None
        cond, _, _ = self._build(batch, self.p_original)
        return cond

    def sample_rows(self, batch, cols=None, cats=None):
        """Real rows; if cols/cats are given, rows having those categories."""
        if cols is None:
            idx = self.rng.integers(0, len(self.data), size=batch)
        else:
            idx = np.array([self.rng.choice(self.rows_by_category[c][k])
                            for c, k in zip(cols, cats)])
        return self.data[idx]


# ---------------------------------------------------------------------------
# 3. Networks
# ---------------------------------------------------------------------------

class _Residual(nn.Module):
    """Linear -> BatchNorm -> ReLU, concatenated with its input."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return torch.cat([F.relu(self.bn(self.fc(x))), x], dim=1)


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers, dim = [], input_dim
        for h in hidden_dims:
            layers.append(_Residual(dim, h))
            dim += h
        layers.append(nn.Linear(dim, output_dim))   # raw output; activations applied later
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """PacGAN critic: judges groups of `pac` records together."""

    def __init__(self, input_dim, hidden_dims, pac):
        super().__init__()
        self.pac_dim = input_dim * pac
        layers, dim = [], self.pac_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(-1, self.pac_dim))


# ---------------------------------------------------------------------------
# 4. CTGAN synthesizer with WGAN-GP loss
# ---------------------------------------------------------------------------

class CTGANSynthesizer:
    """
    Parameters
    ----------
    epochs : int                Number of passes over the training data.
    batch_size : int            Rows per training step (adjusted to the data size).
    embedding_dim : int         Size of the random noise vector.
    generator_dims, critic_dims : tuple of int   Hidden layer sizes.
    generator_lr, critic_lr : float              Adam learning rates.
    weight_decay : float        Adam weight decay.
    critic_steps : int          Critic updates per generator update
                                (CTGAN default 1; classic WGAN-GP uses 5).
    pac : int                   Records per PacGAN group.
    gp_lambda : float           Gradient penalty weight.
    log_frequency : bool        Oversample rare categories during training.
    max_modes : int             Max mixture modes per continuous column.
    random_state : int          Seed for reproducible results.
    device : str                "cpu" or "cuda" (auto-detected if None).
    verbose : bool              Show a progress bar.
    """

    def __init__(self, epochs=500, batch_size=500, embedding_dim=128,
                 generator_dims=(256, 256), critic_dims=(256, 256),
                 generator_lr=2e-4, critic_lr=2e-4, weight_decay=1e-6,
                 critic_steps=5, pac=10, gp_lambda=10.0, log_frequency=True,
                 max_modes=10, random_state=None, device=None, verbose=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dims = generator_dims
        self.critic_dims = critic_dims
        self.generator_lr = generator_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.critic_steps = critic_steps
        self.pac = pac
        self.gp_lambda = gp_lambda
        self.log_frequency = log_frequency
        self.max_modes = max_modes
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.generator = None
        self.critic = None
        self.transformer = None
        self.sampler = None
        self.bounds = {}
        self.dropped_columns = []
        self.loss_history = []

    # ----- validation -----

    def _select_columns(self, data, continuous, discrete):
        declared = continuous + discrete
        if not declared:
            raise ValueError("No columns declared. Provide continuous, categorical "
                             "and/or ordinal column lists.")
        duplicates = sorted({c for c in declared if declared.count(c) > 1})
        if duplicates:
            raise ValueError(f"Columns declared in more than one list: {duplicates}")
        missing = [c for c in declared if c not in data.columns]
        if missing:
            raise ValueError(f"Declared columns not found in the data: {missing}")

        self.dropped_columns = [c for c in data.columns if c not in declared]
        if self.dropped_columns:
            warnings.warn("Columns not declared as continuous, categorical or ordinal are "
                          f"excluded from the synthetic data: {self.dropped_columns}")

        df = data[[c for c in data.columns if c in declared]]
        if df.isnull().any().any():
            raise ValueError("The data contains missing values. Remove or impute them before fitting.")
        if len(df) < 2:
            raise ValueError("At least 2 rows are required to fit the model.")
        return df
        
    def _validate_bounds(self, bounds, continuous):
        """Check {column: (lower, upper)}: continuous columns only, finite numbers, lower <= upper."""
        clean = {}
        for col, value in dict(bounds or {}).items():
            if col not in continuous:
                raise ValueError(f"Bounds given for '{col}', which is not a continuous column.")
            try:
                lower, upper = (float(v) for v in value)
            except (TypeError, ValueError):
                raise ValueError(f"Bounds for '{col}' must be a pair (lower, upper), got {value!r}.")
            if not (np.isfinite(lower) and np.isfinite(upper)):
                raise ValueError(f"Bounds for '{col}' must be finite numbers.")
            if lower > upper:
                raise ValueError(f"Invalid bounds for '{col}': lower ({lower}) > upper ({upper}).")
            clean[col] = (lower, upper)
        return clean

    # ----- building blocks -----

    def _activate(self, raw):
        """tanh for alpha values, Gumbel-softmax for one-hot blocks."""
        outs, st = [], 0
        for width, activation in self.transformer.output_info:
            block = raw[:, st:st + width]
            outs.append(torch.tanh(block) if activation == "tanh"
                        else F.gumbel_softmax(block, tau=0.2))
            st += width
        return torch.cat(outs, dim=1)

    def _cond_loss(self, raw, cond, mask):
        """Cross-entropy: did the generator produce the requested category?"""
        losses = []
        for j, (st, width) in enumerate(self.transformer.discrete_spans):
            c_st = int(self.sampler.cond_offsets[j])
            target = cond[:, c_st:c_st + width].argmax(dim=1)
            losses.append(F.cross_entropy(raw[:, st:st + width], target, reduction="none"))
        losses = torch.stack(losses, dim=1)
        return (losses * mask).sum() / raw.size(0)

    def _gradient_penalty(self, real, fake):
        pac, dim = self._pac, real.size(1)
        alpha = torch.rand(real.size(0) // pac, 1, 1, device=self.device)
        alpha = alpha.repeat(1, pac, dim).view(-1, dim)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        scores = self.critic(interpolates)
        grads = torch.autograd.grad(scores, interpolates, torch.ones_like(scores),
                                    create_graph=True, retain_graph=True)[0]
        grads = grads.view(-1, pac * dim)
        return self.gp_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def _tensor(self, array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    # ----- training -----

    def fit(self, data, continuous_columns=None, categorical_columns=None, ordinal_columns=None, bounds=None):
        continuous = list(continuous_columns or [])
        discrete = list(categorical_columns or []) + list(ordinal_columns or [])
        df = self._select_columns(data, continuous, discrete)
        self.bounds = self._validate_bounds(bounds, continuous)

        self.rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Preprocess
        self.transformer = DataTransformer(self.max_modes, random_state=self.random_state,
                                           bounds=self.bounds)
        self.transformer.fit(df, continuous)
        train = self.transformer.transform(df)
        self.sampler = ConditionalSampler(train, self.transformer.discrete_spans,
                                          self.rng, self.log_frequency)

        # Batch size must be a multiple of pac; use pac=1 for very small data
        n = len(train)
        self._pac = self.pac if n >= 2 * self.pac else 1
        bs = min(self.batch_size, n)
        self._batch = max(bs - bs % self._pac, 2 * self._pac)
        steps_per_epoch = max(n // self._batch, 1)

        data_dim = self.transformer.output_dim
        cond_dim = self.sampler.cond_dim

        self.generator = Generator(self.embedding_dim + cond_dim,
                                   self.generator_dims, data_dim).to(self.device)
        self.critic = Critic(data_dim + cond_dim, self.critic_dims, self._pac).to(self.device)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr,
                                 betas=(0.5, 0.9), weight_decay=self.weight_decay)
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr,
                                 betas=(0.5, 0.9), weight_decay=self.weight_decay)

        epochs = range(self.epochs)
        if self.verbose:
            epochs = tqdm(epochs, desc="Training CTGAN (WGAN-GP)")

        bs = self._batch
        self.loss_history = []

        for _ in epochs:
            for _ in range(steps_per_epoch):

                # --- Critic update(s) ---
                for _ in range(self.critic_steps):
                    z = torch.randn(bs, self.embedding_dim, device=self.device)
                    condvec = self.sampler.sample_condvec(bs)

                    if condvec is None:
                        real = self._tensor(self.sampler.sample_rows(bs))
                        fake = self._activate(self.generator(z))
                        real_in, fake_in = real, fake
                    else:
                        cond, _, cols, cats = condvec
                        perm = self.rng.permutation(bs)
                        real = self._tensor(self.sampler.sample_rows(bs, cols[perm], cats[perm]))
                        c_fake, c_real = self._tensor(cond), self._tensor(cond[perm])
                        fake = self._activate(self.generator(torch.cat([z, c_fake], dim=1)))
                        real_in = torch.cat([real, c_real], dim=1)
                        fake_in = torch.cat([fake, c_fake], dim=1)

                    fake_in = fake_in.detach()
                    loss_c = self.critic(fake_in).mean() - self.critic(real_in).mean()
                    gp = self._gradient_penalty(real_in, fake_in)

                    opt_c.zero_grad(set_to_none=True)
                    (loss_c + gp).backward()
                    opt_c.step()

                # --- Generator update ---
                z = torch.randn(bs, self.embedding_dim, device=self.device)
                condvec = self.sampler.sample_condvec(bs)

                if condvec is None:
                    raw = self.generator(z)
                    scores = self.critic(self._activate(raw))
                    cond_loss = torch.zeros((), device=self.device)
                else:
                    cond, mask, _, _ = condvec
                    c, m = self._tensor(cond), self._tensor(mask)
                    raw = self.generator(torch.cat([z, c], dim=1))
                    scores = self.critic(torch.cat([self._activate(raw), c], dim=1))
                    cond_loss = self._cond_loss(raw, c, m)

                loss_g = -scores.mean() + cond_loss

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()

            self.loss_history.append({"critic": loss_c.item(), "generator": loss_g.item()})
            if self.verbose:
                epochs.set_postfix(critic=f"{loss_c.item():.3f}", generator=f"{loss_g.item():.3f}")

        return self

    # ----- sampling -----

    def sample(self, n_samples):
        if self.generator is None:
            raise RuntimeError("The model is not fitted. Call fit() before sample().")

        n_samples = int(n_samples)
        self.generator.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, n_samples, self._batch):
                b = min(self._batch, n_samples - start)
                z = torch.randn(b, self.embedding_dim, device=self.device)
                cond = self.sampler.sample_original_condvec(b)
                if cond is not None:
                    z = torch.cat([z, self._tensor(cond)], dim=1)
                chunks.append(self._activate(self.generator(z)).cpu().numpy())
        self.generator.train()

        return self.transformer.inverse_transform(np.concatenate(chunks)[:n_samples])
