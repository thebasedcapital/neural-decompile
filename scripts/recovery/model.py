"""Causal linear decoding, compact repairs, and matched recalibration baselines."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np


REPAIR_METHOD = "three_source_separate_bias_rank_6_repair"
REPAIR_RANK = 6

METHODS = [
    {"id": "frozen", "label": "Frozen decoder", "description": "Earlier-day ridge weights behind the shared label-free running z-score; no new-day labels."},
    {"id": "recenter", "label": "Neural recentering", "description": "Additional calibration-prefix mean/variance adjustment of the z-scored features; no new labels."},
    {"id": "scratch", "label": "New-day ridge", "description": "Fit all coefficients from the same calibration prefix."},
    {"id": "anchored", "label": "Prior-anchored ridge", "description": "Fit a full residual correction with a quadratic prior around the frozen decoder."},
    {"id": "output_affine", "label": "Output recalibration", "description": "Affine correction of the seven frozen predictions."},
    {"id": "gain_repair", "label": "Channel-gain repair", "description": "One coefficient gain per neural channel plus seven output offsets; preserve each channel's seven-output direction."},
    {"id": "drift_basis", "label": "Earlier-day drift repair", "description": "Fit a small combination of coefficient corrections learned exclusively from earlier recording days."},
    {"id": REPAIR_METHOD, "label": "Recent-source rank-6 repair", "description": "Average the latest three source-day corrections, then fit rank-six residual slopes and a separately regularized seven-output bias."},
]


def causal_features(counts, bin_ms=20, tau_ms=240):
    """Exactly the finite causal exponential kernel used by the H1 reference."""
    kernel = np.exp(-np.arange(0, tau_ms, bin_ms, dtype=float) / tau_ms)
    kernel /= kernel.sum()
    out = np.zeros_like(counts, dtype=np.float64)
    for lag, weight in enumerate(kernel):
        if lag < len(counts):
            out[lag:] += weight * counts[:len(counts) - lag]
    return out


def ridge(x, y, alpha, unpenalized=1):
    """Mean squared loss + alpha*||coefficients||²; final bias columns unpenalized."""
    if len(x) == 0:
        raise ValueError("No eligible calibration samples")
    gram = x.T @ x / len(x)
    penalty = np.full(x.shape[1], alpha)
    if unpenalized:
        penalty[-unpenalized:] = 0
    gram.flat[::gram.shape[0] + 1] += penalty
    return np.linalg.solve(gram, x.T @ y / len(x))


def score(y, predicted):
    error = np.sum((y - predicted) ** 2, axis=0)
    total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    per_output = np.divide(error, total, out=np.zeros_like(error), where=total > 0)
    per_output = np.where(total > 0, 1 - per_output, np.where(error == 0, 1., 0.))
    if total.sum() <= 0:
        raise ValueError("Evaluation has no target variance")
    return float(per_output @ total / total.sum()), per_output.tolist()


def stacked_features(counts, bin_ms, taus):
    """Stack causal exponential kernels at several taus: channels-major per tau."""
    return np.column_stack([causal_features(counts, bin_ms, tau) for tau in taus])


def smooth(values, beta):
    """Causal exponential smoothing along time; the caller resets at boundaries."""
    if not 0 < beta <= 1:
        raise ValueError(f"Smoothing beta must be in (0, 1]: {beta}")
    out = np.empty_like(values)
    previous = None
    for index, row in enumerate(values):
        out[index] = row if previous is None else beta * row + (1 - beta) * previous
        previous = out[index]
    return out


def prefix_statistics(features):
    """Label-free mean and variance used to start the running z-score.

    Features silent throughout the prefix start at unit variance, so a channel
    that wakes up later cannot be divided by a near-zero scale."""
    if len(features) == 0:
        raise ValueError("Running z-score needs at least one initialization bin")
    sd = features.std(axis=0)
    sd[sd < 1e-8] = 1
    return features.mean(axis=0), sd * sd


def running_zscore(features, mean, variance, tau_s, bin_ms):
    """Causal per-feature z-score with exponentially forgetting statistics.

    Starts from (mean, variance), updates on every bin including the current one,
    and never looks ahead. The streaming Rust decoder replays the same arithmetic.
    """
    k = bin_ms / 1000 / tau_s
    if not 0 < k < 1:
        raise ValueError(f"Running z-score time constant must exceed one bin: {tau_s}s")
    mean = np.array(mean, dtype=np.float64)
    variance = np.array(variance, dtype=np.float64)
    out = np.empty_like(features, dtype=np.float64)
    for index, row in enumerate(features):
        delta = row - mean
        mean = mean + k * delta
        variance = (1 - k) * (variance + k * delta * delta)
        out[index] = (row - mean) / np.sqrt(np.maximum(variance, 1e-8))
    return out


def _recentered_weights(decoder, all_features):
    """Unsupervised normalization update; returns weights, parameters, patch."""
    mu, sd = all_features.mean(axis=0), all_features.std(axis=0)
    sd[sd < 1e-8] = 1
    w = decoder.weights.copy()
    w[:-1] = decoder.weights[:-1] * (decoder.scale / sd)[:, None]
    w[-1] += ((decoder.mean - mu) / sd) @ decoder.weights[:-1]
    return w, 2 * (w.shape[0] - 1), np.concatenate((mu, sd)).tolist()


def _rank_update(decoder, features, targets, alpha, rank):
    """Prior + rank-r slope correction and separate bias on top of the frozen decoder."""
    w = decoder.weights + _recent_prior(decoder)
    x = decoder.design(features)
    y = (targets - decoder.target_mean) / decoder.target_scale
    residual = y - x @ w
    neural = x[:, :-1]
    c = w.shape[0] - 1
    mean = neural.mean(axis=0)
    bias_cross = residual.mean(axis=0)
    bias_normal = 1 + alpha
    # Eliminate the regularized intercept before constraining slope rank.
    gram = neural.T @ neural / len(x)
    gram.flat[::c + 1] += alpha
    gram -= np.outer(mean, mean) / bias_normal
    cross = neural.T @ residual / len(x) - mean[:, None] * bias_cross / bias_normal
    delta = np.linalg.solve(gram, cross)
    # Project in the ridge metric and physical output units, not by
    # truncating the coefficient matrix's Euclidean singular values.
    covariance = delta.T @ cross
    covariance = (covariance + covariance.T) * 0.5
    covariance *= decoder.target_scale[:, None] * decoder.target_scale[None, :]
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, -rank:]
    left = (delta * decoder.target_scale) @ axes
    right = axes.T / decoder.target_scale
    slopes = left @ right
    bias = (bias_cross - mean @ slopes) / bias_normal
    w = w.copy()
    w[:-1] += slopes
    w[-1] += bias
    patch = np.concatenate((left.ravel(), right.ravel(), bias)).tolist()
    return w, patch


@dataclass
class Decoder:
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: np.ndarray
    basis: np.ndarray

    def design(self, features):
        return np.column_stack(((features - self.mean) / self.scale, np.ones(len(features))))

    def predict(self, features, weights=None):
        w = self.weights if weights is None else weights
        return (self.design(features) @ w) * self.target_scale + self.target_mean

    def folded(self, weights):
        """Compile normalization and repair into one raw-feature affine map."""
        slope = weights[:-1] * self.target_scale / self.scale[:, None]
        bias = weights[-1] * self.target_scale + self.target_mean - self.mean @ slope
        return slope, bias


def fit_base(training, alpha):
    """training: day -> (features on eligible rows, target velocities)."""
    raw_x = np.concatenate([xy[0] for xy in training.values()])
    raw_y = np.concatenate([xy[1] for xy in training.values()])
    mean, scale = raw_x.mean(axis=0), raw_x.std(axis=0)
    scale[scale < 1e-8] = 1
    ym, ys = raw_y.mean(axis=0), raw_y.std(axis=0)
    ys[ys < 1e-8] = 1
    x = np.column_stack(((raw_x - mean) / scale, np.ones(len(raw_x))))
    y = (raw_y - ym) / ys
    w = ridge(x, y, alpha)
    decoder = Decoder(mean, scale, ym, ys, w, np.empty((0, *w.shape)))
    corrections = []
    for day_x, day_y in training.values():
        dx = decoder.design(day_x)
        residual = (day_y - ym) / ys - dx @ w
        corrections.append(ridge(dx, residual, alpha))
    # The empirical prior covariance is B B^T / number_of_source_days.
    decoder.basis = np.asarray(corrections) / np.sqrt(len(corrections))
    return decoder


def _recent_prior(decoder):
    """The training pipeline stores source-day corrections chronologically."""
    if len(decoder.basis) < 3 or decoder.weights.shape[1] < REPAIR_RANK:
        raise ValueError("Rank-six repair requires three source days and at least six outputs")
    return np.sqrt(len(decoder.basis)) * decoder.basis[-3:].mean(axis=0)


def adapt(decoder, features, targets, all_features, method, alpha):
    """Return compiled weights, fitted-parameter count, elapsed ms, patch values.

    Every method receives exactly the same chronological calibration prefix.
    all_features also includes that prefix's rest/cue bins for neural recentering.
    No evaluation-time statistics or labels enter this function.
    """
    start = perf_counter()
    w = decoder.weights.copy()
    c, o = w.shape[0] - 1, w.shape[1]
    parameters, patch = 0, []
    if method == "frozen" or len(all_features) == 0:
        return w, parameters, 0., patch
    if method == "recenter":
        w, parameters, patch = _recentered_weights(decoder, all_features)
    elif len(features) == 0:
        # An empty scored prefix cannot support supervised adaptation.
        # Preserve the frozen model and explicitly report zero fitted parameters.
        return w, parameters, (perf_counter() - start) * 1000, patch
    else:
        x = decoder.design(features)
        y = (targets - decoder.target_mean) / decoder.target_scale
        pred = x @ w
        residual = y - pred
        if method == "scratch":
            w = ridge(x, y, alpha)
            parameters, patch = w.size, w.ravel().tolist()
        elif method == "anchored":
            delta = ridge(x, residual, alpha)
            w += delta
            parameters, patch = delta.size, delta.ravel().tolist()
        elif method == "output_affine":
            affine = ridge(np.column_stack((pred, np.ones(len(pred)))), residual, alpha)
            w += w @ affine[:-1]
            w[-1] += affine[-1]
            parameters, patch = affine.size, affine.ravel().tolist()
        elif method == "gain_repair":
            # Preserve the direction of each row of W; alter its scalar gain.
            slopes = np.einsum("tc,co->toc", x[:, :-1], w[:-1])
            biases = np.broadcast_to(np.eye(o), (len(x), o, o))
            design = np.concatenate((slopes, biases), axis=2).reshape(-1, c + o)
            delta = ridge(design, residual.ravel(), alpha, unpenalized=o)
            w[:-1] *= 1 + delta[:c, None]
            w[-1] += delta[c:]
            parameters, patch = delta.size, delta.tolist()
        elif method == REPAIR_METHOD:
            w, patch = _rank_update(decoder, features, targets, alpha, REPAIR_RANK)
            parameters = len(patch)
        elif method == "drift_basis":
            design = np.einsum("tf,kfo->tok", x, decoder.basis).reshape(-1, len(decoder.basis))
            theta = ridge(design, residual.ravel(), alpha, unpenalized=0)
            w += np.einsum("k,kfo->fo", theta, decoder.basis)
            parameters, patch = theta.size, theta.tolist()
        else:
            raise ValueError(f"Unknown adaptation method: {method}")
    if not np.isfinite(w).all():
        raise ValueError(f"Non-finite {method} coefficients")
    return w, int(parameters), (perf_counter() - start) * 1000, patch


def apply_patch(decoder, method, patch):
    """Reconstruct a compact repair without calibration data or refitting."""
    patch = np.asarray(patch, dtype=np.float64)
    w = decoder.weights.copy()
    c, o = w.shape[0] - 1, w.shape[1]
    if method not in ("gain_repair", "drift_basis", REPAIR_METHOD) or patch.ndim != 1:
        raise ValueError(f"Invalid compact repair or patch dimensions: {method}")
    if not np.isfinite(patch).all():
        raise ValueError("Patch contains non-finite coefficients")
    if patch.size == 0:
        return w
    if method == "gain_repair" and patch.shape == (c + o,):
        w[:-1] *= 1 + patch[:c, None]
        w[-1] += patch[c:]
    elif method == REPAIR_METHOD:
        rank = REPAIR_RANK
        split = c * rank
        if patch.shape != (split + rank * o + o,):
            raise ValueError(f"Invalid {method} patch shape {patch.shape}")
        w += _recent_prior(decoder)
        w[:-1] += patch[:split].reshape(c, rank) @ patch[split:-o].reshape(rank, o)
        w[-1] += patch[-o:]
    elif method == "drift_basis" and patch.shape == (len(decoder.basis),):
        w += np.einsum("k,kfo->fo", patch, decoder.basis)
    else:
        raise ValueError(f"Invalid {method} patch shape {patch.shape}")
    if not np.isfinite(w).all():
        raise ValueError("Patch reconstructs non-finite decoder coefficients")
    return w
