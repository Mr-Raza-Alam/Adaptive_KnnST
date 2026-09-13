"""
Compressive Sensing (CS) algorithm for missing-data imputation / matrix
completion -- the baseline compared against KNN-ST.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cs_impute",
    "random_loss_model",
    "normalized_error",
    "grid_search",
]


# --------------------------------------------------------------------- #
#  Algorithm 1 of Zhu et al. (TMC 2013)                                 #
# --------------------------------------------------------------------- #
def cs_impute(M, B, r, lam, t=100, seed=None, use_mask=True):
    rng = np.random.default_rng(seed)
    M = np.asarray(M, dtype=float)
    B = np.asarray(B, dtype=float)
    if M.shape != B.shape:
        raise ValueError("M and B must have the same shape")
    m, n = M.shape
    r = int(r)
    if not (1 <= r <= min(m, n)):
        raise ValueError("rank bound r must be in [1, min(m, n)]")

    # 1) L <- random m x r matrix
    L = rng.standard_normal((m, r))

    L_hat = L.copy()
    S_hat = np.zeros((r, n))
    v_hat = np.inf

    for _ in range(int(t)):
        if use_mask:
            # 3) S (r x n): each column solved over its observed rows only
            S = _als_step(L, M, B, lam)
            # 4) L (m x r): each row solved over its observed columns only
            L = _als_step(S.T, M.T, B.T, lam).T
        else:
            # Literal Fig. 9: updates fit the full M (missing cells = 0)
            S = _solve_reg(L, M, lam)            # r x n
            L = _solve_reg(S.T, M.T, lam).T      # m x r

        # 5) objective on the observed entries only (Eq. 16)
        v = (np.sum((B * (L @ S) - M) ** 2)
             + lam * (np.sum(L ** 2) + np.sum(S ** 2)))

        # 6-8) keep the best factors seen so far
        if v < v_hat:
            v_hat = v
            L_hat, S_hat = L.copy(), S.copy()

    # 10) X_hat <- L_hat * R_hat^T
    return L_hat @ S_hat


def _solve_reg(A, Y, lam):
    """Unmasked regularized least squares, the paper's procedure
    inverse(P, Q) with P = [A; sqrt(2*lam) I]:

        C = (A^T A + 2*lam I)^{-1} A^T Y
    """
    r = A.shape[1]
    G = A.T @ A + 2.0 * lam * np.eye(r)
    H = A.T @ Y
    try:
        return np.linalg.solve(G, H)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, H, rcond=None)[0]


def _als_step(A, Y, W, lam):
    """Weighted (masked) regularized least squares for every column.

    For each column c of Y with 0/1 weights W[:, c], solves

        x_c = (A^T diag(W[:,c]) A + 2*lam*I)^{-1} A^T diag(W[:,c]) Y[:,c]

    This is the exact ALS update of Eq. (16) restricted to observed
    entries.  A column with no observed entry yields 0.
    """
    r = A.shape[1]
    n_cols = Y.shape[1]
    X = np.zeros((r, n_cols))
    two_lam_I = 2.0 * lam * np.eye(r)
    for c in range(n_cols):
        w = W[:, c]
        if not w.any():
            continue
        Aw = A * w[:, None]                 # zero-out unobserved rows
        G = A.T @ Aw + two_lam_I
        h = Aw.T @ Y[:, c]
        try:
            X[:, c] = np.linalg.solve(G, h)
        except np.linalg.LinAlgError:
            X[:, c] = np.linalg.lstsq(G, h, rcond=None)[0]
    return X


# --------------------------------------------------------------------- #
#  Evaluation helpers (matching the KNN-ST paper's protocol)           #
# --------------------------------------------------------------------- #
def random_loss_model(X, loss_prob, rng=None):
    """Random loss model of the KNN-ST paper: each entry is deleted (and
    later deduced) with the same probability `loss_prob`.

    Parameters
    ----------
    X : (m, n) array_like
        Complete (ground-truth) data matrix.
    loss_prob : float in [0, 1)
        Probability that any given entry is missing.
    rng : np.random.Generator or int, optional

    Returns
    -------
    M : (m, n) ndarray
        Measurement matrix (missing entries = 0).
    B : (m, n) ndarray
        Indicator matrix (1 = observed, 0 = missing).
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    B = (rng.random(X.shape) >= loss_prob).astype(float)
    return X * B, B


def normalized_error(X_true, X_hat, B):
    """Normalized (absolute) error -- Eq. (2) of the KNN-ST paper:

        NE = sum |S - S_hat|  over missing entries
             ---------------------------------------
              sum |S|         over missing entries
    """
    missing = (B == 0)
    denom = np.abs(X_true[missing]).sum()
    if denom == 0:
        return np.nan
    return np.abs(X_true[missing] - X_hat[missing]).sum() / denom


def grid_search(M, B, X_true, ranks, lams, t=100, seed=None):
    """'Hit and trial' tuning of r and lambda (as done in the KNN-ST paper).

    Returns
    -------
    best_r, best_lam, best_ne : optimal triplet
    results : list of (r, lam, NE) for every combination tried
    """
    results = []
    best = (None, None, np.inf)
    for r in ranks:
        for lam in lams:
            X_hat = cs_impute(M, B, r=r, lam=lam, t=t, seed=seed)
            ne = normalized_error(X_true, X_hat, B)
            results.append((r, lam, ne))
            if ne < best[2]:
                best = (r, lam, ne)
    return best[0], best[1], best[2], results


# --------------------------------------------------------------------- #
#  Smoke test / demo on synthetic spatio-temporal data                 #
# --------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(42)

    # Synthetic environmental-style matrix: 60 time slots x 48 sensors,
    # built from a few latent factors + a daily periodic component.
    m, n, latent = 60, 48, 3
    A = rng.standard_normal((m, latent))
    C = rng.standard_normal((latent, n))
    X = A @ C
    time_axis = np.arange(m)
    for j in range(n):
        X[:, j] += 10.0 + 3.0 * np.sin(2 * np.pi * time_axis / 24 + 0.3 * j)
    X += 0.05 * rng.standard_normal(X.shape)   # small noise

    print("Compressive Sensing (Zhu et al., TMC 2013) -- matrix completion")
    print(f"matrix: {m} time slots x {n} sensors\n")
    print(" loss_prob |   NE  (r=5, lambda=1)")
    print("-" * 32)
    for p in (0.3, 0.5, 0.7, 0.8, 0.9):
        M, B = random_loss_model(X, p, rng)
        X_hat = cs_impute(M, B, r=5, lam=1.0, t=100, seed=0)
        print(f"   {p:4.2f}    | {normalized_error(X, X_hat, B):7.4f}")

    # Hit-and-trial parameter search at loss probability 0.8.
    M, B = random_loss_model(X, 0.8, rng)
    best_r, best_lam, best_ne, _ = grid_search(
        M, B, X,
        ranks=[1, 2, 3, 4, 5, 6, 8],
        lams=[0.1, 1, 10, 100, 1000],
        t=100, seed=0,
    )
    print(f"\nBest at loss 0.8 (hit-and-trial): "
          f"r={best_r}, lambda={best_lam}, NE={best_ne:.4f}")


if __name__ == "__main__":
    _demo()
