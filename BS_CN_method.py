import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


def BS_dirichlet(S, dS, sigma, r):
    """
    Black-Scholes matrix for the interior points. (see Readme)

    Args:
        S (np.array)): Stock price array
        dS (float): Stock price step
        sigma (float): volatility
        r (float): risk free rate

    Returns:
        M(np.ndarray): Black-Scholes Matrix
    """

    # diagonal elements
    n = len(S) - 2  # only interior points
    A = (0.25 * sigma**2 * S**2 / dS**2) + (0.25 * r * S / dS)
    B = -0.5 * r - (0.5 * sigma**2 * S**2 / dS**2)
    C = (0.25 * sigma**2 * S**2 / dS**2) - (0.25 * r * S / dS)

    diagonals = [C[2:-1], B[1:-1], A[1:-2]]

    offsets = [-1, 0, 1]

    M = diags(diagonals=diagonals, offsets=offsets, shape=(n, n)).toarray()

    return M

    return LHS, RHS, r


if __name__ == "__main__":
    sigma = 0.3
    r = 0.05
    S = np.linspace(0, 100, 11)
    dS = S[1] - S[0]
    M = BS_dirichlet(S, dS, sigma, r)
    print(M)
