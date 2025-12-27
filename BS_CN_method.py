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

    a = (0.25 * sigma**2 * S**2 / dS**2) - (0.25 * r * S / dS)
    b = -0.5 * r - (0.5 * sigma**2 * S**2 / dS**2)
    c = (0.25 * sigma**2 * S**2 / dS**2) + (0.25 * r * S / dS)

    diagonals = [a[2:-1], b[1:-1], c[1:-2]]

    offsets = [-1, 0, 1]

    M = diags(diagonals=diagonals, offsets=offsets, shape=(n, n)).toarray()

    return M


def bs_boundary_vector(dt, S, sigma, r, V0_n, V0_np1, VL_n, VL_np1):
    """
    Function to create a boundary condition vector g

    Args:
        dt (float): Time step
        S (np.array): Stock Price grid
        sigma (float): volatility
        r (float): risk free rate
        V0_n (fuction): LHS boundary value of the option at time ndt
        V0_np1 (function): LHS boundary value of the option at time (n+1)dt
        VL_n (function): RHS boundary value of the option at time ndt
        VL_np1 (function): RHS boundary value of the option at time (n+1)dt

    Returns:
        g (np.array): boundry condition vector
    """
    n = len(S) - 2
    dS = S[1] - S[0]
    g = np.zeros(n)

    g0 = 0.5 * dt * ((0.5 * sigma**2 * S[1] ** 2 / dS**2) - (0.5 * r * S[1] / dS))
    gL = 0.5 * dt * ((0.5 * sigma**2 * S[-1] ** 2 / dS**2) + (0.5 * r * S[-1] / dS))

    g[0] += 0.5 * dt * g0 * (V0_n + V0_np1)
    g[-1] += 0.5 * dt * gL * (VL_n + VL_np1)
    return g


if __name__ == "__main__":
    sigma = 0.3
    r = 0.05
    S = np.linspace(0, 100, 11)
    dS = S[1] - S[0]
    M = BS_dirichlet(S, dS, sigma, r)
    print(M)
