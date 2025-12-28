import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.stats import norm


def BS_dirichlet(S, dS, sigma, r):
    """
    Black-Scholes matrix for the interior points. (see Readme)

    Args:
        S (np.array)): Stock price array
        dS (float): Stock price step
        sigma (float): volatility
        r (float): risk free rate

    Returns:
        A (np.ndarray): Black-Scholes Matrix
    """

    # diagonal elements
    n = len(S) - 2  # only interior points

    a = (0.5 * sigma**2 * S**2 / dS**2) - (0.5 * r * S / dS)
    b = -r - sigma**2 * S**2 / dS**2
    c = (0.5 * sigma**2 * S**2 / dS**2) + (0.5 * r * S / dS)

    diagonals = [a[2:-1], b[1:-1], c[1:-2]]

    offsets = [-1, 0, 1]

    A = diags(diagonals=diagonals, offsets=offsets, shape=(n, n)).toarray()

    return A


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

    g0 = 0.5 * sigma**2 * S[1] ** 2 / dS**2 - 0.5 * r * S[1] / dS
    gL = 0.5 * sigma**2 * S[-2] ** 2 / dS**2 + 0.5 * r * S[-2] / dS

    g[0] += 0.5 * dt * g0 * (V0_n + V0_np1)
    g[-1] += 0.5 * dt * gL * (VL_n + VL_np1)
    return g


def crank_nicolson_step(Vn, A, b, dt):
    """
    Crank-Nicolson step to calculate V[n+1] from V[n]

    Args:
        Vn (np.array): option price at t[n]
        A (np.ndarray): The matrix A
        b (np.array): bounday vector

    Returns:
        Vn_p1 (np.array): option price at t[n+1]
    """
    n = A.shape[0]
    Ix = np.eye(n)
    LHS = Ix - 0.5 * dt * A
    RHS = Ix + 0.5 * dt * A

    rhs = RHS @ Vn + b
    Vn_p1 = np.linalg.solve(LHS, rhs)
    return Vn_p1


def solve_bs(f, g0, gL, S_max, T, nS, nt, sigma, r):
    dS = S_max / (nS - 1)
    dt = T / nt

    S = np.linspace(0, S_max, nS)
    t = np.linspace(0, T, nt + 1)

    V = np.zeros(nS - 2)
    # V[0] = g0(0)
    # V[-1] = gL(0)
    # V[1:-1] = f(S)[1:-1]
    V = f(S)[1:-1]

    A = BS_dirichlet(S, dS, sigma, r)

    for n in range(nt):
        g = bs_boundary_vector(
            dt, S, sigma, r, g0(t[n]), g0(t[n + 1]), gL(t[n]), gL(t[n + 1])
        )

        # V[1:-1] = crank_nicolson_step(V[1:-1], A, g, dt)
        V = crank_nicolson_step(V, A, g, dt)

    return S[1:-1], V


def bs_call_analytical(S, t, T, K, r, sigma):
    """
    Analytical Black–Scholes price of a European call option.

    Parameters
    ----------
    S : np.ndarray or float
        Stock price(s)
    t : float
        Current time
    T : float
        Maturity
    K : float
        Strike price
    r : float
        Risk-free rate
    sigma : float
        Volatility

    Returns
    -------
    V : np.ndarray or float
        Option value(s)
    """
    tau = T - t

    # Handle maturity exactly
    if tau <= 0:
        return np.maximum(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    V = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return V


if __name__ == "__main__":
    # def exact_solution(x, t, alpha):
    #     return np.exp(-(np.pi**2) * alpha * t) * np.sin(np.pi * x)

    # Initial condition
    def f(S):
        global K
        return np.maximum(S - K, 0)

    # Right boundary
    def gL(t):
        global S_max
        global r
        global K
        return S_max - K * np.exp(-r * t)

    # Left boundary
    def g0(t):
        return 0.0

    T = 0.5
    nS = 100
    nt = 100
    sigma = 0.3
    r = 0.05
    K = 100
    S_max = 2 * K

    S, V_final = solve_bs(f, g0, gL, S_max, T, nS, nt, sigma, r)

    V_exact = bs_call_analytical(S=S, t=0, T=T, K=K, r=r, sigma=sigma)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel(r"Stock Price, $S$", fontsize=18)
    ax.set_ylabel(r"Option price, $V$", fontsize=18)

    ax.plot(S, V_final, linewidth=2, label="Crank-Nicolson")
    ax.plot(S, V_exact, "rs", markersize=5, alpha=0.7, label="Exact")

    ax.legend(fontsize=12)
    plt.tight_layout()
    # plt.savefig("heat_equation_CN.jpg")
    plt.show()
