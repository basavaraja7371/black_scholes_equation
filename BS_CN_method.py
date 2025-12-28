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
    """
    Solves the BS equation using CN method

    Args:
        f (function): Initial condition
        g0 (function): LHS boundary condition
        gL (function): RHS boundary condition
        S_max (float): Upper limit of the Stock price
        T (float): Expiration
        nS (int): Number of stock price grid
        nt (int): Number of time steps
        sigma (float): Volatility
        r (float): Risk free rate

    Returns:
        S (np.array): Stock price array
        V (np.array): option price array
    """
    dS = S_max / (nS - 1)
    dt = T / nt

    S = np.linspace(0, S_max, nS)
    t = np.linspace(0, T, nt + 1)

    # Only for interior points
    V = np.zeros(nS - 2)

    # initial condition
    V = f(S)[1:-1]

    A = BS_dirichlet(S, dS, sigma, r)

    # Time march
    for n in range(nt):
        g = bs_boundary_vector(
            dt, S, sigma, r, g0(t[n]), g0(t[n + 1]), gL(t[n]), gL(t[n + 1])
        )

        V = crank_nicolson_step(V, A, g, dt)

    return S[1:-1], V


def bs_analytical(S, t, T, K, r, sigma, option="call"):
    """
    Analytical solution for the Black-Scholes call price

    Args:
        S (np.array): Stock price grid
        t (float): Current time/ date
        T (float): Expiration date / Maturity
        K (float): Strike price
        r (float): Risk free rate
        sigma (float): Volatility
        option (str): call or put

    Returns:
        V (np.array): Option price
    """
    tau = T - t

    # Payoff at maturity
    if tau <= 0:
        if option.lower() == "call":
            return np.maximum(S - K, 0.0)
        elif option.lower() == "put":
            return np.maximum(K - S, 0.0)
        else:
            raise ValueError("option must be 'call' or 'put'")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option.lower() == "call":
        V = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    elif option.lower() == "put":
        V = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    else:
        raise ValueError("option must be 'call' or 'put'")

    return V


if __name__ == "__main__":
    # Initial condition
    def f_call(S):
        global K
        return np.maximum(S - K, 0)

    def f_put(S):
        global K
        return np.maximum(K - S, 0)

    # Right boundary
    def gL_call(t):
        global S_max
        global r
        global K
        return S_max - K * np.exp(-r * t)

    def gL_put(t):
        global S_max
        global r
        global K
        return 0

    # Left boundary
    def g0_call(t):
        return 0.0

    def g0_put(t):
        return K * np.exp(-r * t)

    # Parameters
    T = 1
    nS = 50
    nt = 500
    sigma = 0.1
    r = 0.01
    K = 90
    S_max = 2 * K

    S_call, V_call_final = solve_bs(
        f_call, g0_call, gL_call, S_max, T, nS, nt, sigma, r
    )
    V_call_exact = bs_analytical(
        S=S_call, t=0, T=T, K=K, r=r, sigma=sigma, option="call"
    )

    S_put, V_put_final = solve_bs(f_put, g0_put, gL_put, S_max, T, nS, nt, sigma, r)
    V_put_exact = bs_analytical(S=S_put, t=0, T=T, K=K, r=r, sigma=sigma, option="put")

    fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    ax0, ax1 = axs.flatten()

    ax0.set_xlabel(r"Stock Price, $S$", fontsize=18)
    ax0.set_ylabel(r"Call option price, $V_c$", fontsize=18)

    ax0.plot(S_call, V_call_final, linewidth=2, label="Crank-Nicolson")
    ax0.plot(S_call, V_call_exact, "r^", markersize=5, alpha=0.7, label="Exact")
    ax0.set_title(
        f"Call option, $\\sigma={sigma}$, K={K}, $r={r}$ and $T={T}$", fontsize=16
    )
    ax0.set_xlim(0, S_max)

    ax1.set_xlabel(r"Stock Price, $S$", fontsize=18)
    ax1.set_ylabel(r"Put option price, $V_p$", fontsize=18)

    ax1.plot(S_put, V_put_final, linewidth=2, label="Crank-Nicolson")
    ax1.plot(S_put, V_put_exact, "r^", markersize=5, alpha=0.7, label="Exact")
    ax1.set_title(
        f"Put option, $\\sigma={sigma}$, K={K}, $r={r}$ and $T={T}$", fontsize=16
    )
    ax1.set_xlim(0, S_max)

    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("BS_equation_CN.jpg")
    plt.show()
