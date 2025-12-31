import numpy as np


def bs_mc_price(S0, K, r, sigma, T, n_paths=10000, option="call", random_seed=47):
    """
    Does Monte Carlo simulation of the stock price which follows GBM. Calculates
    the discounted average of option payoff.

    Args:
        S0 (float): Initial Stock Price
        K (float): Strike price
        r (float): Risk free rate
        sigma (float): volatility
        T (float): expiration
        n_paths (int, optional): Number of stock paths to simulate. Defaults to 10000.
        option (str, optional): option type. Defaults to "call".
        random_seed (int, optional): Random seed. Defaults to 47.

    Returns:
        price (float): Simulated option price
        stderr (float): Standard error
    """
    np.random.seed(random_seed)

    # Standard Normal values
    Z = np.random.normal(0.0, 1.0, n_paths)

    # Stock prices at T for eacn path
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # option payoff at T
    if option == "call":
        payoff = np.maximum(ST - K, 0)
    elif option == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option must be 'call' or 'put'")

    # Average over paths and Discount the payoffs
    price = np.exp(-r * T) * payoff.mean()
    stderr = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(n_paths)

    return price, stderr


if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 90
    r = 0.01
    sigma = 0.1
    T = 1
    S_max = 2 * K

    price, err = bs_mc_price(
        S0=S0, K=K, r=r, sigma=sigma, T=T, option="call", random_seed=23
    )

    print(f"MC price = {price:.4f} ± {1.96 * err:.4f}")
