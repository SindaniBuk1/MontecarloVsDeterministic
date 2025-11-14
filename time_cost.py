import numpy as np
import time
import pandas as pd

# ---------------------------------------------------------
# Monte Carlo integration in dimension d
# ---------------------------------------------------------
def monte_carlo_integral(f, dim, N, n_rep=3):
    """
    Monte Carlo estimator in dimension `dim` with N samples.
    We repeat n_rep times and average the CPU time to avoid 0.000000s.
    """
    times = []
    estimates = []
    for _ in range(n_rep):
        start = time.perf_counter()
        X = np.random.rand(N, dim)  # uniform on [0,1]^d
        values = f(X)
        est = values.mean()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        estimates.append(est)
    return np.mean(estimates), np.mean(times)

# ---------------------------------------------------------
# 1D Trapezoidal rule on [0,1]
# ---------------------------------------------------------
def trapezoidal_1d(g, N):
    x = np.linspace(0.0, 1.0, N)
    h = 1.0 / (N - 1)
    y = g(x)
    # Trapezoidal formula in 1D
    I = h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])
    return I

# ---------------------------------------------------------
# Separable d-dimensional trapezoidal via 1D integral
# f(x1,...,xd) = prod_i g(x_i)
# => ∫_{[0,1]^d} f = (∫_0^1 g(t) dt)^d
# ---------------------------------------------------------
def trapezoidal_separable(g1d, dim, N, n_rep=3):
    """
    Trapezoidal rule for separable integrand in dimension `dim`.
    We compute the 1D trapezoidal integral and then take its d-th power.
    """
    times = []
    estimates = []
    for _ in range(n_rep):
        start = time.perf_counter()
        I1 = trapezoidal_1d(g1d, N)
        Id = I1 ** dim
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        estimates.append(Id)
    return np.mean(estimates), np.mean(times)

# ---------------------------------------------------------
# Test function
# f(x) = exp(-sum x_i^2) = prod exp(-x_i^2)
# ---------------------------------------------------------
def f_nd(x):
    # x: shape (N, d)
    return np.exp(-np.sum(x**2, axis=1))

def g_1d(t):
    # t: 1D array
    return np.exp(-t**2)

# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------
dimensions = [1, 2, 3, 4, 5]
Ns = [100, 1000, 5000, 10000]

results = []

# ---------------------------------------------------------
# Time-cost experiments
# ---------------------------------------------------------
for d in dimensions:
    for N in Ns:
        mc_val, mc_time = monte_carlo_integral(f_nd, d, N, n_rep=3)
        tp_val, tp_time = trapezoidal_separable(g_1d, d, N, n_rep=3)

        results.append({
            "Dimension": d,
            "N": N,
            "MC_time_sec": mc_time,
            "TP_time_sec": tp_time,
            "MC_estimate": mc_val,
            "TP_estimate": tp_val
        })

        print(f"Dimension={d}, N={N}  -->  "
              f"MC={mc_time:.6e}s,  TP={tp_time:.6e}s")

# ---------------------------------------------------------
# Convert to DataFrame
# ---------------------------------------------------------
df_timecost = pd.DataFrame(results)

print("\n================ TIME COST TABLE ================")
print(df_timecost)

# Optional: save your results to reuse in LaTeX
df_timecost.to_csv("time_cost_results_separable.csv", index=False)
