# %%
# Wolverine multi-site short series -> GP composite + Granger (statsmodels) + circular-shift null
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

# Granger from statsmodels
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore", category=UserWarning)  # silence statsmodels verbosity for small samples

rng = np.random.default_rng(42)

# ------------------------
# 0) Generate synthetic data (your original set-up)
# ------------------------
loc_ids = [f"loc{i+1}" for i in range(4)]
lengths = rng.integers(10, 41, size=4).tolist()  # per-site N in [10,40]
records = []

for loc, n in zip(loc_ids, lengths):
    # Snow(t): AR(1) -> rank-map to [0,1]
    phi = 0.7
    e = rng.normal(0, 0.3, size=n)
    s = np.zeros(n)
    s[0] = rng.normal(0, 1)
    for t in range(1, n):
        s[t] = phi * s[t-1] + e[t]
    snow = (pd.Series(s).rank(method="average").to_numpy() - 1) / (n - 1)

    # Pop(t): AR(1) + Snow(t-1)
    pop = np.zeros(n)
    pop[0] = rng.normal(0, 1)
    for t in range(1, n):
        pop[t] = 0.55 * pop[t-1] + 0.45 * snow[t-1] + rng.normal(0, 0.01)
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-12) * 10

    years = np.arange(2000, 2000 + n)
    for t in range(n):
        records.append({"location": loc, "year": int(years[t]), "snow": float(snow[t]), "pop": float(pop[t])})

df = pd.DataFrame(records).sort_values(["location", "year"]).reset_index(drop=True)

# ------------------------
# 1) GP composites over the union time axis (pool all sites)
# ------------------------
t_min = int(df["year"].min())
t_max = int(df["year"].max())
t_grid = np.arange(t_min, t_max + 1).astype(float).reshape(-1, 1)

def fit_gp(time_col, y_col):
    X = df[[time_col]].astype(float).to_numpy()
    y = df[y_col].to_numpy()
    # Smooth component + tiny linear part + i.i.d. noise; WhiteKernel allows repeated X
    kernel = (
        C(1.0, (1e-2, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1.0, 50.0))
        + 0.1 * DotProduct(sigma_0=1.0)
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1.0))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-6, random_state=0, n_restarts_optimizer=3
    )
    gpr.fit(X, y)
    mu, std = gpr.predict(t_grid, return_std=True)
    return gpr, mu, std

gp_snow, snow_mu, snow_std = fit_gp("year", "snow")
gp_pop,  pop_mu,  pop_std  = fit_gp("year", "pop")

comp = pd.DataFrame({
    "year": t_grid.ravel().astype(int),
    "snow_mean": snow_mu,
    "snow_std": snow_std,
    "pop_mean": pop_mu,
    "pop_std": pop_std
})

# ------------------------
# 2) Plot the GP composites with uncertainty (no saving)
# ------------------------
x_year = comp["year"].to_numpy()
snow_mu = comp["snow_mean"].to_numpy(dtype=float)
snow_std = comp["snow_std"].to_numpy(dtype=float)
pop_mu  = comp["pop_mean"].to_numpy(dtype=float)
pop_std = comp["pop_std"].to_numpy(dtype=float)

plt.figure(figsize=(10, 4.5))
plt.title("Composite SNOW via GP (mean ± 2σ)")
plt.fill_between(x_year, snow_mu - 2*snow_std, snow_mu + 2*snow_std, alpha=0.25, label="GP 95% band")
plt.plot(x_year, snow_mu, label="GP mean")
plt.scatter(df["year"], df["snow"], s=12, alpha=0.6, label="site observations")
plt.xlabel("Year"); plt.ylabel("Snow persistence (0–1)")
plt.legend(loc="best"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 4.5))
plt.title("Composite POP via GP (mean ± 2σ)")
plt.fill_between(x_year, pop_mu - 2*pop_std, pop_mu + 2*pop_std, alpha=0.25, label="GP 95% band")
plt.plot(x_year, pop_mu, label="GP mean")
plt.scatter(df["year"], df["pop"], s=12, alpha=0.6, label="site observations")
plt.xlabel("Year"); plt.ylabel("Wolverine population index (0–10)")
plt.legend(loc="best"); plt.tight_layout(); plt.show()

# ------------------------
# 3) Granger causality (statsmodels) on z-scored GP means
# ------------------------
# Use lag L=1 to match the DGP (pop_t depends on snow_{t-1}).
L = 1
y_pop  = stats.zscore(comp["pop_mean"].to_numpy(dtype=float))
x_snow = stats.zscore(comp["snow_mean"].to_numpy(dtype=float))

def granger_ssr_ftest(y, x, maxlag=1):
    """
    statsmodels expects an array with columns [y, x]; it tests if x 'Granger-causes' y.
    We extract the SSR-based F-test at the requested lag.
    """
    data = np.column_stack([y, x])
    res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    # 'ssr_ftest' -> (F, pvalue, df_denom, df_num)
    F, p, _, _ = res[maxlag][0]["ssr_ftest"]
    return float(F), float(p)

F_s2p, p_s2p = granger_ssr_ftest(y_pop, x_snow, maxlag=L)   # snow → pop
F_p2s, p_p2s = granger_ssr_ftest(x_snow, y_pop, maxlag=L)   # pop → snow

# ------------------------
# 4) Empirical null via circular shifts (keep autocorr; break cross-dependence)
# ------------------------
def circular_shift(arr, k):
    if k == 0:
        return arr.copy()
    k = k % len(arr)
    return np.concatenate([arr[-k:], arr[:-k]])

Nnull = 1000
F_null_s2p = np.empty(Nnull)
F_null_p2s = np.empty(Nnull)

for i in range(Nnull):
    k = int(rng.integers(1, len(y_pop)-1))  # avoid 0-shift
    # snow → pop with shifted snow
    F_null_s2p[i], _ = granger_ssr_ftest(y_pop, circular_shift(x_snow, k), maxlag=L)
    # pop → snow with shifted pop
    F_null_p2s[i], _ = granger_ssr_ftest(x_snow, circular_shift(y_pop, k), maxlag=L)

p_emp_s2p = (1 + np.sum(F_null_s2p >= F_s2p)) / (1 + Nnull)
p_emp_p2s = (1 + np.sum(F_null_p2s >= F_p2s)) / (1 + Nnull)

# ------------------------
# 5) Plot null histograms with observed F (no saving)
# ------------------------
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title(f"Null F (snow → pop), L={L}")
plt.hist(F_null_s2p, bins=30, alpha=0.8, density=True)
plt.axvline(F_s2p, linestyle="--", linewidth=2, label=f"Observed F = {F_s2p:.2f}")
plt.xlabel("F-statistic"); plt.legend()

plt.subplot(1,2,2)
plt.title(f"Null F (pop → snow), L={L}")
plt.hist(F_null_p2s, bins=30, alpha=0.8, density=True)
plt.axvline(F_p2s, linestyle="--", linewidth=2, label=f"Observed F = {F_p2s:.2f}")
plt.xlabel("F-statistic"); plt.legend()

plt.tight_layout(); plt.show()

# ------------------------
# 6) Print summary
# ------------------------
print({
    "lag_L": L,
    "F_snow_to_pop": F_s2p,
    "theory_p_snow_to_pop": p_s2p,
    "empirical_p_snow_to_pop": p_emp_s2p,
    "F_pop_to_snow": F_p2s,
    "theory_p_pop_to_snow": p_p2s,
    "empirical_p_pop_to_snow": p_emp_p2s,
})

# print the significant test results
if p_emp_s2p < 0.05:
    print("Significant Granger causality detected: snow → pop")
if p_emp_p2s < 0.05:
    print("Significant Granger causality detected: pop → snow")
