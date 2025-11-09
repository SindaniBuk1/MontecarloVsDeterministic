import math

def f1(x):   # lisse, test de base
    return math.sin(x)

def f2(x):   # gaussienne tronquée
    return math.exp(-x**2)

def f3(x):   # fonction rationnelle
    return 1/(1 + x**2)

def f4(x):   # oscillante
    return x*(1-x)*math.sin(200*x*(1-x))

def f5(x):   # légèrement singulière près de 0
    return 1/math.sqrt(x+1e-12)
import time
import math
import random

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# 1. Fonctions tests
def f1(x):
    return math.sin(x)

def f2(x):
    return math.exp(-x**2)

def f3(x):
    return 1/(1 + x**2)

def f4(x):
    return x*(1-x)*math.sin(200*x*(1-x))

def f5(x):
    return 1/math.sqrt(x + 1e-12)

functions = {
    "f1(x)=sin(x)": f1,
    "f2(x)=exp(-x^2)": f2,
    "f3(x)=1/(1+x^2)": f3,
    "f4(x)=x(1-x)sin(200x(1-x))": f4,
    "f5(x)=1/sqrt(x)": f5,
}

a, b = 0.0, 1.0

# 2. méthodes déterministes
def rect_left(f, a, b, n):
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        s += f(a + i*h)
    return h * s

def trapezoid(f, a, b, n):
    h = (b - a) / n
    s = 0.5*(f(a) + f(b))
    for i in range(1, n):
        s += f(a + i*h)
    return h * s

def simpson(f, a, b, n):
    # n doit être pair
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i*h
        if i % 2 == 0:
            s += 2*f(x)
        else:
            s += 4*f(x)
    return s * h / 3.0

# 3. Monte Carlo
def monte_carlo(f, a, b, n):
    x = np.random.uniform(a, b, size=n)
    y = np.array([f(xx) for xx in x])
    return (b - a) * y.mean()

# 4. tailles d'échantillon
N_list = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

# 5. boucle principale
results = {}  # dict[(fname, method)] = { "N": [...], "err": [...], "time": [...] }

for fname, f in functions.items():
    # valeur de référence très précise
    I_true = mp.quad(lambda x: f(float(x)), [a, b])
    for method_name, method in [
        ("Rectangles", rect_left),
        ("Trapèzes", trapezoid),
        ("Simpson", simpson),
        ("Monte Carlo", monte_carlo),
    ]:
        key = (fname, method_name)
        results[key] = {"N": [], "err": [], "time": []}
        for N in N_list:
            t0 = time.perf_counter()
            I_hat = method(f, a, b, N)
            t1 = time.perf_counter()
            err = abs(I_hat - I_true)
            results[key]["N"].append(N)
            results[key]["err"].append(err)
            results[key]["time"].append(t1 - t0)

# 6. affichage graphique
for fname, f in functions.items():
    plt.figure()
    for method_name in ["Rectangles", "Trapèzes", "Simpson", "Monte Carlo"]:
        key = (fname, method_name)
        N = np.array(results[key]["N"])
        err = np.array(results[key]["err"])
        plt.loglog(N, err, marker="o", label=method_name)
    plt.xlabel("N (évaluations / tirages)")
    plt.ylabel("erreur absolue")
    plt.title(f"Convergence pour {fname}")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# 7. graphiques temps -> erreur
for fname, f in functions.items():
    plt.figure()
    for method_name in ["Rectangles", "Trapèzes", "Simpson", "Monte Carlo"]:
        key = (fname, method_name)
        t = np.array(results[key]["time"])
        err = np.array(results[key]["err"])
        plt.loglog(t, err, marker="o", label=method_name)
    plt.xlabel("temps (s)")
    plt.ylabel("erreur absolue")
    plt.title(f"Efficacité temps pour {fname}")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.show()
import numpy as np
import itertools
import mpmath as mp
import matplotlib.pyplot as plt
import time

# ===========================
# 1. Fonctions tests par dimension
# ===========================
def f1(x):           # d = 1
    return np.sin(np.pi * x)

def f2(x):           # d = 2
    return np.exp(-(x[0]**2 + x[1]**2))

def f3(x):           # d = 3
    return 1 / (1 + np.sum(x**2))

def f4(x):           # d = 4
    return np.prod(np.sin(np.pi * x))

def f5(x):           # d = 5
    return np.exp(-np.sum(x))

functions = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5
}
def monte_carlo_nd(f, d, n):
    X = np.random.rand(n, d)
    vals = np.array([f(x) for x in X], dtype=float)
    return float(vals.mean())

def trapezoidal_nd(f, d, n_per_axis):
    grid_1d = np.linspace(0, 1, n_per_axis)
    total = 0.0
    for x_tuple in itertools.product(grid_1d, repeat=d):
        total += float(f(np.array(x_tuple)))
    return total / (n_per_axis ** d)

# ===========================
# 4. Calculs et comparaison
# ===========================
N_list = [100, 500, 1000, 2000, 5000, 10000]
errors_mc = {d: [] for d in range(1,6)}
errors_tp = {d: [] for d in range(1,6)}

# Valeurs exactes via Monte Carlo très précis (1e6 tirages)
np.random.seed(42)
exact_values = {}
for d, f in functions.items():
    exact_values[d] = monte_carlo_nd(f, d, 2_000_000)
    print(f"Dimension {d} : I ≈ {exact_values[d]:.6f}")

for d, f in functions.items():
    print(f"\n=== Dimension {d} ===")
    for N in N_list:
        # Monte Carlo
        I_mc = monte_carlo_nd(f, d, N)
        err_mc = abs(I_mc - exact_values[d])
        errors_mc[d].append(err_mc)

        # Trapèze tensoriel
        n_axis = int(round(N ** (1/d)))  # points par axe
        I_tp = trapezoidal_nd(f, d, n_axis)
        err_tp = abs(I_tp - exact_values[d])
        errors_tp[d].append(err_tp)

        print(f"N={N:6d}  MC_err={err_mc:.3e}  Trap_err={err_tp:.3e}")

# ===========================
# 5. Visualisation log-log
# ===========================
plt.figure(figsize=(10,6))
colors = ['b','g','r','c','m']
for i, d in enumerate(range(1,6)):
    plt.loglog(N_list, errors_mc[d], marker='o', color=colors[i],
               label=f'Monte Carlo d={d}')
    plt.loglog(N_list, errors_tp[d], marker='s', linestyle='--',
               color=colors[i], alpha=0.6,
               label=f'Trapèze d={d}')

plt.xlabel("Nombre d'échantillons N (log)")
plt.ylabel("Erreur absolue (log)")
plt.title("Comparaison Monte Carlo vs Trapèze selon la dimension d")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
import numpy as np
import itertools
import mpmath as mp
import matplotlib.pyplot as plt
import time

# ===========================
# 1. Fonctions tests par dimension
# ===========================
def f1(x):  # d = 1
    return np.sin(np.pi * x)

def f2(x):  # d = 2
    return np.exp(-(x[0]**2 + x[1]**2))

def f3(x):  # d = 3
    return 1 / (1 + np.sum(x**2))

def f4(x):  # d = 4
    return np.prod(np.sin(np.pi * x))

def f5(x):  # d = 5
    return np.exp(-np.sum(x))

functions = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5}

# ===========================
# 2. Monte Carlo en dimension d
# ===========================
def monte_carlo_nd(f, d, n):
    X = np.random.rand(n, d)
    vals = np.array([f(x) for x in X], dtype=float)
    return float(vals.mean())

# ===========================
# 3. Quadrature "trapèzes tensorielle" simplifiée
# ===========================
def trapezoidal_nd(f, d, n_per_axis):
    """Approximation sur une grille régulière [0,1]^d."""
    grid_1d = np.linspace(0, 1, n_per_axis)
    total = 0.0
    for x_tuple in itertools.product(grid_1d, repeat=d):
        total += float(f(np.array(x_tuple)))
    return total / (n_per_axis ** d)  # moyenne simple (volume=1)

# ===========================
# 4. Calculs et comparaison
# ===========================
N_list = [100, 500, 1000, 2000, 5000, 10000]
errors_mc = {d: [] for d in range(1, 6)}
errors_tp = {d: [] for d in range(1, 6)}
times_mc = {d: [] for d in range(1, 6)}
times_tp = {d: [] for d in range(1, 6)}

# Valeurs exactes via Monte Carlo très précis
np.random.seed(42)
exact_values = {}
for d, f in functions.items():
    exact_values[d] = monte_carlo_nd(f, d, 2_000_000)
    print(f"Dimension {d} : I ≈ {exact_values[d]:.6f}")

for d, f in functions.items():
    print(f"\n=== Dimension {d} ===")
    for N in N_list:
        # Monte Carlo
        t0 = time.perf_counter()
        I_mc = monte_carlo_nd(f, d, N)
        t1 = time.perf_counter()
        err_mc = abs(I_mc - exact_values[d])
        errors_mc[d].append(float(err_mc))
        times_mc[d].append(t1 - t0)

        # Trapèze tensoriel
        n_axis = int(round(N ** (1/d)))  # points par axe
        t0 = time.perf_counter()
        I_tp = trapezoidal_nd(f, d, n_axis)
        t1 = time.perf_counter()
        err_tp = abs(I_tp - exact_values[d])
        errors_tp[d].append(float(err_tp))
        times_tp[d].append(t1 - t0)

        print(f"N={N:6d}  MC_err={float(err_mc):.3e}  Trap_err={float(err_tp):.3e}")

# ===========================
# 5. Graphique : Erreur vs N (convergence)
# ===========================
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm']
for i, d in enumerate(range(1, 6)):
    plt.loglog(N_list, errors_mc[d], marker='o', color=colors[i],
               label=f'Monte Carlo d={d}')
    plt.loglog(N_list, errors_tp[d], marker='s', linestyle='--',
               color=colors[i], alpha=0.6,
               label=f'Trapèze d={d}')
plt.xlabel("Nombre d'échantillons N (log)")
plt.ylabel("Erreur absolue (log)")
plt.title("Convergence : Monte Carlo vs Trapèze selon la dimension")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

# ===========================
# 6. Graphique : Erreur vs Temps (efficacité)
# ===========================
plt.figure(figsize=(10, 6))
for i, d in enumerate(range(1, 6)):
    plt.loglog(times_mc[d], errors_mc[d], marker='o', color=colors[i],
               label=f'Monte Carlo d={d}')
    plt.loglog(times_tp[d], errors_tp[d], marker='s', linestyle='--',
               color=colors[i], alpha=0.6,
               label=f'Trapèze d={d}')
plt.xlabel("Temps de calcul (s) [log]")
plt.ylabel("Erreur absolue (log)")
plt.title("Efficacité : Erreur vs Temps (Monte Carlo vs Trapèze)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

