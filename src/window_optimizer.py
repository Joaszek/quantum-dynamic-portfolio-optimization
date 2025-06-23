import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
import os
import pandas as pd


from src.return_calculator import calculate_logs

def optimize_window(mu, Sigma, k, lambda_risk=0.5):
    n = len(mu)
    qp = QuadraticProgram()

    for i in range(n):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": -mu[i] for i in range(n)}
    quadratic = {}

    for i in range(n):
        for j in range(n):
            var_i, var_j = f"x{i}", f"x{j}"
            key = (var_i, var_j)
            coeff = lambda_risk * Sigma[i][j]
            quadratic[key] = coeff
    
    qp.linear_constraint(
        linear={f"x{i}": 1 for i in range(n)},
        sense='==',
        rhs=k,
        name="asset_selection"
    )

    qp.minimize(linear=linear, quadratic=quadratic)

    sampler = Sampler()
    optimizer = COBYLA()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer)
    meo = MinimumEigenOptimizer(qaoa)

    result = meo.solve(qp)

    return result


def run_sliding_window(window_size=20, step=5, k=2):
    log_returns = calculate_logs()
    results = []

    for start in range(0, len(log_returns) - window_size + 1, step):
        window = log_returns.iloc[start: start + window_size]
        mu = window.mean().values
        Sigma = window.cov().values

        print(f"\nWindow {start}-{start + window_size}")
        result = optimize_window(mu, Sigma, k)
        print("Selected assets:", result.x)
        print("Objective value:", result.fval)
        results.append(result)

    save_results(results=results, step=step, window_size=window_size)

    return results

def save_results(results, step, window_size):
    results_data = []
    for idx, res in enumerate(results):
        results_data.append({
            'window_start': idx * step,
            'window_end': idx * step + window_size,
            'selected_assets': list(res.x),
            'objective_value': res.fval
        })

    # Ensure the results directory exists
    os.makedirs('data/results', exist_ok=True)

    # Save to CSV
    df = pd.DataFrame(results_data)
    df.to_csv('data/results/sliding_window_results.csv', index=False)
