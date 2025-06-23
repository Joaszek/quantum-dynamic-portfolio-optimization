# 🧠 Quantum Portfolio Optimization with Sliding Windows

This project demonstrates how to apply **Quantum Approximate Optimization Algorithm (QAOA)** to perform **portfolio optimization** using real financial data in a **sliding window** approach.

It simulates dynamic investment decisions based on a quantum-inspired optimization technique, selecting the best subset of assets at each time interval based on a trade-off between **expected return** and **risk** (via covariance matrix).

---

## 📊 Problem Setup

We solve a quadratic binary optimization problem of the form:

> maximize  
> $$ \mu^T x - \lambda x^T \Sigma x $$
>
> subject to:  
> $$ \sum x_i = k, \quad x_i \in \{0,1\} $$ 

Where:
- \( x_i \): whether asset *i* is selected
- \( \mu \): expected return vector
- \( \Sigma \): covariance matrix
- \( \lambda \): risk aversion parameter
- \( k \): number of assets to select

The problem is solved using:
- `Qiskit Optimization` (`QuadraticProgram`)
- `QAOA` from `qiskit-algorithms`
- `MinimumEigenOptimizer`

---

## 🔁 Sliding Window Approach

The portfolio is optimized repeatedly over moving windows of historical data (e.g. 20-day windows, sliding every 5 days). This mimics real-world rolling rebalancing.

---

## 📁 Folder Structure

```bash
project/
│
├── data/
│   ├── data_apple_cocacola_google.csv    # Raw prices
│   └── results/
│       └── sliding_window_results.csv    # Optimizer output
├── src/
    ├── data_loader.py                        # Loads CSV price data
    ├── return_calculator.py                  # Computes log returns
    ├── window_optimizer.py                   # Main QAOA optimization loop
    ├── show_data.py  
└── main.py                         # main file
