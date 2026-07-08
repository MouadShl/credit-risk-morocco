"""
Generates a synthetic dataset that mirrors the UCI Default of Credit Card Clients dataset.
Statistical properties (distributions, correlations, default rate ~22%) are preserved.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 30000

# --- Demographics ---
sex = np.random.choice([1, 2], size=N, p=[0.40, 0.60])
education = np.random.choice([1, 2, 3, 4, 5, 6], size=N, p=[0.35, 0.39, 0.164, 0.068, 0.010, 0.018])
marriage = np.random.choice([0, 1, 2, 3], size=N, p=[0.02, 0.455, 0.535-0.02-0.01, 0.01])
age = np.clip(np.random.normal(35.5, 9.2, N).astype(int), 21, 75)

# --- Credit limit (TWD) — lognormal ---
credit_limit = np.random.lognormal(mean=11.1, sigma=0.9, size=N)
credit_limit = np.clip(credit_limit, 10000, 1000000).astype(int)
credit_limit = (credit_limit // 1000) * 1000  # round to nearest 1000

# --- Payment status (PAY_0 through PAY_6) ---
# -1 = pay duly, 0 = min pay, 1..9 = months delayed
def gen_pay_status(n):
    return np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                            size=n, p=[0.05, 0.30, 0.32, 0.12, 0.10, 0.05, 0.03, 0.01, 0.01, 0.005, 0.005])

pay_0 = gen_pay_status(N)
pay_2 = np.clip(pay_0 + np.random.randint(-1, 2, N), -2, 8)
pay_3 = np.clip(pay_2 + np.random.randint(-1, 2, N), -2, 8)
pay_4 = np.clip(pay_3 + np.random.randint(-1, 2, N), -2, 8)
pay_5 = np.clip(pay_4 + np.random.randint(-1, 2, N), -2, 8)
pay_6 = np.clip(pay_5 + np.random.randint(-1, 2, N), -2, 8)

# --- Bill amounts ---
util_base = np.random.beta(2, 3, N)  # utilization 0-1
bill1 = (credit_limit * util_base * np.random.uniform(0.7, 1.2, N)).astype(int)
bill2 = (bill1 * np.random.uniform(0.8, 1.1, N)).astype(int)
bill3 = (bill2 * np.random.uniform(0.8, 1.1, N)).astype(int)
bill4 = (bill3 * np.random.uniform(0.8, 1.1, N)).astype(int)
bill5 = (bill4 * np.random.uniform(0.8, 1.1, N)).astype(int)
bill6 = (bill5 * np.random.uniform(0.8, 1.1, N)).astype(int)

# --- Payment amounts ---
pay_ratio = np.random.beta(1.5, 4, N)
pamt1 = np.maximum(0, (bill1 * pay_ratio * np.random.uniform(0.5, 1.5, N))).astype(int)
pamt2 = np.maximum(0, (bill2 * pay_ratio * np.random.uniform(0.5, 1.5, N))).astype(int)
pamt3 = np.maximum(0, (bill3 * pay_ratio * np.random.uniform(0.5, 1.5, N))).astype(int)
pamt4 = np.maximum(0, (bill4 * pay_ratio * np.random.uniform(0.5, 1.5, N))).astype(int)
pamt5 = np.maximum(0, (bill5 * pay_ratio * np.random.uniform(0.5, 1.5, N))).astype(int)
pamt6 = np.maximum(0, (bill6 * pay_ratio * np.random.uniform(0.5, 1.5, N))).astype(int)

# --- Target: logistic model to get ~22% default rate ---
log_odds = (
    -2.5
    + 0.5 * (pay_0 > 0).astype(float)
    + 0.3 * (pay_2 > 0).astype(float)
    + 0.2 * (pay_3 > 0).astype(float)
    - 0.3 * np.log1p(credit_limit / 10000)
    + 0.2 * util_base
    - 0.1 * pay_ratio * 2
    + 0.05 * np.random.randn(N)
)
prob_default = 1 / (1 + np.exp(-log_odds))
default = (np.random.uniform(0, 1, N) < prob_default).astype(int)

print(f"Default rate: {default.mean():.3f}")

# --- Assemble DataFrame ---
df = pd.DataFrame({
    'ID': np.arange(1, N+1),
    'LIMIT_BAL': credit_limit,
    'SEX': sex,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'AGE': age,
    'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3,
    'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
    'BILL_AMT1': bill1, 'BILL_AMT2': bill2, 'BILL_AMT3': bill3,
    'BILL_AMT4': bill4, 'BILL_AMT5': bill5, 'BILL_AMT6': bill6,
    'PAY_AMT1': pamt1, 'PAY_AMT2': pamt2, 'PAY_AMT3': pamt3,
    'PAY_AMT4': pamt4, 'PAY_AMT5': pamt5, 'PAY_AMT6': pamt6,
    'default payment next month': default,
})

out_path = '/home/claude/credit-risk-morocco/data/raw/default_of_credit_card_clients.csv'
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.shape)
