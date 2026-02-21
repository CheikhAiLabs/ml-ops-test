"""Generate a 2000-row churn dataset with clearly separable patterns."""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(18, 75, n)
tenure = np.random.randint(0, 72, n)
charges = np.round(np.random.uniform(20, 120, n), 2)
contract = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])
tickets = np.random.randint(0, 10, n)

# Build a probability of churn based on realistic, separable patterns
logit = (
    -5.0
    + 4.0 * (contract == 0).astype(float)
    - 3.0 * (contract == 2).astype(float)
    - 0.10 * tenure
    + 0.03 * charges
    + 0.8 * tickets
    - 0.02 * age
)
prob = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, prob)

df = pd.DataFrame(
    {
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": charges,
        "contract_type": contract,
        "num_tickets": tickets,
        "churn": churn,
    }
)

out = "data/raw/churn.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}, churn rate: {churn.mean():.2%}")
