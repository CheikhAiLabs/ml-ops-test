"""Generate a realistic churn dataset with 13 features."""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# -- Demographics --
gender = np.random.choice([0, 1], n)
age = np.random.normal(45, 14, n).clip(18, 80).astype(int)
partner = np.random.choice([0, 1], n, p=[0.48, 0.52])
dependents = np.array(
    [int(np.random.random() < (0.55 if p else 0.15)) for p in partner]
)

# -- Tenure (bimodal: many new + many long-term) --
t1 = np.random.exponential(8, n // 3).clip(1, 72)
t2 = np.random.normal(35, 12, n // 3).clip(1, 72)
t3 = np.random.normal(58, 10, n - 2 * (n // 3)).clip(1, 72)
tenure_months = np.concatenate([t1, t2, t3]).astype(int)
np.random.shuffle(tenure_months)

# -- Contract (correlated with tenure) --
contract_type = np.array(
    [
        np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
        if t < 12
        else np.random.choice([0, 1, 2], p=[0.40, 0.35, 0.25])
        if t < 36
        else np.random.choice([0, 1, 2], p=[0.20, 0.30, 0.50])
        for t in tenure_months
    ]
)

# -- Internet service --
internet_service = np.random.choice([0, 1, 2], n, p=[0.22, 0.34, 0.44])

# -- Monthly charges (depends on internet tier) --
monthly_charges = (
    np.array(
        [
            np.random.normal(25, 5)
            if s == 0
            else np.random.normal(55, 12)
            if s == 1
            else np.random.normal(85, 14)
            for s in internet_service
        ]
    )
    .clip(18.25, 118.75)
    .round(2)
)

# -- Add-on services (require internet) --
online_security = np.where(
    internet_service > 0, (np.random.random(n) < 0.5).astype(int), 0
)
tech_support = np.where(
    internet_service > 0, (np.random.random(n) < 0.5).astype(int), 0
)

# -- Billing --
payment_method = np.random.choice([0, 1, 2, 3], n, p=[0.22, 0.22, 0.34, 0.22])
paperless_billing = np.random.choice([0, 1], n, p=[0.40, 0.60])

# -- Support tickets (correlated with dissatisfaction) --
base_tickets = np.random.poisson(1.5, n)
for i in range(n):
    if internet_service[i] == 2 and tech_support[i] == 0:
        base_tickets[i] += np.random.poisson(2)
    if contract_type[i] == 0:
        base_tickets[i] += np.random.poisson(0.5)
num_tickets = base_tickets.clip(0, 10).astype(int)

# -- Churn (logistic — stronger signals, less noise) --
logit = (
    -1.5
    + 1.8 * (contract_type == 0).astype(float)
    - 1.2 * (contract_type == 2).astype(float)
    + 0.018 * monthly_charges
    - 0.05 * tenure_months
    + 0.9 * (internet_service == 2).astype(float)
    - 0.7 * online_security
    - 0.6 * tech_support
    + 0.7 * (payment_method == 2).astype(float)
    + 0.35 * paperless_billing
    - 0.4 * partner
    - 0.35 * dependents
    + 0.25 * num_tickets
    + 0.2 * (age >= 65).astype(float)
    + np.random.normal(0, 0.25, n)
)
churn_prob = 1 / (1 + np.exp(-logit))
churn = (np.random.random(n) < churn_prob).astype(int)

df = pd.DataFrame(
    {
        "gender": gender,
        "age": age,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "num_tickets": num_tickets,
        "churn": churn,
    }
)

df.to_csv("data/raw/churn.csv", index=False)
print(f"Churn rate: {churn.mean():.1%}")
print(f"Shape: {df.shape}")
print(df.head(5))
