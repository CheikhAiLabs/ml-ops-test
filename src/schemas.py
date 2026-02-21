"""Data validation schemas using Pandera."""

import pandera.pandas as pa

churn_schema = pa.DataFrameSchema(
    columns={
        "age": pa.Column(
            int,
            checks=[
                pa.Check.in_range(18, 100),
            ],
            description="Customer age",
        ),
        "tenure_months": pa.Column(
            int,
            checks=[
                pa.Check.in_range(0, 120),
            ],
            description="Months as customer",
        ),
        "monthly_charges": pa.Column(
            float,
            checks=[
                pa.Check.gt(0),
                pa.Check.lt(500),
            ],
            description="Monthly bill amount",
            coerce=True,
        ),
        "contract_type": pa.Column(
            int,
            checks=[
                pa.Check.isin([0, 1, 2]),
            ],
            description="0=month-to-month, 1=one-year, 2=two-year",
        ),
        "num_tickets": pa.Column(
            int,
            checks=[
                pa.Check.ge(0),
                pa.Check.le(50),
            ],
            description="Support tickets opened",
            coerce=True,
        ),
        "churn": pa.Column(
            int,
            checks=[
                pa.Check.isin([0, 1]),
            ],
            description="1=churned, 0=retained",
        ),
    },
    strict=True,
    coerce=True,
    description="Schema for the churn prediction dataset",
)
