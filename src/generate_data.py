"""
Customer data generator for the Customer Intelligence Platform.
Generates realistic synthetic data with statistically meaningful
relationships between features and churn.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker("en_IN")   # Indian locale — names, cities match your background
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ── helpers ────────────────────────────────────────────────────────────────────

def churn_probability(tenure, spend, tickets, plan, age):
    """
    Compute churn probability from feature values.
    This is the core business logic — churn is NOT random.
    Real insight: short tenure + high tickets + low spend = likely churner.
    """
    prob = 0.10                              # base rate 10%

    # tenure effect: new customers churn more
    if tenure < 6:
        prob += 0.25
    elif tenure < 12:
        prob += 0.10
    elif tenure > 36:
        prob -= 0.08                         # loyal customers less likely to churn

    # support tickets: biggest driver
    if tickets >= 5:
        prob += 0.30
    elif tickets >= 3:
        prob += 0.15

    # spend effect: high spenders are more invested
    if spend < 50:
        prob += 0.12
    elif spend > 300:
        prob -= 0.10

    # plan effect
    plan_modifiers = {"Basic": 0.15, "Standard": 0.0, "Premium": -0.12, "Enterprise": -0.18}
    prob += plan_modifiers.get(plan, 0)

    # age effect: very young customers explore more options
    if age < 25:
        prob += 0.08

    return max(0.02, min(0.97, prob))        # clamp between 2% and 97%


def generate_customer():
    """Generate one customer record with correlated feature values."""
    plan = random.choices(
        ["Basic", "Standard", "Premium", "Enterprise"],
        weights=[35, 40, 18, 7]
    )[0]

    tenure = int(np.random.exponential(scale=18))   # most customers < 2 years
    tenure = max(1, min(tenure, 72))                # cap at 6 years

    # spend correlates with plan
    spend_ranges = {
        "Basic":      (15,  80),
        "Standard":   (60,  200),
        "Premium":    (180, 500),
        "Enterprise": (400, 1200),
    }
    lo, hi = spend_ranges[plan]
    monthly_spend = round(random.uniform(lo, hi), 2)

    # support tickets: negatively correlated with tenure (newer = more confused)
    ticket_base = max(0, int(np.random.poisson(lam=max(0.5, 4 - tenure / 12))))
    num_tickets = min(ticket_base, 15)

    age = int(np.random.normal(loc=38, scale=12))
    age = max(18, min(75, age))

    region = random.choices(
        ["North", "South", "East", "West", "Central"],
        weights=[22, 28, 18, 20, 12]
    )[0]

    # churn label — derived from features, not random
    prob = churn_probability(tenure, monthly_spend, num_tickets, plan, age)
    churn_flag = int(random.random() < prob)

    # satisfaction score: inversely related to tickets, directly to tenure
    satisfaction = round(
        max(1.0, min(5.0,
            np.random.normal(
                loc=4.2 - (num_tickets * 0.25) + (tenure / 60),
                scale=0.5
            )
        )), 1
    )

    return {
        "customer_id":          fake.uuid4(),
        "name":                 fake.name(),
        "email":                fake.email(),
        "age":                  age,
        "region":               region,
        "plan":                 plan,
        "tenure_months":        tenure,
        "monthly_spend":        monthly_spend,
        "num_support_tickets":  num_tickets,
        "satisfaction_score":   satisfaction,
        "payment_method":       random.choice(["Credit Card", "UPI", "Net Banking", "Wallet"]),
        "has_mobile_app":       random.choices([1, 0], weights=[68, 32])[0],
        "num_products_used":    random.choices([1, 2, 3, 4], weights=[40, 35, 18, 7])[0],
        "last_login_days_ago":  int(np.random.exponential(scale=12)),
        "churn_flag":           churn_flag,
    }


# ── main generation ────────────────────────────────────────────────────────────

def generate_dataset(n=10_000, output_path="data/customers.csv"):
    print(f"Generating {n:,} customer records...")
    records = [generate_customer() for _ in range(n)]
    df = pd.DataFrame(records)

    # cap last_login at 180 days
    df["last_login_days_ago"] = df["last_login_days_ago"].clip(upper=180)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}\n")
    return df


def print_summary(df):
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total records   : {len(df):,}")
    print(f"Churn rate      : {df['churn_flag'].mean():.1%}")
    print(f"Avg tenure      : {df['tenure_months'].mean():.1f} months")
    print(f"Avg monthly spend: ₹{df['monthly_spend'].mean():.2f}")
    print(f"Avg support tkts: {df['num_support_tickets'].mean():.2f}")
    print(f"Avg satisfaction: {df['satisfaction_score'].mean():.2f}/5.0")
    print()
    print("── Churn rate by plan ──")
    print(df.groupby("plan")["churn_flag"].mean().sort_values(ascending=False).map("{:.1%}".format).to_string())
    print()
    print("── Churn rate by region ──")
    print(df.groupby("region")["churn_flag"].mean().sort_values(ascending=False).map("{:.1%}".format).to_string())
    print()
    print("── Plan distribution ──")
    print(df["plan"].value_counts().to_string())
    print("=" * 50)


if __name__ == "__main__":
    df = generate_dataset(n=10_000)
    print_summary(df)
