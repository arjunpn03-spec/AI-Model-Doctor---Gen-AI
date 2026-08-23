import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 1000

# Generate realistic customer data
tenure = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
total_charges = tenure * monthly_charges + np.random.normal(0, 25, size=n_samples)
contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20]) # 0: Month-to-month, 1: One year, 2: Two year
support_calls = np.random.poisson(lam=1.5, size=n_samples)

# Create an imbalanced churn outcome (Only ~20% churn - typical "sick" data challenge)
churn_prob = (
    0.35 * (contract_type == 0) + 
    0.25 * (monthly_charges > 80) + 
    0.20 * (support_calls > 3) - 
    0.20 * (tenure > 36)
)
churn_prob = np.clip(churn_prob, 0.05, 0.85)
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

# Assemble into DataFrame
df = pd.DataFrame({
    'Tenure': tenure,
    'MonthlyCharges': np.round(monthly_charges, 2),
    'TotalCharges': np.round(total_charges, 2),
    'ContractType': contract_type,
    'SupportCalls': support_calls,
    'Churn': churn
})

# Save dataset
df.to_csv('data/churn_data.csv', index=False)
print("✅ Created dataset at data/churn_data.csv with shape:", df.shape)
print("Class distribution (Churn vs Retained):\n", df['Churn'].value_counts(normalize=True))