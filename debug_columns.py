import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np

# Load the data
train_df = pd.read_csv("artifacts/train.csv")
test_df = pd.read_csv("artifacts/test.csv")

print("Train DataFrame Info:")
print(f"Shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print("\nFirst few rows:")
print(train_df.head())

# Check what happens when we drop columns
target_column_name = "credit_score"
columns_to_drop = ["credit_score", "user_id"]

input_feature_train_df = train_df.drop(columns=columns_to_drop, axis=1)
print(f"\nAfter dropping columns:")
print(f"Input features shape: {input_feature_train_df.shape}")
print(f"Input features columns: {input_feature_train_df.columns.tolist()}")

# Define the columns that should be processed
numerical_columns = ['age',
'monthly_income_usd',
'monthly_expenses_usd',
'savings_usd',
'loan_amount_usd',
'loan_term_months',
'monthly_emi_usd',
'loan_interest_rate_pct',
'debt_to_income_ratio',
'savings_to_income_ratio']

categorical_columns = ['gender',
'education_level',
'employment_status',
'job_title',
'has_loan',
'loan_type',
'region',
'record_date']

# Check which columns actually exist
available_numerical = [col for col in numerical_columns if col in input_feature_train_df.columns]
available_categorical = [col for col in categorical_columns if col in input_feature_train_df.columns]

print(f"\nAvailable numerical columns: {available_numerical}")
print(f"Available categorical columns: {available_categorical}")

# Check if any columns are missing
missing_numerical = [col for col in numerical_columns if col not in input_feature_train_df.columns]
missing_categorical = [col for col in categorical_columns if col not in input_feature_train_df.columns]

print(f"\nMissing numerical columns: {missing_numerical}")
print(f"Missing categorical columns: {missing_categorical}")

# Check data types
print(f"\nData types of available numerical columns:")
for col in available_numerical:
    print(f"{col}: {input_feature_train_df[col].dtype}")

print(f"\nData types of available categorical columns:")
for col in available_categorical:
    print(f"{col}: {input_feature_train_df[col].dtype}") 