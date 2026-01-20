import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load dataset
#raw_house_data = pd.read_csv('house_prices.csv')
#raw_house_data = raw_house_data.dropna()

from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()
print(housing_data.DESCR)

def restructure_data(df):
    # 1. Clean Numerical Columns (Extract numbers from strings)
    df['area_sqft'] = df['Carpet Area'].str.extract('(\d+)').astype(float)
    df['bathrooms'] = pd.to_numeric(df['Bathroom'], errors='coerce').fillna(1)
    df['balcony'] = pd.to_numeric(df['Balcony'], errors='coerce').fillna(0)
    
    # 2. Handle Outliers (Documentation Requirement: Catch 20+ bedrooms)
    # Based on doc, standard should be ~300 sqft per bedroom
    df = df[df['area_sqft'] / df['bathrooms'] > 100] 
    
    # 3. Categorical Encoding (One-Hot Encoding)
    categorical_cols = ['Furnishing', 'Status', 'Transaction', 'facing']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Temporal Trend (Settlyfe Requirement: Long-term direction)
    df['market_trend'] = (df.index.year - df.index.year.min()) * 12 + df.index.month
    
    return df.dropna()

# Restructured Dataset
processed_data = restructure_data(raw_house_data)


def train_with_intervals(X_train, y_train, X_test):
    # Base Point Estimate
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    
    # Range Intervals (Quantile Regression)
    # alpha=0.1 is 10th percentile (lower bound), alpha=0.9 is 90th (upper bound)
    low_model = GradientBoostingRegressor(loss='quantile', alpha=0.1)
    high_model = GradientBoostingRegressor(loss='quantile', alpha=0.9)
    
    low_model.fit(X_train, y_train)
    high_model.fit(X_train, y_train)
    
    predictions = pd.DataFrame({
        "estimate": rf.predict(X_test),
        "lower_bound": low_model.predict(X_test),
        "upper_bound": high_model.predict(X_test)
    })
    
    return predictions