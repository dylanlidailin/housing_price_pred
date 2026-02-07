import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

price_data = pd.read_csv("house_prices.csv")

# 1. Enhanced Feature Engineering with Trend & Spatial Placeholders
def create_settlyfe_features(df):
    # Target: predict price 3 months in future
    df["target"] = df["price"].shift(-3)
    
    # Temporal: Seasonal Sine/Cosine
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Temporal: Long-term Market Trend (Months since baseline)
    # We use the index start as our t=0
    df["market_trend"] = (df.index.year - df.index.year.min()) * 12 + df.index.month
    
    # Market Momentum
    df["price_momentum"] = df["price"].pct_change(3)
    
    # Outlier Filter Placeholder
    # For index data, no negative prices or extreme single-month jumps
    df = df[df['price'] > 0] 
    
    return df.dropna()

# 2. Updated Backtesting Engine for Confidence Intervals
def backtest_with_intervals(data, predictors, start=24, step=6):
    all_predictions = []
    
    # Point Estimate Model
    rf = RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=1)
    
    # Quantile Models for Valuation Range
    # alpha=0.1 captures the 10th percentile (Lower Bound)
    # alpha=0.9 captures the 90th percentile (Upper Bound)
    low_model = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=50)
    high_model = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=50)
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Fit all three models
        rf.fit(train[predictors], train["target"])
        low_model.fit(train[predictors], train["target"])
        high_model.fit(train[predictors], train["target"])
        
        # Generate predictions
        preds = rf.predict(test[predictors])
        lower_bound = low_model.predict(test[predictors])
        upper_bound = high_model.predict(test[predictors])
        
        combined = pd.DataFrame({
            "actual": test["target"],
            "prediction": preds,
            "lower_range": lower_bound,
            "upper_range": upper_bound
        }, index=test.index)
        
        all_predictions.append(combined)
        
    return pd.concat(all_predictions)

# 3. Execution and Range Visualization
predictors = ["CPIAUCSL", "MORTGAGE30US", "month_sin", "month_cos", "price_momentum", "market_trend"]

# Prepare data
price_data_enriched = create_settlyfe_features(price_data)

# Run enriched backtest
predictions = backtest_with_intervals(price_data_enriched, predictors)

# 4. Accuracy Calculation (MdAPE)
mdape = np.median(np.abs((predictions["actual"] - predictions["prediction"]) / predictions["actual"])) * 100
print(f"Settlyfe LyfeEstimate Accuracy (MdAPE): {mdape:.2f}%")

# Plotting the Zillow-style Valuation Range (Confidence interval)
plt.figure(figsize=(12, 6))
plt.plot(predictions.index, predictions["actual"], label="Actual Market Value", color='black', linewidth=2)
plt.plot(predictions.index, predictions["prediction"], label="LyfeEstimate (Point)", color='blue', linestyle='--')
plt.fill_between(predictions.index, predictions["lower_range"], predictions["upper_range"], 
                 color='blue', alpha=0.2, label="90% Confidence Range")
plt.title("LyfeEstimate: Property Value Prediction with Range Intervals")
plt.xlabel("Date")
plt.ylabel("Value Index")
plt.legend()
plt.show()