import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import tkinter as tk
from tkinter import filedialog

# ---------------- Constants ----------------
PAST_HORIZON = 14
FUTURE_HORIZON = 14
LAGS = [1, 7]
ROLLING_WINDOWS = [7]
CAT_COLS = [
    'product_type', 'variant_name', 'product_attribute',
    'promo_name', 'promo_code', 'outlet_id', 'outlet_name'
]

# ---------------- Feature Engineering ----------------
def create_features(df):
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values(['variant_name', 'created_at'])
    
    # Calendar features
    df['day_of_week'] = df['created_at'].dt.weekday
    df['month'] = df['created_at'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # Lag features
    for lag in LAGS:
        df[f'lag_{lag}'] = df.groupby('variant_name')['quantity'].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f'roll_mean_{window}'] = df.groupby('variant_name')['quantity'].shift(1).rolling(window).mean()
    
    # Fill NAs with median per product
    df.fillna(df.groupby('variant_name')['quantity'].transform('median'), inplace=True)
    return df

# ---------------- Encode Categoricals ----------------
def encode_categoricals(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("NA")
            df[col] = df[col].astype("category").cat.codes
    return df

# ---------------- Create Multi-Horizon Targets ----------------
def create_targets(df, horizon=FUTURE_HORIZON):
    df = df.copy()
    for h in range(1, horizon+1):
        df[f'sales_t+{h}'] = df.groupby('variant_name')['quantity'].shift(-h)
    return df

# ---------------- Train LightGBM ----------------
def train_model(df):
    df = encode_categoricals(df, CAT_COLS)
    df = create_features(df)
    df = create_targets(df)
    
    target_cols = [f'sales_t+{h}' for h in range(1, FUTURE_HORIZON+1)]
    df_train = df.dropna(subset=target_cols)
    
    feature_cols = [c for c in df_train.columns if c not in target_cols + ['created_at', 'variant_name']]
    X = df_train[feature_cols]
    y = df_train[target_cols]
    
    model = MultiOutputRegressor(LGBMRegressor(
        objective='regression',
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ))
    model.fit(X, y)
    return model, feature_cols

# ---------------- Forecast Function ----------------
def forecast(model, df_latest, feature_cols):
    df_feat = encode_categoricals(df_latest, CAT_COLS)
    df_feat = create_features(df_feat)
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0
    X = df_feat[feature_cols]
    
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, None)
    
    forecast_df = df_feat[['variant_name', 'created_at']].copy()
    for h in range(FUTURE_HORIZON):
        forecast_df[f'day_{h+1}'] = np.round(y_pred[:,h]).astype(int)
    forecast_df['recommended_stock_2w'] = forecast_df[[f'day_{h+1}' for h in range(FUTURE_HORIZON)]].sum(axis=1)
    return forecast_df

# ---------------- Combine Past + Future ----------------
def combined_past_future(df, forecast_df):
    # Aggregate duplicates per variant_name + created_at
    past_df = (
        df.groupby(['variant_name', 'created_at'])['quantity']
          .sum()
          .reset_index()
    )

    # Keep only last PAST_HORIZON days per variant
    past_df = past_df.groupby('variant_name').apply(
        lambda x: x.sort_values('created_at').tail(PAST_HORIZON)
    ).reset_index(drop=True)
    
    # Pivot to wide format
    past_wide = past_df.pivot(index='variant_name', columns='created_at', values='quantity')
    past_wide.columns = [f'past_{i+1}' for i in range(past_wide.shape[1])]
    
    combined = forecast_df.merge(past_wide, on='variant_name', how='left')
    return combined

# ---------------- Top 5 Summary ----------------
def summarize_top5(df, forecast_df, past_horizon=PAST_HORIZON, future_horizon=FUTURE_HORIZON):
    # Past 2 weeks sum
    past_summary = (
        df.groupby('variant_name')['quantity']
          .tail(past_horizon)
          .groupby(df['variant_name']).sum()
          .reset_index()
          .rename(columns={'quantity':'Actual_Past_2w'})
    )

    # Future 2 weeks sum
    forecast_cols = [f'day_{i+1}' for i in range(future_horizon)]
    future_summary = forecast_df.copy()
    future_summary['Forecast_Next_2w'] = future_summary[forecast_cols].sum(axis=1)
    future_summary = future_summary[['variant_name', 'Forecast_Next_2w']]
    
    # Merge past + future
    summary = past_summary.merge(future_summary, on='variant_name', how='inner')
    top5 = summary.sort_values('Forecast_Next_2w', ascending=False).head(5)
    return top5

# ---------------- CSV Upload ----------------
def load_csv():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not file_path:
        print("No file selected.")
        return None
    df = pd.read_csv(file_path)
    return df

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Select CSV file with columns: variant_name, created_at, quantity")
    df = load_csv()
    if df is None:
        exit()
    
    # Train model
    print("Training model...")
    model, features = train_model(df)
    
    # Forecast using latest snapshot
    latest_df = df.groupby('variant_name').tail(1)
    forecast_df = forecast(model, latest_df, features)
    
    # Combine past 2 weeks + future 2 weeks forecast
    combined_df = combined_past_future(df, forecast_df)
    
    # Show top 5 summary
    top5_summary = summarize_top5(df, forecast_df)
    print("\nTop 5 Products (Past 2 Weeks vs Next 2 Weeks Forecast):")
    print(top5_summary.to_string(index=False))
    
    # Save full forecast with daily past/future to CSV
    combined_df.to_csv("forecast_2w_combined.csv", index=False)
    print("\nFull forecast saved to forecast_2w_combined.csv")
