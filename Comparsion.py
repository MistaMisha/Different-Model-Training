import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_and_prepare_data():
    """Load forecast and actual sales data"""
    print("Loading forecast and actual sales data...")

    # Load forecast data
    try:
        forecast_df = pd.read_csv('simple_forecast.csv')
        print(f"✅ Loaded {len(forecast_df)} forecast records")
    except FileNotFoundError:
        print("❌ Error: simple_forecast.csv not found.")
        return None, None

    # Load actual sales data
    try:
        actual_df = pd.read_csv('actual_2weeks.csv')
        print(f"✅ Loaded {len(actual_df)} actual sales records")

        # Aggregate actual sales (2-week totals)
        actual_summary = actual_df.groupby(
            ['product_id', 'variant_name', 'product_type', 'product_attribute',
             'outlet_id', 'outlet_name']
        ).agg({
            'quantity': 'sum'
        }).reset_index()

        actual_summary.rename(columns={'quantity': 'actual_sales'}, inplace=True)
        return forecast_df, actual_summary

    except FileNotFoundError:
        print("❌ Error: actual_2weeks.csv not found.")
        return None, None


def prepare_comparison_data(forecast_df, actual_df):
    """Prepare comparison dataset by merging forecast with actual sales"""
    print("Preparing forecast comparison data...")

    # Merge on product + outlet keys
    comparison = forecast_df.merge(
        actual_df,
        on=['product_id', 'variant_name', 'product_type',
            'product_attribute', 'outlet_id', 'outlet_name'],
        how='outer'
    )

    # Fill missing values
    comparison['recommended_stock_2w'] = comparison['recommended_stock_2w'].fillna(0)
    comparison['actual_sales'] = comparison['actual_sales'].fillna(0)

    # Error calculations
    comparison['difference'] = comparison['recommended_stock_2w'] - comparison['actual_sales']
    comparison['abs_error'] = comparison['difference'].abs()

    # Avoid divide by zero
    comparison['pct_error'] = np.where(
        comparison['actual_sales'] > 0,
        (comparison['abs_error'] / comparison['actual_sales']) * 100,
        np.nan
    )

    return comparison


def calculate_metrics(comparison):
    """Calculate overall forecast accuracy metrics"""
    print("Calculating forecast accuracy metrics...")

    total_forecasted = comparison['recommended_stock_2w'].sum()
    total_actual = comparison['actual_sales'].sum()
    total_difference = total_forecasted - total_actual

    mae = comparison['abs_error'].mean()
    mape = comparison['pct_error'].mean(skipna=True)

    metrics = {
        "Total Forecasted": total_forecasted,
        "Total Actual": total_actual,
        "Difference": total_difference,
        "Overall Accuracy": 1 - (abs(total_difference) / total_actual if total_actual else 0),
        "Mean Absolute Error": mae,
        "Mean Absolute Percentage Error": mape
    }

    return metrics


def generate_report(comparison, metrics):
    """Generate CSV and print metrics"""
    today = datetime.today().strftime('%Y%m%d')
    output_file = f"forecast_comparison_{today}.csv"

    comparison.to_csv(output_file, index=False)
    print(f"\n📂 Forecast comparison saved as: {output_file}")

    print("\n📊 FORECAST ACCURACY REPORT")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:,.2f}")
        else:
            print(f"{k}: {v}")


def main():
    forecast_df, actual_df = load_and_prepare_data()
    if forecast_df is None or actual_df is None:
        return

    comparison = prepare_comparison_data(forecast_df, actual_df)
    metrics = calculate_metrics(comparison)
    generate_report(comparison, metrics)


if __name__ == "__main__":
    main()
