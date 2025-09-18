import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def compare_forecast_vs_actual(forecast_file='hybrid_forecast.csv', actual_file='actual_2weeks.csv'):
    """
    Compare your forecasts against actual 2-week sales data
    """
    
    print("🔍 Loading forecast and actual data...")
    
    # Load the data
    forecast_df = pd.read_csv(forecast_file)
    actual_df = pd.read_csv(actual_file)
    
    print(f"📊 Loaded {len(forecast_df)} forecasts and {len(actual_df)} actual sales records")
    
    # Prepare actual data - aggregate to match forecast structure
    print("📈 Aggregating actual sales by variant and outlet...")
    
    # If actual data has date column, might need to aggregate over the 2-week period
    if 'created_at' in actual_df.columns or 'date' in actual_df.columns:
        date_col = 'created_at' if 'created_at' in actual_df.columns else 'date'
        actual_df[date_col] = pd.to_datetime(actual_df[date_col])
    
    # Aggregate actual sales by variant and outlet
    actual_agg = actual_df.groupby(['variant_name', 'outlet_name'], as_index=False).agg({
        'quantity': 'sum'
    }).rename(columns={'quantity': 'actual_sales_2w'})
    
    # Merge with forecasts
    print("🔗 Merging forecasts with actual sales...")
    
    # Match on variant_name and outlet_name
    comparison = forecast_df.merge(
        actual_agg, 
        on=['variant_name', 'outlet_name'], 
        how='left'
    )
    
    # Fill missing actuals with 0 (products that had forecasts but no sales)
    comparison['actual_sales_2w'] = comparison['actual_sales_2w'].fillna(0)
    
    # Also check for products that had sales but no forecast (right join)
    missed_products = actual_agg.merge(
        forecast_df[['variant_name', 'outlet_name']], 
        on=['variant_name', 'outlet_name'], 
        how='left', 
        indicator=True
    )
    missed_count = len(missed_products[missed_products['_merge'] == 'left_only'])
    
    print(f"📋 Comparison Results:")
    print(f"   • Products with both forecast & actual: {len(comparison)}")
    print(f"   • Products with sales but no forecast: {missed_count}")
    
    return analyze_forecast_accuracy(comparison)

def analyze_forecast_accuracy(comparison_df):
    """
    Comprehensive analysis of forecast accuracy
    """
    
    print("\n🎯 FORECAST ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    df = comparison_df.copy()
    df = df[df['actual_sales_2w'] >= 0]  # Remove any negative values
    
    total_forecasted = df['recommended_stock_2w'].sum()
    total_actual = df['actual_sales_2w'].sum()
    
    print(f"📊 OVERALL TOTALS:")
    print(f"   • Total Forecasted: {total_forecasted:,} units")
    print(f"   • Total Actual:     {total_actual:,} units")
    print(f"   • Difference:       {total_forecasted - total_actual:+,} units")
    print(f"   • Overall Accuracy: {(1 - abs(total_forecasted - total_actual) / max(total_actual, 1)) * 100:.1f}%")
    
    # Calculate accuracy metrics
    df['absolute_error'] = abs(df['recommended_stock_2w'] - df['actual_sales_2w'])
    df['percentage_error'] = np.where(
        df['actual_sales_2w'] > 0,
        df['absolute_error'] / df['actual_sales_2w'] * 100,
        np.where(df['recommended_stock_2w'] <= 1, 10, 100)  # Low penalty for predicting 1 when actual was 0
    )
    
    # Cap percentage error at 200% for extreme cases
    df['percentage_error'] = np.minimum(df['percentage_error'], 200)
    
    # Categorize predictions
    df['prediction_category'] = 'Exact Match'
    df.loc[df['absolute_error'] > 0, 'prediction_category'] = 'Close (±1-3)'
    df.loc[df['absolute_error'] > 3, 'prediction_category'] = 'Moderate Error (±4-10)'
    df.loc[df['absolute_error'] > 10, 'prediction_category'] = 'Large Error (±11+)'
    
    df.loc[(df['actual_sales_2w'] == 0) & (df['recommended_stock_2w'] <= 2), 'prediction_category'] = 'Conservative (Low/No Sales)'
    df.loc[(df['actual_sales_2w'] > 0) & (df['recommended_stock_2w'] == 0), 'prediction_category'] = 'Missed Sales'
    
    print(f"\n📈 ACCURACY METRICS:")
    print(f"   • Mean Absolute Error (MAE):    {df['absolute_error'].mean():.2f} units")
    print(f"   • Median Absolute Error:        {df['absolute_error'].median():.2f} units")
    print(f"   • Mean Percentage Error (MAPE): {df['percentage_error'].mean():.1f}%")
    print(f"   • Median Percentage Error:      {df['percentage_error'].median():.1f}%")
    
    # Accuracy by category
    print(f"\n🎯 PREDICTION BREAKDOWN:")
    category_counts = df['prediction_category'].value_counts()
    for category, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"   • {category:<25}: {count:>6} ({pct:>5.1f}%)")
    
    # Performance by volume category
    if 'volume_category' in df.columns:
        print(f"\n📊 ACCURACY BY VOLUME CATEGORY:")
        vol_analysis = df.groupby('volume_category').agg({
            'percentage_error': ['mean', 'median', 'count'],
            'absolute_error': 'mean'
        }).round(2)
        print(vol_analysis.to_string())
    
    # Find best and worst predictions
    print(f"\n🏆 BEST PREDICTIONS (Exact or Close):")
    best = df[df['absolute_error'] <= 1].head(10)
    if len(best) > 0:
        for _, row in best.iterrows():
            print(f"   • {row['variant_name'][:30]:<30} | Forecast: {row['recommended_stock_2w']:>3} | Actual: {row['actual_sales_2w']:>3}")
    
    print(f"\n⚠️  WORST PREDICTIONS (Large Errors):")
    worst = df.nlargest(10, 'absolute_error')
    for _, row in worst.iterrows():
        error_type = "Over" if row['recommended_stock_2w'] > row['actual_sales_2w'] else "Under"
        print(f"   • {row['variant_name'][:30]:<30} | Forecast: {row['recommended_stock_2w']:>3} | Actual: {row['actual_sales_2w']:>3} | {error_type}-predicted by {row['absolute_error']:.0f}")
    
    # Products with zero sales but high forecasts (potential overstock)
    overstock_risk = df[(df['actual_sales_2w'] == 0) & (df['recommended_stock_2w'] > 5)]
    if len(overstock_risk) > 0:
        print(f"\n📦 POTENTIAL OVERSTOCK (Forecast >5, Actual 0): {len(overstock_risk)} products")
        print("   Top overstock risks:")
        for _, row in overstock_risk.nlargest(5, 'recommended_stock_2w').iterrows():
            print(f"   • {row['variant_name'][:30]:<30} | Forecasted: {row['recommended_stock_2w']} units")
    
    # Products with sales but no/low forecast (potential stockout)
    stockout_risk = df[(df['actual_sales_2w'] > 5) & (df['recommended_stock_2w'] <= 2)]
    if len(stockout_risk) > 0:
        print(f"\n🚨 POTENTIAL STOCKOUTS (Actual >5, Forecast ≤2): {len(stockout_risk)} products")
        print("   Top stockout risks:")
        for _, row in stockout_risk.nlargest(5, 'actual_sales_2w').iterrows():
            print(f"   • {row['variant_name'][:30]:<30} | Forecasted: {row['recommended_stock_2w']}, Actual: {row['actual_sales_2w']}")
    
    return create_accuracy_visualizations(df)

def create_accuracy_visualizations(df):
    """
    Create visualizations for forecast accuracy
    """
    print(f"\n📊 Creating accuracy visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Forecast vs Actual Sales Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Forecast vs Actual
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['actual_sales_2w'], df['recommended_stock_2w'], 
                         alpha=0.6, s=30, c='blue')
    
    # Perfect prediction line
    max_val = max(df['actual_sales_2w'].max(), df['recommended_stock_2w'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', alpha=0.7)
    
    ax1.set_xlabel('Actual Sales (2 weeks)')
    ax1.set_ylabel('Forecasted Sales (2 weeks)')
    ax1.set_title('Forecast vs Actual Sales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    df['forecast_error'] = df['recommended_stock_2w'] - df['actual_sales_2w']
    ax2.hist(df['forecast_error'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', label='Perfect Forecast')
    ax2.set_xlabel('Forecast Error (Forecast - Actual)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Forecast Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy by volume category (if available)
    ax3 = axes[1, 0]
    if 'volume_category' in df.columns:
        vol_accuracy = df.groupby('volume_category')['percentage_error'].mean().sort_values()
        vol_accuracy.plot(kind='bar', ax=ax3, color='green', alpha=0.7)
        ax3.set_title('Average Percentage Error by Volume Category')
        ax3.set_ylabel('Mean Percentage Error (%)')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Volume Category\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Volume Category Analysis')
    
    # 4. Prediction category breakdown
    ax4 = axes[1, 1]
    category_counts = df['prediction_category'].value_counts()
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'purple']
    wedges, texts, autotexts = ax4.pie(category_counts.values, 
                                      labels=category_counts.index, 
                                      autopct='%1.1f%%',
                                      colors=colors[:len(category_counts)],
                                      startangle=90)
    ax4.set_title('Prediction Accuracy Breakdown')
    
    plt.tight_layout()
    plt.savefig('forecast_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Saved visualization as 'forecast_accuracy_analysis.png'")
    
    # Create summary report
    create_summary_report(df)
    
    return df

def create_summary_report(df):
    """
    Create a summary report file
    """
    report_lines = []
    report_lines.append("FORECAST ACCURACY SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Key metrics
    total_forecasted = df['recommended_stock_2w'].sum()
    total_actual = df['actual_sales_2w'].sum()
    overall_accuracy = (1 - abs(total_forecasted - total_actual) / max(total_actual, 1)) * 100
    
    report_lines.append("KEY METRICS:")
    report_lines.append(f"  • Total Products Analyzed: {len(df):,}")
    report_lines.append(f"  • Total Forecasted Units: {total_forecasted:,}")
    report_lines.append(f"  • Total Actual Units: {total_actual:,}")
    report_lines.append(f"  • Overall Accuracy: {overall_accuracy:.1f}%")
    report_lines.append(f"  • Mean Absolute Error: {df['absolute_error'].mean():.2f} units")
    report_lines.append(f"  • Median Absolute Error: {df['absolute_error'].median():.2f} units")
    report_lines.append("")
    
    # Accuracy breakdown
    report_lines.append("PREDICTION ACCURACY:")
    category_counts = df['prediction_category'].value_counts()
    for category, count in category_counts.items():
        pct = count / len(df) * 100
        report_lines.append(f"  • {category}: {count:,} products ({pct:.1f}%)")
    
    # Save report
    with open('forecast_accuracy_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("💾 Saved detailed report as 'forecast_accuracy_report.txt'")
    
    return df

# Usage example
if __name__ == "__main__":
    # Run the comparison
    results = compare_forecast_vs_actual(
        forecast_file='hybrid_forecast.csv',
        actual_file='actual_2weeks.csv'  # Your actual sales file
    )
    
    print("\n✅ Analysis complete! Check the generated files:")
    print("   • forecast_accuracy_analysis.png (visualizations)")
    print("   • forecast_accuracy_report.txt (detailed report)")

