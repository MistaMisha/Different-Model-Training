import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import joblib
import secrets
from auth import auth_bp 
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# ---------- Constants ----------
UPLOAD_FOLDER = 'uploads'
DATA_FILE = 'all_data.csv'
FORECAST_FILE = 'hybrid_forecast_results.json'

# ---------- Flask Setup ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Secret key setup
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Register the auth blueprint
app.register_blueprint(auth_bp)

# ======================================================================
#                    HYBRID FORECASTING WITH TIME SERIES FOR HIGH-VOLUME
# ======================================================================

def identify_product_type_enhanced(variant_name, product_type, avg_daily_sales, max_daily_sales, sales_frequency):
    """
    Enhanced product type identification with stricter thresholds
    """
    variant_lower = str(variant_name).lower()
    
    # High-confidence consumable keywords
    consumable_keywords = [
        'ultra bar', 'vaporesso', 'pod', 'coil', 'disposable', 'cartridge',
        'replacement', 'refill', 'atomizer', 'tank', 'clearomizer', 'terea'
    ]
    
    # Check if it's a consumable by name
    is_consumable = any(keyword in variant_lower for keyword in consumable_keywords)
    
    # Enhanced thresholds for high-volume classification
    has_high_frequency = sales_frequency > 0.4  # Sold on >40% of days
    has_consistent_volume = avg_daily_sales > 2.5  # >2.5 units per day
    has_peak_demand = max_daily_sales > 15  # At least one day with 15+ sales
    
    # Strict classification for high-volume consumables
    if is_consumable and has_high_frequency and has_consistent_volume:
        if avg_daily_sales > 10:
            return "Ultra High Consumable"
        elif avg_daily_sales > 5:
            return "Very High Consumable"
        elif avg_daily_sales > 2.5:
            return "High Consumable"
        else:
            return "Medium Consumable"
    
    # Non-consumables but high volume
    elif has_high_frequency and has_consistent_volume and has_peak_demand:
        if avg_daily_sales > 8:
            return "High Volume Product"
        else:
            return "Medium Volume Product"
    
    # Everything else gets conservative treatment
    elif avg_daily_sales > 1.5:
        return "Low-Medium Volume"
    else:
        return "Low Volume Retail"

def holt_winters_forecast(data, periods=14, alpha=None, beta=None, gamma=None):
    """
    Simple Holt-Winters exponential smoothing for high-volume items
    """
    if len(data) < 10:  # Need sufficient data
        return np.mean(data) * periods
    
    # Auto-optimize parameters if not provided
    if alpha is None or beta is None:
        def optimize_params(params):
            a, b = params
            return holt_winters_mse(data, a, b, periods)
        
        result = minimize(optimize_params, [0.3, 0.1], bounds=[(0.01, 0.9), (0.01, 0.5)])
        alpha, beta = result.x
    
    # Initialize
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    
    # Initial values
    level[0] = data[0]
    if n > 1:
        trend[0] = data[1] - data[0]
    
    # Apply Holt-Winters
    for i in range(1, n):
        level_prev = level[i-1]
        trend_prev = trend[i-1]
        
        level[i] = alpha * data[i] + (1 - alpha) * (level_prev + trend_prev)
        trend[i] = beta * (level[i] - level_prev) + (1 - beta) * trend_prev
    
    # Forecast
    forecast = (level[-1] + trend[-1] * periods)
    
    # Add some safety margin for high-volume items
    safety_margin = 1.1 if np.mean(data) > 5 else 1.05
    return max(1, forecast * safety_margin)

def holt_winters_mse(data, alpha, beta, periods):
    """Calculate MSE for parameter optimization"""
    try:
        n = len(data)
        if n < 4:
            return 1000
        
        # Use first 70% for training, rest for validation
        train_size = max(3, int(n * 0.7))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        if len(test_data) == 0:
            return 1000
        
        # Forecast
        pred = holt_winters_forecast(train_data, len(test_data), alpha, beta)
        actual = np.sum(test_data)
        
        return (pred - actual) ** 2
    except:
        return 1000

def moving_average_with_trend(data, periods=14):
    """
    Moving average with trend adjustment for medium-volume items
    """
    if len(data) < 7:
        return np.mean(data) * periods
    
    # Recent performance weighted more heavily
    recent_data = data[-7:]  # Last week
    earlier_data = data[-14:-7] if len(data) >= 14 else data[:-7]
    
    recent_avg = np.mean(recent_data)
    earlier_avg = np.mean(earlier_data) if len(earlier_data) > 0 else recent_avg
    
    # Trend factor
    trend = recent_avg / earlier_avg if earlier_avg > 0 else 1.0
    trend = max(0.5, min(2.0, trend))  # Cap extreme trends
    
    # Forecast with trend
    base_forecast = recent_avg * periods
    trending_forecast = base_forecast * trend
    
    return trending_forecast

def calculate_hybrid_forecast_enhanced(variant_data, variant_name, product_type):
    """
    Enhanced hybrid forecast using different models for different product types
    """
    # Basic statistics
    total_sales = variant_data['quantity'].sum()
    total_days = len(variant_data)
    non_zero_days = (variant_data['quantity'] > 0).sum()
    max_daily = variant_data['quantity'].max()
    
    # Calculate averages
    avg_daily_historical = total_sales / total_days
    sales_frequency = non_zero_days / total_days
    
    # Get recent data
    recent_14d = variant_data.tail(14)['quantity'].values
    recent_sales = np.sum(recent_14d)
    avg_daily_recent = recent_sales / len(recent_14d)
    
    # Enhanced product classification
    product_type_category = identify_product_type_enhanced(
        variant_name, product_type, avg_daily_historical, max_daily, sales_frequency
    )
    
    # Select forecasting method based on product type
    if product_type_category in ["Ultra High Consumable", "Very High Consumable"]:
        # TIME SERIES FORECASTING for high-volume consumables
        daily_data = variant_data['quantity'].values
        
        # Use Holt-Winters for high-volume items
        base_forecast = holt_winters_forecast(daily_data, periods=14, alpha=0.4, beta=0.2)
        
        # Additional safety stock for ultra-high volume
        if product_type_category == "Ultra High Consumable":
            safety_multiplier = 1.3  # 30% safety stock
        else:
            safety_multiplier = 1.2  # 20% safety stock
        
        forecast = base_forecast * safety_multiplier
        
        # Reasonable caps but higher than before
        if avg_daily_historical > 15:
            forecast = min(forecast, 800)  # Increased cap
        elif avg_daily_historical > 8:
            forecast = min(forecast, 500)
        elif avg_daily_historical > 4:
            forecast = min(forecast, 300)
        else:
            forecast = min(forecast, 150)
        
        forecast = max(15, round(forecast))  # Higher minimum
        
    elif product_type_category in ["High Consumable", "High Volume Product"]:
        # MOVING AVERAGE WITH TREND for high-volume items
        daily_data = variant_data['quantity'].values
        base_forecast = moving_average_with_trend(daily_data, periods=14)
        
        # Safety stock
        safety_multiplier = 1.15  # 15% safety stock
        forecast = base_forecast * safety_multiplier
        
        # Caps
        if avg_daily_historical > 10:
            forecast = min(forecast, 400)
        elif avg_daily_historical > 5:
            forecast = min(forecast, 200)
        else:
            forecast = min(forecast, 100)
        
        forecast = max(8, round(forecast))
        
    elif product_type_category == "Medium Consumable":
        # ENHANCED WEIGHTED AVERAGE for medium consumables
        weighted_daily = (avg_daily_recent * 0.6) + (avg_daily_historical * 0.4)
        
        # Trend factor
        if len(variant_data) >= 14:
            recent_7d = variant_data.tail(7)['quantity'].sum()
            previous_7d = variant_data.tail(14).head(7)['quantity'].sum()
            trend = recent_7d / previous_7d if previous_7d > 0 else 1.5 if recent_7d > 0 else 1.0
            trend = max(0.7, min(1.8, trend))
        else:
            trend = 1.1
        
        base_forecast = weighted_daily * 14 * trend
        forecast = min(base_forecast, 80)
        forecast = max(4, round(forecast))
        
    elif product_type_category in ["Medium Volume Product", "Low-Medium Volume"]:
        # CONSERVATIVE APPROACH for medium volume
        weighted_daily = (avg_daily_recent * 0.4) + (avg_daily_historical * 0.6)
        
        growth_factor = 1.1 if avg_daily_recent > avg_daily_historical * 1.2 else 1.0
        base_forecast = weighted_daily * 14 * growth_factor
        
        if product_type_category == "Medium Volume Product":
            forecast = min(base_forecast, 50)
            forecast = max(3, round(forecast))
        else:
            forecast = min(base_forecast, 30)
            forecast = max(2, round(forecast))
    
    else:
        # VERY CONSERVATIVE for low-volume retail (keep existing logic)
        base_forecast = max(avg_daily_historical, avg_daily_recent) * 14
        
        if avg_daily_historical < 0.1:
            forecast = min(base_forecast, 2)
        elif avg_daily_historical < 0.3:
            forecast = min(base_forecast, 4)
        elif avg_daily_historical < 0.7:
            forecast = min(base_forecast, 8)
        elif avg_daily_historical < 1.2:
            forecast = min(base_forecast, 15)
        else:
            forecast = min(base_forecast, 25)
        
        forecast = max(1, round(forecast))
        
        # Reality check for low-volume
        if forecast > max_daily * 2:
            forecast = max(1, round(max_daily * 1.5))
    
    return {
        'forecast': int(forecast),
        'product_type_category': product_type_category,
        'avg_daily_historical': avg_daily_historical,
        'avg_daily_recent': avg_daily_recent,
        'max_daily': max_daily,
        'sales_frequency': sales_frequency,
        'total_sales': total_sales,
        'recent_sales': recent_sales
    }

def calculate_hybrid_metrics_enhanced(df_daily):
    """Calculate metrics using enhanced hybrid approach"""
    results = []
    variants = df_daily['variant_name'].unique()
    
    print(f"Calculating enhanced hybrid metrics for {len(variants)} variants...")
    
    for i, variant in enumerate(variants):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(variants)} variants...")
        
        variant_data = df_daily[df_daily['variant_name'] == variant].copy()
        
        if len(variant_data) >= 7:
            variant_data = variant_data.sort_values('date')
            recent_record = variant_data.iloc[-1]
            product_type = recent_record.get('product_type', '')
            
            # Calculate forecast using enhanced hybrid logic
            forecast_result = calculate_hybrid_forecast_enhanced(
                variant_data, variant, product_type
            )
            
            # Risk classification
            avg_daily = forecast_result['avg_daily_historical']
            recent_avg = forecast_result['avg_daily_recent']
            frequency = forecast_result['sales_frequency']
            
            if recent_avg > avg_daily * 1.5:
                risk_category = "Growing Demand"
            elif recent_avg < avg_daily * 0.6:
                risk_category = "Declining Demand"
            elif frequency < 0.15:
                risk_category = "Low Velocity"
            elif forecast_result['product_type_category'] in ["Ultra High Consumable", "Very High Consumable"]:
                risk_category = "High Volume Critical"
            else:
                risk_category = "Stable"
            
            # Volume category based on forecast method used
            if forecast_result['product_type_category'] in ["Ultra High Consumable", "Very High Consumable"]:
                volume_category = forecast_result['product_type_category']
            elif forecast_result['product_type_category'] in ["High Consumable", "High Volume Product"]:
                volume_category = forecast_result['product_type_category']
            else:
                # Use traditional volume classification
                if avg_daily < 0.2:
                    volume_category = "Very Low Volume"
                elif avg_daily < 0.8:
                    volume_category = "Low Volume"
                elif avg_daily < 1.5:
                    volume_category = "Medium-Low Volume"
                else:
                    volume_category = "Medium Volume"
            
            results.append({
                'variant_name': variant,
                'product_id': recent_record.get('product_id', ''),
                'product_type': product_type,
                'product_attribute': recent_record.get('product_attribute', ''),
                'total_sales_history': int(forecast_result['total_sales']),
                'recent_14d_sales': int(forecast_result['recent_sales']),
                'avg_daily_sales': round(forecast_result['avg_daily_historical'], 3),
                'avg_daily_recent': round(forecast_result['avg_daily_recent'], 3),
                'max_daily_sale': int(forecast_result['max_daily']),
                'sales_frequency_pct': round(forecast_result['sales_frequency'] * 100, 1),
                'recommended_stock_2w': forecast_result['forecast'],
                'volume_category': volume_category,
                'risk_category': risk_category,
                'product_type_category': forecast_result['product_type_category'],
                'forecasting_method': get_forecasting_method(forecast_result['product_type_category']),
                'last_date': recent_record['date'].strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(results)

def get_forecasting_method(product_type_category):
    """Return the forecasting method used for each category"""
    if product_type_category in ["Ultra High Consumable", "Very High Consumable"]:
        return "Holt-Winters + Safety Stock"
    elif product_type_category in ["High Consumable", "High Volume Product"]:
        return "Moving Average + Trend"
    elif product_type_category == "Medium Consumable":
        return "Weighted Average + Trend"
    elif product_type_category in ["Medium Volume Product", "Low-Medium Volume"]:
        return "Conservative Weighted"
    else:
        return "Conservative Fixed"

def distribute_to_outlets_enhanced(variant_forecasts, raw_df):
    """Enhanced outlet distribution with method-aware caps"""
    results = []
    
    print("Calculating outlet shares with method-aware distribution...")
    
    # Calculate outlet shares
    outlet_shares = raw_df.groupby(['variant_name', 'outlet_id', 'outlet_name']).agg({
        'quantity': 'sum'
    }).reset_index()
    
    variant_totals = outlet_shares.groupby('variant_name')['quantity'].sum().reset_index()
    variant_totals.rename(columns={'quantity': 'total_quantity'}, inplace=True)
    
    outlet_shares = outlet_shares.merge(variant_totals, on='variant_name')
    outlet_shares['share'] = outlet_shares['quantity'] / outlet_shares['total_quantity']
    outlet_shares['share'] = outlet_shares['share'].fillna(1.0)
    
    method_based_adjustments = 0
    
    for _, forecast_row in variant_forecasts.iterrows():
        variant = forecast_row['variant_name']
        variant_outlets = outlet_shares[outlet_shares['variant_name'] == variant]
        
        if len(variant_outlets) > 0:
            for _, outlet_row in variant_outlets.iterrows():
                raw_allocation = forecast_row['recommended_stock_2w'] * outlet_row['share']
                
                forecasting_method = forecast_row.get('forecasting_method', 'Conservative Fixed')
                product_type_cat = forecast_row.get('product_type_category', 'Low Volume Retail')
                
                # Method-aware outlet caps
                if forecasting_method == "Holt-Winters + Safety Stock":
                    outlet_cap = 500
                    historical_multiplier = 4.0  # More generous
                elif forecasting_method == "Moving Average + Trend":
                    outlet_cap = 300
                    historical_multiplier = 3.0
                elif forecasting_method == "Weighted Average + Trend":
                    outlet_cap = 100
                    historical_multiplier = 2.5
                elif forecasting_method == "Conservative Weighted":
                    outlet_cap = 50
                    historical_multiplier = 2.0
                else:
                    outlet_cap = 25
                    historical_multiplier = 1.5
                
                historical_outlet_sales = outlet_row['quantity']
                max_reasonable = max(5 if "High" in product_type_cat else 2, 
                                   historical_outlet_sales * historical_multiplier)
                
                # Apply caps
                outlet_forecast = max(1, min(round(raw_allocation), outlet_cap))
                
                # Less restrictive historical checks for time-series forecasted items
                if outlet_forecast > max_reasonable and forecasting_method not in ["Holt-Winters + Safety Stock", "Moving Average + Trend"]:
                    method_based_adjustments += 1
                    outlet_forecast = max_reasonable
                
                results.append({
                    'variant_name': variant,
                    'product_id': forecast_row['product_id'],
                    'product_type': forecast_row['product_type'],
                    'product_attribute': forecast_row['product_attribute'],
                    'outlet_id': outlet_row['outlet_id'],
                    'outlet_name': outlet_row['outlet_name'],
                    'recommended_stock_2w': outlet_forecast,
                    'volume_category': forecast_row.get('volume_category', 'Unknown'),
                    'risk_category': forecast_row.get('risk_category', 'Unknown'),
                    'product_type_category': product_type_cat,
                    'forecasting_method': forecasting_method,
                    'outlet_share': round(outlet_row['share'] * 100, 1),
                    'historical_outlet_sales': int(historical_outlet_sales),
                    'created_at': forecast_row['last_date']
                })
        else:
            # Fallback
            results.append({
                'variant_name': variant,
                'product_id': forecast_row['product_id'],
                'product_type': forecast_row['product_type'],
                'product_attribute': forecast_row['product_attribute'],
                'outlet_id': 'all',
                'outlet_name': 'All Outlets',
                'recommended_stock_2w': forecast_row['recommended_stock_2w'],
                'volume_category': forecast_row.get('volume_category', 'Unknown'),
                'risk_category': forecast_row.get('risk_category', 'Unknown'),
                'product_type_category': forecast_row.get('product_type_category', 'Unknown'),
                'forecasting_method': forecast_row.get('forecasting_method', 'Unknown'),
                'outlet_share': 100.0,
                'historical_outlet_sales': 0,
                'created_at': forecast_row['last_date']
            })
    
    print(f"Applied method-based caps to {method_based_adjustments} outlet forecasts")
    
    return pd.DataFrame(results)

def process_hybrid_forecast_enhanced(raw_df):
    """Main processing with enhanced hybrid approach"""
    print("Processing ENHANCED HYBRID forecast with time-series for high-volume items...")
    
    # Data preparation
    raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
    raw_df['date'] = raw_df['created_at'].dt.date
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    
    df_daily = raw_df.groupby(['variant_name', 'date'], as_index=False).agg({
        'quantity': 'sum',
        'product_id': 'first',
        'product_type': 'first',
        'product_attribute': 'first'
    })
    
    print(f"Processing {len(df_daily['variant_name'].unique())} variants with enhanced hybrid logic")
    
    # Calculate forecasts
    forecast_df = calculate_hybrid_metrics_enhanced(df_daily)
    
    # Show method breakdown
    method_breakdown = forecast_df['forecasting_method'].value_counts()
    print(f"Forecasting method breakdown:")
    for method, count in method_breakdown.items():
        print(f"   • {method}: {count} variants")
    
    # Show product type breakdown
    type_breakdown = forecast_df['product_type_category'].value_counts()
    print(f"Product type breakdown:")
    for ptype, count in type_breakdown.items():
        print(f"   • {ptype}: {count} variants")
    
    # Distribute to outlets
    final_results = distribute_to_outlets_enhanced(forecast_df, raw_df)
    final_results = final_results.sort_values('recommended_stock_2w', ascending=False)
    
    # Summary statistics
    total_variants = len(final_results['variant_name'].unique())
    total_outlets = len(final_results['outlet_name'].unique())
    final_total_forecast = final_results['recommended_stock_2w'].sum()
    
    # Calculate historical baseline for comparison
    historical_total = raw_df['quantity'].sum() / len(raw_df['date'].unique()) * 14
    
    print(f"Enhanced hybrid forecast summary:")
    print(f"   • Total forecast: {final_total_forecast:,}")
    print(f"   • Historical 14-day equivalent: {historical_total:,.0f}")
    print(f"   • Forecast vs historical ratio: {final_total_forecast / historical_total:.2f}x")
    
    # Save results
    with open(FORECAST_FILE, 'w') as f:
        json.dump({
            'forecast_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_variants': total_variants,
            'total_outlets': total_outlets,
            'total_forecasts': len(final_results),
            'summary_stats': {
                'total_recommended_stock': int(final_results['recommended_stock_2w'].sum()),
                'historical_14day_equivalent': int(historical_total),
                'forecast_vs_historical_ratio': float(final_total_forecast / historical_total),
                'avg_recommended_stock': float(final_results['recommended_stock_2w'].mean()),
                'median_recommended_stock': float(final_results['recommended_stock_2w'].median()),
                'max_recommended_stock': int(final_results['recommended_stock_2w'].max()),
                'p95_recommended_stock': float(final_results['recommended_stock_2w'].quantile(0.95))
            },
            'forecasting_method_breakdown': final_results['forecasting_method'].value_counts().to_dict(),
            'product_type_breakdown': final_results['product_type_category'].value_counts().to_dict(),
            'risk_category_breakdown': final_results['risk_category'].value_counts().to_dict(),
            'model_info': {
                'model_type': 'Enhanced Hybrid Forecasting with Time Series',
                'methodology': 'Different forecasting algorithms per product category with safety stock',
                'methods_used': {
                    'high_volume': 'Holt-Winters Exponential Smoothing + 20-30% safety stock',
                    'medium_volume': 'Moving Average with Trend + 15% safety stock',
                    'low_volume': 'Conservative weighted averaging'
                },
                'key_improvements': [
                    'Time-series forecasting for high-volume consumables',
                    'Adaptive safety stock based on product category',
                    'Method-aware outlet distribution caps',
                    'Preserved conservative logic for low-volume items',
                    'Enhanced product classification thresholds'
                ]
            }
        }, f, indent=2)
    
    # Show top time-series forecasts
    time_series_forecasts = final_results[
        final_results['forecasting_method'].str.contains('Holt-Winters|Moving Average')
    ].nlargest(20, 'recommended_stock_2w')
    
    if len(time_series_forecasts) > 0:
        print(f"\nTop 20 time-series forecasts (should handle high-volume better):")
        for _, row in time_series_forecasts.iterrows():
            print(f"   • {row['variant_name'][:40]:<40} | Forecast: {row['recommended_stock_2w']:>4} | Method: {row['forecasting_method']}")
    
    return final_results

# ======================================================================
#                         FLASK ROUTES
# ======================================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    upload_mode = request.args.get('upload', None)

    if os.path.exists('hybrid_forecast.csv') and upload_mode != '1':
        return redirect(url_for('show_results', page=1))

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        try:
            new_df = pd.read_csv(path)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            return f"Error reading file: {e}", 500
        finally:
            if os.path.exists(path):
                os.remove(path)

        if os.path.exists(DATA_FILE):
            old_df = pd.read_csv(DATA_FILE)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df = combined_df.drop_duplicates()
        combined_df.to_csv(DATA_FILE, index=False)

        try:
            forecast_df = process_hybrid_forecast_enhanced(combined_df)
            
            if forecast_df is not None and len(forecast_df) > 0:
                forecast_df.to_csv('hybrid_forecast.csv', index=False)
                print(f"Enhanced hybrid forecast completed! Generated {len(forecast_df)} forecasts")
            else:
                return "Error: No forecasts were generated.", 500
            
        except Exception as e:
            print(f"Error processing enhanced hybrid forecast: {e}")
            return f"Error processing enhanced hybrid forecast: {e}", 500

        return redirect(url_for('show_results', page=1))

    if upload_mode == '1' or not os.path.exists('hybrid_forecast.csv'):
        return render_template('upload.html')

    return redirect(url_for('show_results', page=1))

@app.route('/results', methods=['GET'])
def show_results():
    user = session.get("user")
    if not user:
        return redirect(url_for("auth.login"))

    page = int(request.args.get('page', 1))
    selected_outlet = request.args.get('outlet', 'all')
    selected_risk = request.args.get('risk', 'all')
    selected_product_type = request.args.get('product_type_category', 'all')
    selected_method = request.args.get('forecasting_method', 'all')
    filter_variant_name = request.args.get('variant_name', '').strip()

    if not os.path.exists('hybrid_forecast.csv'):
        return redirect(url_for('home', upload=1))

    forecast_df = pd.read_csv('hybrid_forecast.csv')

    if user["role"] == "store_Manager" and "outlet_name" in forecast_df.columns:
        forecast_df = forecast_df[
            forecast_df["outlet_name"].astype(str).str.strip().str.lower() ==
            str(user["outlet_name"]).strip().lower()
        ]
        selected_outlet = user["outlet_name"]

    # Apply filters
    if filter_variant_name and 'variant_name' in forecast_df.columns:
        forecast_df = forecast_df[
            forecast_df['variant_name'].astype(str).str.contains(filter_variant_name, case=False, na=False)
        ]

    if selected_outlet != 'all' and 'outlet_name' in forecast_df.columns:
        forecast_df = forecast_df[
            forecast_df['outlet_name'].astype(str).str.contains(selected_outlet, case=False, na=False)
        ]

    if selected_risk != 'all' and 'risk_category' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['risk_category'] == selected_risk]

    if selected_product_type != 'all' and 'product_type_category' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['product_type_category'] == selected_product_type]
    
    if selected_method != 'all' and 'forecasting_method' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['forecasting_method'] == selected_method]

    # Pagination
    ROWS_PER_PAGE = 50
    total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)
    start = (page - 1) * ROWS_PER_PAGE
    end = start + ROWS_PER_PAGE

    display_cols = [
        'variant_name', 'product_id', 'product_type', 'outlet_name', 
        'recommended_stock_2w', 'product_type_category', 'forecasting_method',
        'risk_category', 'avg_daily_sales', 'created_at'
    ]
    display_cols = [c for c in display_cols if c in forecast_df.columns]

    forecast_page = forecast_df.iloc[start:end][display_cols]
    table_html = forecast_page.to_html(classes='table table-striped', index=False, float_format='{:.3f}'.format)

    outlets = forecast_df['outlet_name'].unique().tolist() if 'outlet_name' in forecast_df.columns else []
    risk_categories = forecast_df['risk_category'].unique().tolist() if 'risk_category' in forecast_df.columns else []
    product_type_categories = forecast_df['product_type_category'].unique().tolist() if 'product_type_category' in forecast_df.columns else []
    forecasting_methods = forecast_df['forecasting_method'].unique().tolist() if 'forecasting_method' in forecast_df.columns else []

    return render_template(
        'result.html',
        forecast_table=table_html,
        outlets=outlets,
        risk_categories=risk_categories,
        product_type_categories=product_type_categories,
        forecasting_methods=forecasting_methods,
        page=page,
        total_pages=total_pages,
        selected_outlet=selected_outlet,
        selected_risk=selected_risk,
        selected_product_type=selected_product_type,
        selected_method=selected_method,
        filter_variant_name=filter_variant_name,
        total_items=len(forecast_df)
    )

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    if os.path.exists(FORECAST_FILE):
        with open(FORECAST_FILE, 'r') as f:
            summary = json.load(f)
        return jsonify(summary)
    else:
        return jsonify({"error": "No enhanced hybrid forecast data available."}), 404

@app.route('/api/method_performance', methods=['GET'])
def api_method_performance():
    """API endpoint for analyzing performance by forecasting method"""
    if not os.path.exists('hybrid_forecast.csv'):
        return jsonify({"error": "No hybrid forecast available."}), 404
    
    forecast_df = pd.read_csv('hybrid_forecast.csv')
    
    method_analysis = {}
    for method in forecast_df['forecasting_method'].unique():
        method_data = forecast_df[forecast_df['forecasting_method'] == method]
        method_analysis[method] = {
            'count': len(method_data),
            'avg_forecast': float(method_data['recommended_stock_2w'].mean()),
            'total_forecast': int(method_data['recommended_stock_2w'].sum()),
            'max_forecast': int(method_data['recommended_stock_2w'].max()),
            'median_forecast': float(method_data['recommended_stock_2w'].median())
        }
    
    return jsonify({
        'method_performance': method_analysis,
        'total_products': len(forecast_df),
        'high_volume_methods': [
            'Holt-Winters + Safety Stock',
            'Moving Average + Trend'
        ],
        'conservative_methods': [
            'Conservative Fixed',
            'Conservative Weighted'
        ]
    })

@app.route('/api/high_volume_analysis', methods=['GET'])
def api_high_volume_analysis():
    """API endpoint for analyzing high-volume product forecasts"""
    if not os.path.exists('hybrid_forecast.csv'):
        return jsonify({"error": "No hybrid forecast available."}), 404
    
    forecast_df = pd.read_csv('hybrid_forecast.csv')
    
    # High-volume analysis
    high_volume = forecast_df[
        forecast_df['product_type_category'].isin(['Ultra High Consumable', 'Very High Consumable'])
    ]
    
    time_series_products = forecast_df[
        forecast_df['forecasting_method'].str.contains('Holt-Winters|Moving Average', na=False)
    ]
    
    analysis = {
        'high_volume_count': len(high_volume),
        'time_series_count': len(time_series_products),
        'high_volume_total_forecast': int(high_volume['recommended_stock_2w'].sum()) if len(high_volume) > 0 else 0,
        'time_series_total_forecast': int(time_series_products['recommended_stock_2w'].sum()) if len(time_series_products) > 0 else 0,
        'top_forecasts': time_series_products.nlargest(10, 'recommended_stock_2w')[
            ['variant_name', 'recommended_stock_2w', 'forecasting_method', 'product_type_category', 'avg_daily_sales']
        ].to_dict('records') if len(time_series_products) > 0 else [],
        'method_distribution': time_series_products['forecasting_method'].value_counts().to_dict() if len(time_series_products) > 0 else {}
    }
    
    return jsonify(analysis)

@app.route('/api/validation', methods=['GET'])
def api_validation():
    """Enhanced validation endpoint"""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
            df['date'] = pd.to_datetime(df['date'])
            
            # Aggregate for validation
            df_daily = df.groupby(['variant_name', 'date'], as_index=False).agg({
                'quantity': 'sum'
            })
            
            validation_results = validate_enhanced_model(df_daily)
            return jsonify(validation_results)
        except Exception as e:
            return jsonify({"error": f"Validation failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "No data available for validation"}), 404

def validate_enhanced_model(df_daily, test_days=14):
    """Enhanced validation for the hybrid model"""
    accuracy_scores = []
    method_scores = {'time_series': [], 'traditional': []}
    variants = df_daily['variant_name'].unique()
    
    print(f"Validating enhanced hybrid model on {len(variants)} variants...")
    
    for variant in variants:
        variant_data = df_daily[df_daily['variant_name'] == variant].sort_values('date')
        if len(variant_data) >= 28:
            # Split data
            split_point = int(len(variant_data) * 0.7)
            train_data = variant_data.iloc[:split_point]
            test_data = variant_data.iloc[split_point:split_point+test_days]
            
            if len(test_data) == test_days and len(train_data) >= 14:
                # Determine which method would be used
                avg_daily = train_data['quantity'].sum() / len(train_data)
                frequency = (train_data['quantity'] > 0).sum() / len(train_data)
                max_daily = train_data['quantity'].max()
                
                # Simulate product classification
                is_high_volume = avg_daily > 2.5 and frequency > 0.4
                
                if is_high_volume:
                    # Use time-series approach
                    try:
                        prediction = holt_winters_forecast(train_data['quantity'].values, test_days)
                        method_type = 'time_series'
                    except:
                        # Fallback to traditional
                        prediction = (train_data['quantity'].sum() / len(train_data)) * test_days
                        method_type = 'traditional'
                else:
                    # Traditional approach
                    recent_avg = train_data.tail(7)['quantity'].sum() / 7
                    historical_avg = train_data['quantity'].sum() / len(train_data)
                    prediction = max(historical_avg, recent_avg) * test_days
                    method_type = 'traditional'
                
                actual = test_data['quantity'].sum()
                
                # Calculate accuracy
                if actual > 0:
                    error = abs(prediction - actual) / actual
                    # Penalize under-prediction more heavily for high-volume items
                    if is_high_volume and prediction < actual:
                        error *= 1.3
                    accuracy_scores.append(min(error, 2.0))
                    method_scores[method_type].append(min(error, 2.0))
                elif prediction <= 2:
                    accuracy_scores.append(0.1)
                    method_scores[method_type].append(0.1)
                else:
                    penalty = min(prediction / 10, 1.5)
                    accuracy_scores.append(penalty)
                    method_scores[method_type].append(penalty)
    
    if accuracy_scores:
        avg_error = np.mean(accuracy_scores)
        accuracy_pct = max(0, (1 - avg_error) * 100)
        
        # Method-specific accuracy
        ts_accuracy = max(0, (1 - np.mean(method_scores['time_series'])) * 100) if method_scores['time_series'] else 0
        trad_accuracy = max(0, (1 - np.mean(method_scores['traditional'])) * 100) if method_scores['traditional'] else 0
        
        return {
            'overall_validation_accuracy': round(accuracy_pct, 2),
            'avg_percentage_error': round(avg_error * 100, 2),
            'samples_tested': len(accuracy_scores),
            'time_series_accuracy': round(ts_accuracy, 2),
            'traditional_accuracy': round(trad_accuracy, 2),
            'time_series_samples': len(method_scores['time_series']),
            'traditional_samples': len(method_scores['traditional']),
            'model_version': 'Enhanced Hybrid with Time Series'
        }
    else:
        return {
            'overall_validation_accuracy': 0,
            'avg_percentage_error': 100,
            'samples_tested': 0,
            'model_version': 'Enhanced Hybrid with Time Series'
        }

@app.route('/api/results', methods=['GET'])
def api_results():
    page = request.args.get('page', '1').strip()
    filter_outlet_id = request.args.get('outlet_id', '').strip()
    filter_outlet_name = request.args.get('outlet_name', '').strip()
    filter_product_id = request.args.get('product_id', '').strip()
    filter_variant_name = request.args.get('variant_name', '').strip()
    filter_product_attr = request.args.get('product_attribute', '').strip()
    filter_product_type = request.args.get('product_type', '').strip()
    filter_volume_cat = request.args.get('volume_category', '').strip()
    filter_risk_cat = request.args.get('risk_category', '').strip()
    filter_product_type_cat = request.args.get('product_type_category', '').strip()
    filter_method = request.args.get('forecasting_method', '').strip()
    match_type = request.args.get('match', 'contains').lower()

    if not os.path.exists('hybrid_forecast.csv'):
        return jsonify({"error": "No enhanced hybrid forecast available. Please upload data first."}), 404

    forecast_df = pd.read_csv('hybrid_forecast.csv')

    def apply_filter(df, col, value):
        if col not in df.columns or not value:
            return df
        if match_type == "exact":
            return df[df[col].astype(str).str.strip().str.lower() == value.strip().lower()]
        else:
            return df[df[col].astype(str).str.contains(value, case=False, na=False, regex=False)]

    # Apply filters
    if filter_outlet_id and 'outlet_id' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['outlet_id'].astype(str) == filter_outlet_id]
    forecast_df = apply_filter(forecast_df, 'outlet_name', filter_outlet_name)
    forecast_df = apply_filter(forecast_df, 'product_id', filter_product_id)
    forecast_df = apply_filter(forecast_df, 'variant_name', filter_variant_name)
    forecast_df = apply_filter(forecast_df, 'product_attribute', filter_product_attr)
    forecast_df = apply_filter(forecast_df, 'product_type', filter_product_type)
    forecast_df = apply_filter(forecast_df, 'volume_category', filter_volume_cat)
    forecast_df = apply_filter(forecast_df, 'risk_category', filter_risk_cat)
    forecast_df = apply_filter(forecast_df, 'product_type_category', filter_product_type_cat)
    forecast_df = apply_filter(forecast_df, 'forecasting_method', filter_method)

    # Sort by recommended stock
    forecast_df = forecast_df.sort_values('recommended_stock_2w', ascending=False)

    ROWS_PER_PAGE = 50
    total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)

    if page.lower() == 'all':
        display_cols = [c for c in [
            'product_id', 'variant_name', 'product_attribute', 'product_type',
            'outlet_id', 'outlet_name', 'recommended_stock_2w', 'volume_category',
            'risk_category', 'product_type_category', 'forecasting_method',
            'avg_daily_sales', 'created_at'
        ] if c in forecast_df.columns]
        forecast_page = forecast_df[display_cols]
        page_out = 'all'
    else:
        try:
            page_int = int(page)
            page_out = page_int
        except ValueError:
            page_int = 1
            page_out = 1

        start = (page_int - 1) * ROWS_PER_PAGE
        end = start + ROWS_PER_PAGE

        display_cols = [c for c in [
            'product_id', 'variant_name', 'product_attribute', 'product_type',
            'outlet_id', 'outlet_name', 'recommended_stock_2w', 'volume_category',
            'risk_category', 'product_type_category', 'forecasting_method',
            'avg_daily_sales', 'created_at'
        ] if c in forecast_df.columns]
        forecast_page = forecast_df.iloc[start:end][display_cols]

    return jsonify({
        "page": page_out,
        "total_pages": total_pages,
        "count": len(forecast_page),
        "results": forecast_page.to_dict(orient="records"),
        "model_version": "Enhanced Hybrid with Time Series",
        "summary": {
            "total_forecast": int(forecast_df['recommended_stock_2w'].sum()),
            "avg_forecast": float(forecast_df['recommended_stock_2w'].mean()),
            "method_breakdown": forecast_df['forecasting_method'].value_counts().to_dict() if 'forecasting_method' in forecast_df.columns else {}
        }
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)