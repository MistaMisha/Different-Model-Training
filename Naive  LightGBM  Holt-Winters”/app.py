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
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
#                    ENHANCED HYBRID FORECASTING SYSTEM
# ======================================================================

def classify_product_sales_pattern(variant_data, variant_name, product_type):
    """
    Classify products into categories to determine which forecasting method to use
    Returns: 'naive', 'lightgbm', or 'holt_winters'
    """
    # Basic statistics
    total_sales = variant_data['quantity'].sum()
    total_days = len(variant_data)
    non_zero_days = (variant_data['quantity'] > 0).sum()
    max_daily = variant_data['quantity'].max()
    avg_daily = total_sales / total_days
    sales_frequency = non_zero_days / total_days
    
    # Check for seasonality/trend
    if len(variant_data) >= 14:
        # Simple trend detection
        recent_7d = variant_data.tail(7)['quantity'].mean()
        earlier_7d = variant_data.tail(14).head(7)['quantity'].mean() if len(variant_data) >= 14 else recent_7d
        trend_strength = abs(recent_7d - earlier_7d) / (earlier_7d + 1)  # +1 to avoid division by zero
    else:
        trend_strength = 0
    
    # Coefficient of variation (measure of variability)
    cv = variant_data['quantity'].std() / (variant_data['quantity'].mean() + 1)
    
    variant_lower = str(variant_name).lower()
    consumable_keywords = [
        'ultra bar', 'vaporesso', 'pod', 'coil', 'disposable', 'cartridge',
        'replacement', 'refill', 'atomizer', 'tank', 'clearomizer', 'terea'
    ]
    is_consumable = any(keyword in variant_lower for keyword in consumable_keywords)
    
    # Decision logic
    # 1. High-volume, high-frequency, seasonal/trendy products -> Holt-Winters
    if (avg_daily >= 3.0 and sales_frequency > 0.5 and 
        (trend_strength > 0.3 or cv > 0.8) and total_days >= 14):
        return 'holt_winters'
    
    # 2. Very high volume consumables -> Holt-Winters
    elif (is_consumable and avg_daily >= 5.0 and sales_frequency > 0.4):
        return 'holt_winters'
    
    # 3. Low volume, sparse sales -> Naive
    elif (avg_daily < 0.5 or sales_frequency < 0.2 or total_sales < 10):
        return 'naive'
    
    # 4. Everything else (medium volume, irregular patterns) -> LightGBM
    else:
        return 'lightgbm'

def naive_forecast(variant_data, periods=14):
    """
    Naive forecasting for low/sparse sales products
    Uses simple average with conservative adjustments
    """
    total_sales = variant_data['quantity'].sum()
    total_days = len(variant_data)
    avg_daily = total_sales / total_days
    
    # Get recent performance
    recent_data = variant_data.tail(7) if len(variant_data) >= 7 else variant_data
    recent_avg = recent_data['quantity'].mean()
    
    # Conservative approach - use higher of recent or historical average
    daily_estimate = max(avg_daily, recent_avg)
    
    # Apply conservative multiplier
    if avg_daily < 0.1:
        multiplier = 0.8  # Very conservative for very low sales
    elif avg_daily < 0.5:
        multiplier = 1.0
    else:
        multiplier = 1.1
    
    forecast = daily_estimate * periods * multiplier
    
    # Conservative caps
    max_historical_daily = variant_data['quantity'].max()
    reasonable_cap = max_historical_daily * periods * 0.8
    
    return max(1, min(round(forecast), reasonable_cap, 25))

def prepare_features_for_lightgbm(variant_data):
    """
    Prepare features for LightGBM model
    """
    df = variant_data.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time-based features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['day_of_month'] = pd.to_datetime(df['date']).dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features
    for lag in [1, 2, 3, 7]:
        if len(df) > lag:
            df[f'lag_{lag}'] = df['quantity'].shift(lag)
    
    # Rolling statistics
    for window in [3, 7]:
        if len(df) >= window:
            df[f'rolling_mean_{window}'] = df['quantity'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['quantity'].rolling(window=window, min_periods=1).std()
    
    # Trend features
    if len(df) >= 7:
        df['trend_7d'] = df['quantity'].rolling(window=7, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
        )
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def lightgbm_forecast(variant_data, periods=14):
    """
    LightGBM forecasting for medium volume, irregular sales
    """
    try:
        # Prepare features
        df_features = prepare_features_for_lightgbm(variant_data)
        
        if len(df_features) < 7:
            # Fallback to simple average if insufficient data
            return variant_data['quantity'].mean() * periods
        
        # Feature columns (exclude target and date)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'quantity'] and df_features[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) == 0:
            # Fallback if no features
            return variant_data['quantity'].mean() * periods
        
        # Prepare training data
        X = df_features[feature_cols].values
        y = df_features['quantity'].values
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=min(3, len(df_features)//3))
        
        best_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 10,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        models = []
        
        # Train multiple models with cross-validation
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 3 or len(val_idx) < 1:
                continue
                
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                best_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=50,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            models.append(model)
        
        if not models:
            # If no models were trained, fallback
            return variant_data['quantity'].mean() * periods
        
        # Generate forecast by extending the time series
        last_row = df_features.iloc[-1:][feature_cols]
        daily_forecasts = []
        
        # Forecast each day
        for day in range(periods):
            # Average predictions from all models
            day_predictions = []
            for model in models:
                pred = model.predict(last_row.values, num_iteration=model.best_iteration)
                day_predictions.append(max(0, pred[0]))  # Ensure non-negative
            
            daily_forecast = np.mean(day_predictions)
            daily_forecasts.append(daily_forecast)
            
            # Update features for next day (simplified)
            # This is a basic approach - in practice, you'd want more sophisticated feature updating
            if len(feature_cols) > 0:
                last_row = last_row.copy()
                # Update some basic features (this is simplified)
                if 'lag_1' in last_row.columns:
                    last_row['lag_1'].iloc[0] = daily_forecast
        
        total_forecast = sum(daily_forecasts)
        
        # Apply reasonable constraints
        historical_max_daily = variant_data['quantity'].max()
        historical_avg_daily = variant_data['quantity'].mean()
        
        # Cap the forecast
        reasonable_max = max(historical_max_daily * periods, historical_avg_daily * periods * 2)
        total_forecast = min(total_forecast, reasonable_max)
        
        return max(1, round(total_forecast))
        
    except Exception as e:
        print(f"LightGBM failed for variant, using fallback: {str(e)}")
        # Fallback to moving average
        recent_avg = variant_data.tail(7)['quantity'].mean() if len(variant_data) >= 7 else variant_data['quantity'].mean()
        return max(1, round(recent_avg * periods))

def holt_winters_forecast(variant_data, periods=14, alpha=None, beta=None):
    """
    Holt-Winters exponential smoothing for high-volume, seasonal products
    """
    try:
        data = variant_data['quantity'].values
        
        if len(data) < 10:
            raise ValueError("Insufficient data for Holt-Winters")
        
        # Auto-optimize parameters if not provided
        if alpha is None or beta is None:
            def optimize_params(params):
                a, b = params
                return holt_winters_mse(data, a, b, periods)
            
            result = minimize(optimize_params, [0.3, 0.1], 
                            bounds=[(0.01, 0.9), (0.01, 0.5)], 
                            method='L-BFGS-B')
            alpha, beta = result.x
        
        # Initialize
        n = len(data)
        level = np.zeros(n)
        trend = np.zeros(n)
        
        # Initial values
        level[0] = data[0]
        if n > 1:
            trend[0] = data[1] - data[0]
        else:
            trend[0] = 0
        
        # Apply Holt-Winters
        for i in range(1, n):
            level_prev = level[i-1]
            trend_prev = trend[i-1]
            
            level[i] = alpha * data[i] + (1 - alpha) * (level_prev + trend_prev)
            trend[i] = beta * (level[i] - level_prev) + (1 - beta) * trend_prev
        
        # Forecast
        forecast = (level[-1] + trend[-1] * periods)
        
        # Add safety margin for high-volume items
        safety_margin = 1.15  # 15% safety stock
        final_forecast = max(1, forecast * safety_margin)
        
        # Reasonable cap based on historical data
        max_daily = np.max(data)
        avg_daily = np.mean(data)
        reasonable_cap = max(max_daily * periods, avg_daily * periods * 2.5)
        
        return min(round(final_forecast), reasonable_cap)
        
    except Exception as e:
        print(f"Holt-Winters failed: {str(e)}")
        raise  # Re-raise to trigger LightGBM fallback

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
        level = np.zeros(train_size)
        trend = np.zeros(train_size)
        
        level[0] = train_data[0]
        trend[0] = train_data[1] - train_data[0] if train_size > 1 else 0
        
        for i in range(1, train_size):
            level_prev = level[i-1]
            trend_prev = trend[i-1]
            
            level[i] = alpha * train_data[i] + (1 - alpha) * (level_prev + trend_prev)
            trend[i] = beta * (level[i] - level_prev) + (1 - beta) * trend_prev
        
        pred = (level[-1] + trend[-1] * len(test_data))
        actual = np.sum(test_data)
        
        return (pred - actual) ** 2
    except:
        return 1000

def calculate_hybrid_forecast_enhanced(variant_data, variant_name, product_type):
    """
    Enhanced hybrid forecast using Naive/LightGBM/Holt-Winters approach
    """
    # Basic statistics
    total_sales = variant_data['quantity'].sum()
    total_days = len(variant_data)
    non_zero_days = (variant_data['quantity'] > 0).sum()
    max_daily = variant_data['quantity'].max()
    avg_daily_historical = total_sales / total_days
    sales_frequency = non_zero_days / total_days
    
    # Get recent data
    recent_14d = variant_data.tail(14)['quantity'].values
    recent_sales = np.sum(recent_14d)
    avg_daily_recent = recent_sales / len(recent_14d)
    
    # Classify the product to determine forecasting method
    forecasting_method = classify_product_sales_pattern(variant_data, variant_name, product_type)
    
    # Apply the appropriate forecasting method
    forecast_error = None
    
    if forecasting_method == 'naive':
        forecast = naive_forecast(variant_data, periods=14)
        method_used = "Naive (Conservative)"
        
    elif forecasting_method == 'holt_winters':
        try:
            forecast = holt_winters_forecast(variant_data, periods=14)
            method_used = "Holt-Winters (Seasonal)"
        except:
            # Fallback to LightGBM if Holt-Winters fails
            print(f"Holt-Winters failed for {variant_name}, falling back to LightGBM")
            try:
                forecast = lightgbm_forecast(variant_data, periods=14)
                method_used = "LightGBM (Fallback from HW)"
            except:
                # Final fallback to naive
                forecast = naive_forecast(variant_data, periods=14)
                method_used = "Naive (Double Fallback)"
            
    else:  # lightgbm
        try:
            forecast = lightgbm_forecast(variant_data, periods=14)
            method_used = "LightGBM (ML)"
        except:
            # Fallback to naive if LightGBM fails
            forecast = naive_forecast(variant_data, periods=14)
            method_used = "Naive (Fallback from LGB)"
    
    # Risk classification
    if avg_daily_recent > avg_daily_historical * 1.5:
        risk_category = "Growing Demand"
    elif avg_daily_recent < avg_daily_historical * 0.6:
        risk_category = "Declining Demand"
    elif sales_frequency < 0.15:
        risk_category = "Low Velocity"
    elif forecasting_method == 'holt_winters':
        risk_category = "High Volume Critical"
    else:
        risk_category = "Stable"
    
    # Volume category
    if avg_daily_historical >= 5.0:
        volume_category = "Very High Volume"
    elif avg_daily_historical >= 2.5:
        volume_category = "High Volume"
    elif avg_daily_historical >= 1.0:
        volume_category = "Medium Volume"
    elif avg_daily_historical >= 0.3:
        volume_category = "Low-Medium Volume"
    else:
        volume_category = "Low Volume"
    
    return {
        'forecast': int(forecast),
        'method_used': method_used,
        'forecasting_method_class': forecasting_method,
        'avg_daily_historical': avg_daily_historical,
        'avg_daily_recent': avg_daily_recent,
        'max_daily': max_daily,
        'sales_frequency': sales_frequency,
        'total_sales': total_sales,
        'recent_sales': recent_sales,
        'volume_category': volume_category,
        'risk_category': risk_category
    }

def calculate_hybrid_metrics_enhanced(df_daily):
    """Calculate metrics using enhanced hybrid approach"""
    results = []
    variants = df_daily['variant_name'].unique()
    
    method_counts = {'naive': 0, 'lightgbm': 0, 'holt_winters': 0, 'fallbacks': 0}
    
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
            
            # Count methods used
            method_class = forecast_result['forecasting_method_class']
            if 'Fallback' in forecast_result['method_used']:
                method_counts['fallbacks'] += 1
            else:
                method_counts[method_class] += 1
            
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
                'volume_category': forecast_result['volume_category'],
                'risk_category': forecast_result['risk_category'],
                'forecasting_method': forecast_result['method_used'],
                'forecasting_method_class': forecast_result['forecasting_method_class'],
                'last_date': recent_record['date'].strftime('%Y-%m-%d')
            })
    
    print(f"Method usage breakdown:")
    print(f"   • Naive: {method_counts['naive']} variants")
    print(f"   • LightGBM: {method_counts['lightgbm']} variants")
    print(f"   • Holt-Winters: {method_counts['holt_winters']} variants")
    print(f"   • Fallbacks: {method_counts['fallbacks']} variants")
    
    return pd.DataFrame(results)

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
                
                forecasting_method_class = forecast_row.get('forecasting_method_class', 'naive')
                
                # Method-aware outlet caps
                if forecasting_method_class == 'holt_winters':
                    outlet_cap = 400
                    historical_multiplier = 3.0
                elif forecasting_method_class == 'lightgbm':
                    outlet_cap = 200
                    historical_multiplier = 2.5
                else:  # naive
                    outlet_cap = 50
                    historical_multiplier = 1.5
                
                historical_outlet_sales = outlet_row['quantity']
                max_reasonable = max(2, historical_outlet_sales * historical_multiplier)
                
                # Apply caps
                outlet_forecast = max(1, min(round(raw_allocation), outlet_cap))
                
                # Apply historical checks (more lenient for ML methods)
                if outlet_forecast > max_reasonable and forecasting_method_class == 'naive':
                    method_based_adjustments += 1
                    outlet_forecast = int(max_reasonable)
                
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
                    'forecasting_method': forecast_row.get('forecasting_method', 'Unknown'),
                    'forecasting_method_class': forecast_row.get('forecasting_method_class', 'Unknown'),
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
                'forecasting_method': forecast_row.get('forecasting_method', 'Unknown'),
                'forecasting_method_class': forecast_row.get('forecasting_method_class', 'Unknown'),
                'outlet_share': 100.0,
                'historical_outlet_sales': 0,
                'created_at': forecast_row['last_date']
            })
    
    print(f"Applied method-based caps to {method_based_adjustments} outlet forecasts")
    
    return pd.DataFrame(results)

def process_hybrid_forecast_enhanced(raw_df):
    """Main processing with enhanced hybrid approach"""
    print("Processing ENHANCED HYBRID forecast with Naive/LightGBM/Holt-Winters...")
    
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
    print(f"Final forecasting method breakdown:")
    for method, count in method_breakdown.items():
        print(f"   • {method}: {count} variants")
    
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
            'forecasting_method_class_breakdown': final_results['forecasting_method_class'].value_counts().to_dict(),
            'risk_category_breakdown': final_results['risk_category'].value_counts().to_dict(),
            'model_info': {
                'model_type': 'Enhanced Hybrid: Naive/LightGBM/Holt-Winters',
                'methodology': 'Different forecasting algorithms based on sales patterns and volume',
                'methods_used': {
                    'naive': 'Simple conservative averaging for low/sparse sales',
                    'lightgbm': 'Machine learning for medium volume irregular patterns',
                    'holt_winters': 'Exponential smoothing for high-volume seasonal products',
                    'fallback_strategy': 'LightGBM fallback for failed Holt-Winters, Naive for failed LightGBM'
                },
                'classification_criteria': {
                    'naive': 'avg_daily < 0.5 OR sales_frequency < 0.2 OR total_sales < 10',
                    'holt_winters': 'High volume (avg_daily >= 3.0) AND high frequency (>0.5) AND (seasonal/trend OR consumable with avg_daily >= 5.0)',
                    'lightgbm': 'Medium volume irregular patterns (everything else)'
                },
                'key_improvements': [
                    'Intelligent method selection based on sales patterns',
                    'LightGBM for complex irregular patterns',
                    'Holt-Winters for seasonal high-volume items',
                    'Conservative Naive approach for low-volume items',
                    'Robust fallback strategy prevents failures'
                ]
            }
        }, f, indent=2)
    
    # Show top forecasts by method
    for method_class in ['holt_winters', 'lightgbm', 'naive']:
        method_forecasts = final_results[
            final_results['forecasting_method_class'] == method_class
        ].nlargest(10, 'recommended_stock_2w')
        
        if len(method_forecasts) > 0:
            print(f"\nTop 10 {method_class.upper()} forecasts:")
            for _, row in method_forecasts.head(10).iterrows():
                print(f"   • {row['variant_name'][:40]:<40} | Forecast: {row['recommended_stock_2w']:>4} | {row['forecasting_method']}")
    
    return final_results

# ======================================================================
#                         VALIDATION FUNCTIONS
# ======================================================================

def validate_enhanced_model(df_daily, test_days=14):
    """Enhanced validation for the hybrid model"""
    accuracy_scores = []
    method_scores = {'naive': [], 'lightgbm': [], 'holt_winters': []}
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
                method_class = classify_product_sales_pattern(
                    train_data, variant, train_data.iloc[-1].get('product_type', '')
                )
                
                # Make prediction using the appropriate method
                try:
                    if method_class == 'naive':
                        prediction = naive_forecast(train_data, test_days)
                    elif method_class == 'holt_winters':
                        try:
                            prediction = holt_winters_forecast(train_data, test_days)
                        except:
                            # Fallback to LightGBM
                            try:
                                prediction = lightgbm_forecast(train_data, test_days)
                                method_class = 'lightgbm'  # Update method class for scoring
                            except:
                                prediction = naive_forecast(train_data, test_days)
                                method_class = 'naive'
                    else:  # lightgbm
                        try:
                            prediction = lightgbm_forecast(train_data, test_days)
                        except:
                            prediction = naive_forecast(train_data, test_days)
                            method_class = 'naive'
                    
                    actual = test_data['quantity'].sum()
                    
                    # Calculate accuracy
                    if actual > 0:
                        error = abs(prediction - actual) / actual
                        # Penalize under-prediction more heavily for high-volume items
                        if method_class == 'holt_winters' and prediction < actual:
                            error *= 1.2
                        accuracy_scores.append(min(error, 2.0))
                        method_scores[method_class].append(min(error, 2.0))
                    elif prediction <= 2:
                        accuracy_scores.append(0.1)
                        method_scores[method_class].append(0.1)
                    else:
                        penalty = min(prediction / 10, 1.5)
                        accuracy_scores.append(penalty)
                        method_scores[method_class].append(penalty)
                        
                except Exception as e:
                    print(f"Validation error for {variant}: {str(e)}")
                    continue
    
    if accuracy_scores:
        avg_error = np.mean(accuracy_scores)
        accuracy_pct = max(0, (1 - avg_error) * 100)
        
        # Method-specific accuracy
        method_accuracies = {}
        for method, scores in method_scores.items():
            if scores:
                method_accuracies[method] = max(0, (1 - np.mean(scores)) * 100)
            else:
                method_accuracies[method] = 0
        
        return {
            'overall_validation_accuracy': round(accuracy_pct, 2),
            'avg_percentage_error': round(avg_error * 100, 2),
            'samples_tested': len(accuracy_scores),
            'method_accuracies': {k: round(v, 2) for k, v in method_accuracies.items()},
            'method_sample_counts': {k: len(v) for k, v in method_scores.items()},
            'model_version': 'Enhanced Hybrid: Naive/LightGBM/Holt-Winters'
        }
    else:
        return {
            'overall_validation_accuracy': 0,
            'avg_percentage_error': 100,
            'samples_tested': 0,
            'model_version': 'Enhanced Hybrid: Naive/LightGBM/Holt-Winters'
        }

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
    selected_method_class = request.args.get('forecasting_method_class', 'all')
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

    if selected_method_class != 'all' and 'forecasting_method_class' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['forecasting_method_class'] == selected_method_class]
    
    if selected_method != 'all' and 'forecasting_method' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['forecasting_method'] == selected_method]

    # Pagination
    ROWS_PER_PAGE = 50
    total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)
    start = (page - 1) * ROWS_PER_PAGE
    end = start + ROWS_PER_PAGE

    display_cols = [
        'variant_name', 'product_id', 'product_type', 'outlet_name', 
        'recommended_stock_2w', 'forecasting_method_class', 'forecasting_method',
        'risk_category', 'volume_category', 'avg_daily_sales', 'created_at'
    ]
    display_cols = [c for c in display_cols if c in forecast_df.columns]

    forecast_page = forecast_df.iloc[start:end][display_cols]
    table_html = forecast_page.to_html(classes='table table-striped', index=False, float_format='{:.3f}'.format)

    outlets = forecast_df['outlet_name'].unique().tolist() if 'outlet_name' in forecast_df.columns else []
    risk_categories = forecast_df['risk_category'].unique().tolist() if 'risk_category' in forecast_df.columns else []
    forecasting_method_classes = forecast_df['forecasting_method_class'].unique().tolist() if 'forecasting_method_class' in forecast_df.columns else []
    forecasting_methods = forecast_df['forecasting_method'].unique().tolist() if 'forecasting_method' in forecast_df.columns else []

    return render_template(
        'result.html',
        forecast_table=table_html,
        outlets=outlets,
        risk_categories=risk_categories,
        forecasting_method_classes=forecasting_method_classes,
        forecasting_methods=forecasting_methods,
        page=page,
        total_pages=total_pages,
        selected_outlet=selected_outlet,
        selected_risk=selected_risk,
        selected_method_class=selected_method_class,
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
    
    # Analysis by method class
    method_class_analysis = {}
    for method_class in forecast_df['forecasting_method_class'].unique():
        method_data = forecast_df[forecast_df['forecasting_method_class'] == method_class]
        method_class_analysis[method_class] = {
            'count': len(method_data),
            'avg_forecast': float(method_data['recommended_stock_2w'].mean()),
            'total_forecast': int(method_data['recommended_stock_2w'].sum()),
            'max_forecast': int(method_data['recommended_stock_2w'].max()),
            'median_forecast': float(method_data['recommended_stock_2w'].median())
        }
    
    # Analysis by specific method (including fallbacks)
    method_analysis = {}
    for method in forecast_df['forecasting_method'].unique():
        method_data = forecast_df[forecast_df['forecasting_method'] == method]
        method_analysis[method] = {
            'count': len(method_data),
            'avg_forecast': float(method_data['recommended_stock_2w'].mean()),
            'total_forecast': int(method_data['recommended_stock_2w'].sum())
        }
    
    return jsonify({
        'method_class_performance': method_class_analysis,
        'method_performance': method_analysis,
        'total_products': len(forecast_df),
        'ml_methods': ['lightgbm', 'holt_winters'],
        'conservative_methods': ['naive']
    })

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
                'quantity': 'sum',
                'product_type': 'first'
            })
            
            validation_results = validate_enhanced_model(df_daily)
            return jsonify(validation_results)
        except Exception as e:
            return jsonify({"error": f"Validation failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "No data available for validation"}), 404

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
    filter_method_class = request.args.get('forecasting_method_class', '').strip()
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
    forecast_df = apply_filter(forecast_df, 'forecasting_method_class', filter_method_class)
    forecast_df = apply_filter(forecast_df, 'forecasting_method', filter_method)

    # Sort by recommended stock
    forecast_df = forecast_df.sort_values('recommended_stock_2w', ascending=False)

    # Pagination
    ROWS_PER_PAGE = 50
    
    if page.lower() == 'all':
        forecast_page = forecast_df
        page_out = 'all'
        total_pages = 1
    else:
        try:
            page_int = int(page)
            total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)
            page_out = max(1, min(page_int, total_pages))
            
            start = (page_out - 1) * ROWS_PER_PAGE
            end = start + ROWS_PER_PAGE
            forecast_page = forecast_df.iloc[start:end]
        except ValueError:
            page_out = 1
            total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)
            forecast_page = forecast_df.iloc[:ROWS_PER_PAGE]

    return jsonify({
        "page": page_out,
        "total_pages": total_pages,
        "count": len(forecast_page),
        "results": forecast_page.to_dict(orient="records"),
        "model_version": "Enhanced Hybrid: Naive/LightGBM/Holt-Winters",
        "filters_applied": {
            "outlet_id": filter_outlet_id,
            "outlet_name": filter_outlet_name,
            "product_id": filter_product_id,
            "variant_name": filter_variant_name,
            "product_attribute": filter_product_attr,
            "product_type": filter_product_type,
            "volume_category": filter_volume_cat,
            "risk_category": filter_risk_cat,
            "forecasting_method_class": filter_method_class,
            "forecasting_method": filter_method
        }
    })

# Load forecast data if it exists
forecast_df = None
if os.path.exists('hybrid_forecast.csv'):
    try:
        forecast_df = pd.read_csv('hybrid_forecast.csv')
    except Exception as e:
        print(f"Warning: Could not load forecast data: {e}")
        forecast_df = pd.DataFrame()

@app.route("/forecast", methods=["GET"])
def get_forecast():
    global forecast_df
    
    if forecast_df is None or len(forecast_df) == 0:
        return jsonify({"error": "No forecast data available. Please upload data and run forecasting first."}), 404
    
    page = request.args.get("page", "1")
    ROWS_PER_PAGE = 50
    total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)

    # Columns to display (only include columns that exist in the DataFrame)
    display_cols = [c for c in [
        'product_id', 'variant_name', 'product_attribute', 'product_type',
        'outlet_id', 'outlet_name', 'recommended_stock_2w', 'volume_category',
        'risk_category', 'forecasting_method_class', 'forecasting_method',
        'avg_daily_sales', 'created_at'
    ] if c in forecast_df.columns]

    if page.lower() == 'all':
        forecast_page = forecast_df[display_cols]
        page_out = 'all'
    else:
        try:
            page_int = int(page)
            page_out = page_int
        except ValueError:
            page_int = 1
            page_out = 1

        # Make sure page_int is within range
        if page_int < 1:
            page_int = 1
            page_out = 1
        elif page_int > total_pages:
            page_int = total_pages
            page_out = total_pages

        start = (page_int - 1) * ROWS_PER_PAGE
        end = start + ROWS_PER_PAGE
        forecast_page = forecast_df.iloc[start:end][display_cols]

    return jsonify({
        "page": page_out,
        "total_pages": total_pages,
        "count": len(forecast_page),
        "results": forecast_page.to_dict(orient="records"),
        "model_version": "Enhanced Hybrid: Naive/LightGBM/Holt-Winters",
        "summary": {
            "total_forecast": int(forecast_df['recommended_stock_2w'].sum()) if 'recommended_stock_2w' in forecast_df.columns else 0,
            "avg_forecast": float(forecast_df['recommended_stock_2w'].mean()) if 'recommended_stock_2w' in forecast_df.columns else 0,
            "method_class_breakdown": forecast_df['forecasting_method_class'].value_counts().to_dict() if 'forecasting_method_class' in forecast_df.columns else {},
            "method_breakdown": forecast_df['forecasting_method'].value_counts().to_dict() if 'forecasting_method' in forecast_df.columns else {}
        }
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)