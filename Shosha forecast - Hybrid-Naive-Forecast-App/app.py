import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import joblib
import secrets
from auth import auth_bp 
import warnings
warnings.filterwarnings('ignore')

# ---------- Constants ----------
UPLOAD_FOLDER = 'uploads'
DATA_FILE = 'all_data.csv'
FORECAST_FILE = 'simple_forecast_results.json'

# ---------- Flask Setup ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Secret key setup
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Register the auth blueprint
app.register_blueprint(auth_bp)

# ======================================================================
#                    HYBRID FORECASTING: LOW-VOLUME + HIGH-VOLUME
# ======================================================================

def identify_product_type(variant_name, product_type, avg_daily_sales, max_daily_sales):
    """
    Identify if product is consumable/high-volume or traditional low-volume retail
    """
    variant_lower = str(variant_name).lower()
    product_type_lower = str(product_type).lower()
    
    # High-volume consumable keywords
    consumable_keywords = [
        'ultra bar', 'vaporesso', 'pod', 'coil', 'disposable', 'cartridge',
        'replacement', 'refill', 'atomizer', 'tank', 'clearomizer'
    ]
    
    # Check if it's a consumable by name
    is_consumable = any(keyword in variant_lower for keyword in consumable_keywords)
    
    # Check if it's high-volume by sales pattern
    is_high_volume = avg_daily_sales > 5 or max_daily_sales > 20
    
    if is_consumable or is_high_volume:
        return "High-Volume Consumable"
    elif avg_daily_sales > 2:
        return "Medium-Volume Product"
    else:
        return "Low-Volume Retail"

def classify_volume_category(avg_daily, product_type_category):
    """Enhanced volume classification considering product type"""
    if product_type_category == "High-Volume Consumable":
        if avg_daily < 1:
            return "Low Consumable"
        elif avg_daily < 5:
            return "Medium Consumable" 
        elif avg_daily < 15:
            return "High Consumable"
        else:
            return "Very High Consumable"
    else:
        # Original low-volume classification
        if avg_daily < 0.1:
            return "Very Low Volume"
        elif avg_daily < 0.5:
            return "Low Volume"
        elif avg_daily < 1.0:
            return "Medium-Low Volume"
        elif avg_daily < 2.0:
            return "Medium Volume"
        else:
            return "Higher Volume"

def classify_risk(frequency, avg_sales, recent_avg):
    """Classify products by inventory risk"""
    if frequency < 0.1:  # Sales < 10% of days
        return "Very Low Velocity"
    elif frequency < 0.3:  # Sales < 30% of days  
        return "Low Velocity"
    elif recent_avg > avg_sales * 1.5:  # Recent sales much higher
        return "Growing Demand"
    elif recent_avg < avg_sales * 0.5:  # Recent sales much lower
        return "Declining Demand" 
    else:
        return "Stable"

def calculate_hybrid_forecast(variant_data, variant_name, product_type):
    """
    Calculate forecast using appropriate logic based on product type
    """
    # Basic statistics
    total_sales = variant_data['quantity'].sum()
    total_days = len(variant_data)
    recent_14d = variant_data.tail(14)
    recent_sales = recent_14d['quantity'].sum()
    non_zero_days = (variant_data['quantity'] > 0).sum()
    max_daily = variant_data['quantity'].max()
    
    # Calculate daily averages
    avg_daily_historical = total_sales / total_days
    avg_daily_recent = recent_sales / len(recent_14d)
    sales_frequency = non_zero_days / total_days
    
    # Identify product type
    product_type_category = identify_product_type(
        variant_name, product_type, avg_daily_historical, max_daily
    )
    
    if product_type_category == "High-Volume Consumable":
        # HIGH-VOLUME CONSUMABLE LOGIC
        # Use replenishment-style forecasting
        
        # Weight recent sales more heavily for consumables
        weighted_daily = (avg_daily_recent * 0.7) + (avg_daily_historical * 0.3)
        base_forecast = weighted_daily * 14
        
        # Growth factor for trending products
        if avg_daily_recent > avg_daily_historical * 1.2:
            growth_factor = 1.2  # 20% buffer for growing products
        else:
            growth_factor = 1.1  # 10% standard buffer
        
        forecast = base_forecast * growth_factor
        
        # Apply reasonable caps for consumables (much higher than low-volume)
        if avg_daily_historical > 15:  # Very high volume
            forecast = min(forecast, 500)
        elif avg_daily_historical > 5:  # High volume
            forecast = min(forecast, 200)
        elif avg_daily_historical > 1:  # Medium volume
            forecast = min(forecast, 100)
        else:  # Lower volume consumables
            forecast = min(forecast, 50)
        
        # Minimum viable quantity for consumables
        forecast = max(5, round(forecast))
        
    else:
        # LOW-VOLUME RETAIL LOGIC (original logic)
        base_forecast = max(avg_daily_historical, avg_daily_recent) * 14
        
        # Apply original low-volume caps
        if avg_daily_historical < 0.1:
            forecast = min(base_forecast, 3)
        elif avg_daily_historical < 0.5:
            forecast = min(base_forecast, 7)
        elif avg_daily_historical < 1.0:
            forecast = min(base_forecast, 14)
        elif avg_daily_historical < 2.0:
            forecast = min(base_forecast, 28)
        else:
            forecast = min(base_forecast, 50)
        
        forecast = max(1, round(forecast))
        
        # Original data quality checks for low-volume
        if forecast > max_daily * 5:
            old_forecast = forecast
            forecast = min(forecast, max_daily * 3)
            print(f"CAPPED: {variant_name}: {old_forecast} -> {forecast} (max daily: {max_daily})")
    
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

def calculate_hybrid_metrics_fast(df_daily):
    """Hybrid calculation handling both low-volume and high-volume products"""
    results = []
    variants = df_daily['variant_name'].unique()
    
    print(f"Calculating hybrid metrics for {len(variants)} variants...")
    
    for i, variant in enumerate(variants):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(variants)} variants...")
        
        variant_data = df_daily[df_daily['variant_name'] == variant].copy()
        
        if len(variant_data) >= 7:  # Need at least 1 week of data
            # Sort by date
            variant_data = variant_data.sort_values('date')
            
            # Get product details
            recent_record = variant_data.iloc[-1]
            product_type = recent_record.get('product_type', '')
            
            # Calculate forecast using hybrid logic
            forecast_result = calculate_hybrid_forecast(
                variant_data, variant, product_type
            )
            
            # Classify using enhanced categories
            volume_category = classify_volume_category(
                forecast_result['avg_daily_historical'],
                forecast_result['product_type_category']
            )
            
            risk_category = classify_risk(
                forecast_result['sales_frequency'],
                forecast_result['avg_daily_historical'],
                forecast_result['avg_daily_recent']
            )
            
            results.append({
                'variant_name': variant,
                'product_id': recent_record.get('product_id', ''),
                'product_type': product_type,
                'product_attribute': recent_record.get('product_attribute', ''),
                'total_sales_history': int(forecast_result['total_sales']),
                'recent_14d_sales': int(forecast_result['recent_sales']),
                'avg_daily_sales': round(forecast_result['avg_daily_historical'], 2),
                'avg_daily_recent': round(forecast_result['avg_daily_recent'], 2),
                'max_daily_sale': int(forecast_result['max_daily']),
                'sales_frequency_pct': round(forecast_result['sales_frequency'] * 100, 1),
                'recommended_stock_2w': forecast_result['forecast'],
                'volume_category': volume_category,
                'risk_category': risk_category,
                'product_type_category': forecast_result['product_type_category'],
                'last_date': recent_record['date'].strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(results)

def distribute_to_outlets_hybrid(variant_forecasts, raw_df):
    """Enhanced outlet distribution handling both product types"""
    results = []
    
    print("Calculating outlet shares and distributing hybrid forecasts...")
    
    # Calculate outlet shares for each variant
    outlet_shares = raw_df.groupby(['variant_name', 'outlet_id', 'outlet_name']).agg({
        'quantity': 'sum'
    }).reset_index()
    
    # Get total per variant
    variant_totals = outlet_shares.groupby('variant_name')['quantity'].sum().reset_index()
    variant_totals.rename(columns={'quantity': 'total_quantity'}, inplace=True)
    
    # Calculate shares
    outlet_shares = outlet_shares.merge(variant_totals, on='variant_name')
    outlet_shares['share'] = outlet_shares['quantity'] / outlet_shares['total_quantity']
    outlet_shares['share'] = outlet_shares['share'].fillna(1.0)
    
    # Track adjustments
    high_forecast_adjustments = 0
    consumable_distributions = 0
    
    # Distribute forecasts with product-type-aware logic
    for _, forecast_row in variant_forecasts.iterrows():
        variant = forecast_row['variant_name']
        variant_outlets = outlet_shares[outlet_shares['variant_name'] == variant]
        
        if len(variant_outlets) > 0:
            for _, outlet_row in variant_outlets.iterrows():
                # Calculate base allocation
                raw_allocation = forecast_row['recommended_stock_2w'] * outlet_row['share']
                
                # Apply caps based on product type
                product_type_cat = forecast_row.get('product_type_category', 'Low-Volume Retail')
                
                if product_type_cat == "High-Volume Consumable":
                    consumable_distributions += 1
                    # Higher caps for consumables
                    volume_cat = forecast_row.get('volume_category', 'Medium Consumable')
                    
                    if volume_cat == "Very High Consumable":
                        outlet_cap = 300
                    elif volume_cat == "High Consumable":
                        outlet_cap = 150
                    elif volume_cat == "Medium Consumable":
                        outlet_cap = 75
                    else:
                        outlet_cap = 30
                    
                    # Less restrictive historical checks for consumables
                    historical_outlet_sales = outlet_row['quantity']
                    max_reasonable = max(10, historical_outlet_sales * 2)  # Allow 2x historical
                    
                else:
                    # Original logic for low-volume products
                    volume_cat = forecast_row.get('volume_category', 'Medium Volume')
                    
                    if volume_cat == "Very Low Volume":
                        outlet_cap = 2
                    elif volume_cat == "Low Volume":
                        outlet_cap = 5
                    elif volume_cat == "Medium-Low Volume":
                        outlet_cap = 10
                    elif volume_cat == "Medium Volume":
                        outlet_cap = 20
                    else:
                        outlet_cap = 35
                    
                    historical_outlet_sales = outlet_row['quantity']
                    max_reasonable = max(3, historical_outlet_sales + 5)
                
                # Apply caps
                outlet_forecast = max(1, min(round(raw_allocation), outlet_cap))
                
                # Apply historical reasonableness check
                if outlet_forecast > max_reasonable:
                    high_forecast_adjustments += 1
                    if high_forecast_adjustments <= 10:  # Log first 10
                        print(f"Capping {variant} at {outlet_row['outlet_name']}: {outlet_forecast} -> {max_reasonable}")
                    outlet_forecast = max_reasonable
                
                results.append({
                    'variant_name': variant,
                    'product_id': forecast_row['product_id'],
                    'product_type': forecast_row['product_type'],
                    'product_attribute': forecast_row['product_attribute'],
                    'outlet_id': outlet_row['outlet_id'],
                    'outlet_name': outlet_row['outlet_name'],
                    'recommended_stock_2w': outlet_forecast,
                    'volume_category': volume_cat,
                    'risk_category': forecast_row.get('risk_category', 'Unknown'),
                    'product_type_category': product_type_cat,
                    'outlet_share': round(outlet_row['share'] * 100, 1),
                    'historical_outlet_sales': int(historical_outlet_sales),
                    'created_at': forecast_row['last_date']
                })
        else:
            # Fallback if no outlet data
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
                'outlet_share': 100.0,
                'historical_outlet_sales': 0,
                'created_at': forecast_row['last_date']
            })
    
    print(f"Distributed {consumable_distributions} high-volume consumable forecasts")
    if high_forecast_adjustments > 0:
        print(f"Applied caps to {high_forecast_adjustments} outlet-variant combinations")
    
    return pd.DataFrame(results)

def process_hybrid_forecast(raw_df):
    """Hybrid processing function handling both low-volume and high-volume products"""
    print("Processing hybrid forecast (low-volume + high-volume logic)...")
    
    # Basic data cleaning
    raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
    raw_df['date'] = raw_df['created_at'].dt.date
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    
    # Aggregate at variant level
    print("Aggregating daily sales by variant (all outlets combined)...")
    df_daily = raw_df.groupby(['variant_name', 'date'], as_index=False).agg({
        'quantity': 'sum',
        'product_id': 'first',
        'product_type': 'first',
        'product_attribute': 'first'
    })
    
    print(f"Processing {len(df_daily['variant_name'].unique())} variants with hybrid logic")
    
    # Calculate forecasts using hybrid approach
    forecast_df = calculate_hybrid_metrics_fast(df_daily)
    
    # Show breakdown of product types
    type_breakdown = forecast_df['product_type_category'].value_counts()
    print(f"Product type breakdown:")
    for ptype, count in type_breakdown.items():
        print(f"   • {ptype}: {count} variants")
    
    # Distribute to outlets with hybrid logic
    print("Distributing forecasts to outlets with hybrid controls...")
    final_results = distribute_to_outlets_hybrid(forecast_df, raw_df)
    
    # Sort by recommended stock (highest first)
    final_results = final_results.sort_values('recommended_stock_2w', ascending=False)
    
    # Enhanced summary statistics
    total_variants = len(final_results['variant_name'].unique())
    total_outlets = len(final_results['outlet_name'].unique())
    
    # Breakdowns
    volume_breakdown = final_results['volume_category'].value_counts().to_dict()
    risk_breakdown = final_results['risk_category'].value_counts().to_dict()
    product_type_breakdown = final_results['product_type_category'].value_counts().to_dict()
    
    # Save detailed results
    with open(FORECAST_FILE, 'w') as f:
        json.dump({
            'forecast_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_variants': total_variants,
            'total_outlets': total_outlets,
            'total_forecasts': len(final_results),
            'summary_stats': {
                'avg_recommended_stock': float(final_results['recommended_stock_2w'].mean()),
                'total_recommended_stock': int(final_results['recommended_stock_2w'].sum()),
                'median_recommended_stock': float(final_results['recommended_stock_2w'].median()),
                'max_recommended_stock': int(final_results['recommended_stock_2w'].max()),
                'min_recommended_stock': int(final_results['recommended_stock_2w'].min()),
                'p95_recommended_stock': float(final_results['recommended_stock_2w'].quantile(0.95)),
                'p90_recommended_stock': float(final_results['recommended_stock_2w'].quantile(0.90))
            },
            'volume_category_breakdown': volume_breakdown,
            'risk_category_breakdown': risk_breakdown,
            'product_type_breakdown': product_type_breakdown,
            'forecasts_by_product_type': {
                ptype: {
                    'count': int(group['recommended_stock_2w'].count()),
                    'avg_forecast': float(group['recommended_stock_2w'].mean()),
                    'max_forecast': int(group['recommended_stock_2w'].max()),
                    'total_forecast': int(group['recommended_stock_2w'].sum())
                }
                for ptype, group in final_results.groupby('product_type_category')
            },
            'model_info': {
                'model_type': 'Hybrid Low-Volume + High-Volume Forecasting',
                'methodology': 'Product-type aware forecasting with separate logic for consumables vs retail',
                'features_used': ['recent_sales_history', 'historical_avg', 'outlet_historical_share', 'product_type_classification'],
                'forecast_horizon': '14 days',
                'processing_time': 'Fast (< 5 minutes)',
                'improvements': [
                    'Automatic product type classification',
                    'High-volume consumable forecasting logic',
                    'Replenishment-style forecasting for fast-movers',
                    'Enhanced caps based on product category',
                    'Growth factor for trending consumables'
                ]
            }
        }, f, indent=2)
    
    print(f"Generated hybrid forecasts for {len(final_results)} variant-outlet combinations")
    print(f"Max forecast: {final_results['recommended_stock_2w'].max()}")
    print(f"95th percentile: {final_results['recommended_stock_2w'].quantile(0.95):.1f}")
    
    # Show high-volume products
    high_volume = final_results[final_results['product_type_category'] == 'High-Volume Consumable'].nlargest(10, 'recommended_stock_2w')
    if len(high_volume) > 0:
        print(f"\nTop 10 high-volume consumable forecasts:")
        for _, row in high_volume.iterrows():
            print(f"   • {row['variant_name'][:40]:<40} | Forecast: {row['recommended_stock_2w']:>3}")
    
    return final_results

def validate_simple_model(df_daily, test_days=14):
    """Simple validation: test how well recent average predicts immediate past"""
    accuracy_scores = []
    variants = df_daily['variant_name'].unique()
    
    print(f"Validating model on {len(variants)} variants...")
    
    for variant in variants:
        variant_data = df_daily[df_daily['variant_name'] == variant].sort_values('date')
        if len(variant_data) >= 28:  # Need enough data for train/test split
            # Use first half to predict second half
            midpoint = len(variant_data) // 2
            train_data = variant_data.iloc[:midpoint]
            test_data = variant_data.iloc[midpoint:midpoint+test_days]
            
            if len(test_data) == test_days:
                # Simple prediction: average of training data
                train_avg_daily = train_data['quantity'].sum() / len(train_data)
                prediction = train_avg_daily * test_days
                actual = test_data['quantity'].sum()
                
                # Calculate percentage error (capped at 200%)
                if actual > 0:
                    error = abs(prediction - actual) / actual
                    accuracy_scores.append(min(error, 2.0))
                elif prediction <= 1:  # Correctly predicted very low sales
                    accuracy_scores.append(0.1)  # Small error for correct low prediction
                else:
                    accuracy_scores.append(1.0)  # Moderate error for overprediction
    
    if accuracy_scores:
        avg_error = np.mean(accuracy_scores)
        accuracy_pct = max(0, (1 - avg_error) * 100)
        return {
            'validation_accuracy': round(accuracy_pct, 2),
            'avg_percentage_error': round(avg_error * 100, 2),
            'samples_tested': len(accuracy_scores)
        }
    else:
        return {
            'validation_accuracy': 0,
            'avg_percentage_error': 100,
            'samples_tested': 0
        }

# ======================================================================
#                         FLASK ROUTES
# ======================================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    upload_mode = request.args.get('upload', None)

    if os.path.exists('simple_forecast.csv') and upload_mode != '1':
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

        # Combine with existing data if available
        if os.path.exists(DATA_FILE):
            old_df = pd.read_csv(DATA_FILE)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Remove duplicates and save
        combined_df = combined_df.drop_duplicates()
        combined_df.to_csv(DATA_FILE, index=False)

        try:
            # Process with hybrid forecasting
            forecast_df = process_hybrid_forecast(combined_df)
            
            if forecast_df is not None and len(forecast_df) > 0:
                forecast_df.to_csv('simple_forecast.csv', index=False)
                print(f"Hybrid forecast completed! Generated {len(forecast_df)} forecasts")
            else:
                return "Error: No forecasts were generated. Check if data has sufficient history.", 500
            
        except Exception as e:
            print(f"Error processing forecast: {e}")
            return f"Error processing forecast: {e}", 500

        return redirect(url_for('show_results', page=1))

    if upload_mode == '1' or not os.path.exists('simple_forecast.csv'):
        return render_template('upload.html')

    return redirect(url_for('show_results', page=1))

@app.route('/results', methods=['GET'])
def show_results():
    # Check if user is logged in
    user = session.get("user")
    if not user:
        return redirect(url_for("auth.login"))

    # Get query params
    page = int(request.args.get('page', 1))
    selected_outlet = request.args.get('outlet', 'all')
    selected_risk = request.args.get('risk', 'all')
    selected_volume = request.args.get('volume', 'all')
    selected_product_type = request.args.get('product_type_category', 'all')
    filter_variant_name = request.args.get('variant_name', '').strip()

    if not os.path.exists('simple_forecast.csv'):
        return redirect(url_for('home', upload=1))

    forecast_df = pd.read_csv('simple_forecast.csv')

    # Apply role-based filtering
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

    if selected_volume != 'all' and 'volume_category' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['volume_category'] == selected_volume]

    if selected_product_type != 'all' and 'product_type_category' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['product_type_category'] == selected_product_type]

    # Pagination
    ROWS_PER_PAGE = 50
    total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)
    start = (page - 1) * ROWS_PER_PAGE
    end = start + ROWS_PER_PAGE

    # Display columns - include new product type category
    display_cols = [
        'variant_name', 'product_id', 'product_type', 'product_attribute',
        'outlet_id', 'outlet_name', 'recommended_stock_2w', 'volume_category', 
        'risk_category', 'product_type_category', 'outlet_share', 'created_at'
    ]
    display_cols = [c for c in display_cols if c in forecast_df.columns]

    forecast_page = forecast_df.iloc[start:end][display_cols]

    # Convert to HTML table
    table_html = forecast_page.to_html(classes='table table-striped', index=False, float_format='{:.2f}'.format)

    # Get filter options
    outlets = forecast_df['outlet_name'].unique().tolist() if 'outlet_name' in forecast_df.columns else []
    risk_categories = forecast_df['risk_category'].unique().tolist() if 'risk_category' in forecast_df.columns else []
    volume_categories = forecast_df['volume_category'].unique().tolist() if 'volume_category' in forecast_df.columns else []
    product_type_categories = forecast_df['product_type_category'].unique().tolist() if 'product_type_category' in forecast_df.columns else []

    return render_template(
        'result.html',
        forecast_table=table_html,
        outlets=outlets,
        risk_categories=risk_categories,
        volume_categories=volume_categories,
        product_type_categories=product_type_categories,
        page=page,
        total_pages=total_pages,
        selected_outlet=selected_outlet,
        selected_risk=selected_risk,
        selected_volume=selected_volume,
        selected_product_type=selected_product_type,
        filter_variant_name=filter_variant_name,
        total_items=len(forecast_df)
    )

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint for forecast summary statistics"""
    if os.path.exists(FORECAST_FILE):
        with open(FORECAST_FILE, 'r') as f:
            summary = json.load(f)
        return jsonify(summary)
    else:
        return jsonify({"error": "No forecast data available. Please upload data and generate forecasts first."}), 404

@app.route('/api/validate', methods=['GET'])
def api_validate():
    """API endpoint for model validation"""
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
            
            validation_results = validate_simple_model(df_daily)
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
    filter_product_type_cat = request.args.get('product_type_category', '').strip()
    match_type = request.args.get('match', 'contains').lower()

    if not os.path.exists('simple_forecast.csv'):
        return jsonify({"error": "No forecast available. Please upload data first."}), 404

    forecast_df = pd.read_csv('simple_forecast.csv')

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

    # Sort by recommended stock
    forecast_df = forecast_df.sort_values('recommended_stock_2w', ascending=False)

    ROWS_PER_PAGE = 50
    total_pages = max(1, (len(forecast_df) - 1) // ROWS_PER_PAGE + 1)

    if page.lower() == 'all':
        display_cols = [c for c in [
            'product_id', 'variant_name', 'product_attribute', 'product_type',
            'outlet_id', 'outlet_name', 'recommended_stock_2w', 'volume_category',
            'risk_category', 'product_type_category', 'outlet_share', 'created_at'
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
            'risk_category', 'product_type_category', 'outlet_share', 'created_at'
        ] if c in forecast_df.columns]
        forecast_page = forecast_df.iloc[start:end][display_cols]

    return jsonify({
        "page": page_out,
        "total_pages": total_pages,
        "count": len(forecast_page),
        "results": forecast_page.to_dict(orient="records")
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)