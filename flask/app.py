from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'sales_demand_forecasting.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define expected feature names
feature_columns = [
    'total_price', 'base_price', 'is_featured_sku', 'is_display_sku',
    'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7',
    'rolling_mean_3', 'expanding_mean', 'lag1_lag2_interaction', 'lag1_plus_lag2',
    'store_encoded', 'sku_encoded'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    # Get input values for day 1-4 from form
    day_1 = float(request.form['day_1'])
    day_2 = float(request.form['day_2'])
    day_3 = float(request.form['day_3'])
    day_4 = float(request.form['day_4'])

    # Compute derived features
    lag1_lag2_interaction = day_1 * day_2
    lag1_plus_lag2 = day_1 + day_2

    # Create a single row dataframe with all expected features
    input_data = pd.DataFrame([{
        'total_price': 100.0,               # default / example value
        'base_price': 80.0,                 # default / example value
        'is_featured_sku': 0,               # default value
        'is_display_sku': 1,                # default value
        'day_1': day_1,
        'day_2': day_2,
        'day_3': day_3,
        'day_4': day_4,
        'day_5': 0.0,
        'day_6': 0.0,
        'day_7': 0.0,
        'rolling_mean_3': np.mean([day_1, day_2, day_3]),
        'expanding_mean': np.mean([day_1, day_2, day_3, day_4]),
        'lag1_lag2_interaction': lag1_lag2_interaction,
        'lag1_plus_lag2': lag1_plus_lag2,
        'store_encoded': 500.0,             # average store value
        'sku_encoded': 300.0               # average sku value
    }], columns=feature_columns)

    # Predict
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction_text=f"{prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
