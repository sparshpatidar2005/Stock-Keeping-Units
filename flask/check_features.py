import pickle

with open('sales_demand_forecasting.pkl', 'rb') as f:
    model = pickle.load(f)

try:
    print("✅ Model loaded.")
    print("Expected features:\n", model.feature_names_in_)
except AttributeError:
    print("❌ Model has no attribute 'feature_names_in_' — might not be a scikit-learn compatible estimator.")
