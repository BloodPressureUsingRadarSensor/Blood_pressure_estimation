import pandas as pd
import numpy as np
import json
import os
import sys

# ================= CONFIG =================
MODEL_FILE = r"E:\Document\spml\radar_model_v1.json"

def predict_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # 1. Load Model
    if not os.path.exists(MODEL_FILE):
        print("Error: Trained model not found. Please run train_bp_model.py first.")
        return
    
    with open(MODEL_FILE, "r") as f:
        model_data = json.load(f)
    
    features = model_data["features"]
    intercepts = np.array(model_data["intercept"])
    coeffs = np.array(model_data["coefficients"])

    # 2. Load User Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check for required features
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Error: CSV is missing signal features required for prediction: {missing}")
        return

    # 3. Apply ML Model
    # X shape: (samples, num_features)
    # coeffs shape: (2, num_features) -> SBP, DBP
    X = df[features].values
    
    # Prediction: Y = X * W^T + b
    predictions = np.dot(X, coeffs.T) + intercepts

    # 4. Result Summarization
    sbp_results = predictions[:, 0]
    dbp_results = predictions[:, 1]

    # Filter by reliability if available
    if 'ReliabilityScore' in df.columns:
        valid_mask = df['ReliabilityScore'] >= 70
        if valid_mask.any():
            final_sbp = np.mean(sbp_results[valid_mask])
            final_dbp = np.mean(dbp_results[valid_mask])
            quality = "HIGH"
        else:
            final_sbp = np.mean(sbp_results)
            final_dbp = np.mean(dbp_results)
            quality = "LOW (Caution: Signal quality was sub-optimal)"
    else:
        final_sbp = np.mean(sbp_results)
        final_dbp = np.mean(dbp_results)
        quality = "UNKNOWN (Reliability score missing)"

    print("\n" + "="*40)
    print(f"RADAR BLOOD PRESSURE PREDICTION REPORT")
    print("="*40)
    print(f"Subject Data: {os.path.basename(csv_path)}")
    print(f"Signal Quality: {quality}")
    print("-" * 40)
    print(f"ESTIMATED SYSTOLIC (SBP):  {final_sbp:.1f} mmHg")
    print(f"ESTIMATED DIASTOLIC (DBP): {final_dbp:.1f} mmHg")
    print("-" * 40)
    # print(f"Interpretation: {interpret_bp(final_sbp, final_dbp)}")
    print("="*40)

# def interpret_bp(sbp, dbp):
#     if sbp < 120 and dbp < 80: return "NORMAL"
#     if 120 <= sbp <= 129 and dbp < 80: return "ELEVATED"
#     if 130 <= sbp <= 139 or 80 <= dbp <= 89: return "HYPERTENSION STAGE 1"
#     if sbp >= 140 or dbp >= 90: return "HYPERTENSION STAGE 2"
#     return "UNKNOWN"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_bp.py <path_to_user_csv>")
    else:
        predict_from_csv(sys.argv[1])
