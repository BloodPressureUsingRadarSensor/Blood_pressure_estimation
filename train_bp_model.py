import pandas as pd
import numpy as np
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATA_DIR = r"E:\Document\spml\BP_Datasets"
MODEL_OUTPUT = r"E:\Document\spml\radar_model_v1.json"
RELIABILITY_THRESHOLD = 70

# Features to use for training
FEATURES = [
    'HR', 'RR', 'RMS', 'Energy', 'PeakToPeak', 
    'IBI_Mean', 'IBI_STD', 'Skewness', 'Kurtosis', 
    'DominantFreq', 'SpectralCentroid', 'SpectralEntropy', 
    'PulseWidth', 'RiseTime', 'BandPower'
]

TARGETS = ['SBP_Estimate', 'DBP_Estimate']

def train_model():
    print(f"Scanning for data in {DATA_DIR}...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in BP_Datasets!")
        return

    all_data = []
    total_files = len(csv_files)
    
    for i, f in enumerate(csv_files):
        try:
            df = pd.read_csv(f)
            # Filter by reliability to ensure high-quality training signals
            if 'ReliabilityScore' in df.columns:
                clean_df = df[df['ReliabilityScore'] >= RELIABILITY_THRESHOLD]
                if not clean_df.empty:
                    all_data.append(clean_df)
            
            if (i+1) % 10 == 0 or (i+1) == total_files:
                print(f"Processed {i+1}/{total_files} files...")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No high-quality data found after filtering!")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    print(f"\nFinal dataset size: {len(master_df)} rows.")

    # Drop rows with NaN in features or targets
    master_df = master_df.dropna(subset=FEATURES + TARGETS)
    
    X = master_df[FEATURES]
    Y = master_df[TARGETS]

    # Split for validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("\nTraining Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Predictions for validation
    Y_pred = model.predict(X_test)

    # Metrics
    mae_sbp = mean_absolute_error(Y_test['SBP_Estimate'], Y_pred[:, 0])
    mae_dbp = mean_absolute_error(Y_test['DBP_Estimate'], Y_pred[:, 1])
    r2 = r2_score(Y_test, Y_pred)

    print(f"\n=== Model Performance ===")
    print(f"SBP Mean Absolute Error: {mae_sbp:.2f} mmHg")
    print(f"DBP Mean Absolute Error: {mae_dbp:.2f} mmHg")
    print(f"Overall R² Score: {r2:.4f}")

    # Save model weights to JSON (Syllabus/White-Box compliance)
    model_data = {
        "features": FEATURES,
        "intercept": model.intercept_.tolist(),
        "coefficients": model.coef_.tolist(),
        "metadata": {
            "mae_sbp": float(mae_sbp),
            "mae_dbp": float(mae_dbp),
            "r2_score": float(r2),
            "training_samples": int(len(master_df))
        }
    }

    with open(MODEL_OUTPUT, "w") as f:
        json.dump(model_data, f, indent=4)
    
    print(f"\nModel intelligence exported to: {MODEL_OUTPUT}")

    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test['SBP_Estimate'], Y_pred[:, 0], alpha=0.5, color='blue')
    plt.plot([80, 180], [80, 180], 'r--')
    plt.title("SBP: Truth vs Predicted")
    plt.xlabel("Actual SBP (mmHg)")
    plt.ylabel("Predicted SBP (mmHg)")

    plt.subplot(1, 2, 2)
    plt.scatter(Y_test['DBP_Estimate'], Y_pred[:, 1], alpha=0.5, color='green')
    plt.plot([50, 110], [50, 110], 'r--')
    plt.title("DBP: Truth vs Predicted")
    plt.xlabel("Actual DBP (mmHg)")
    plt.ylabel("Predicted DBP (mmHg)")

    plt.tight_layout()
    plot_path = MODEL_OUTPUT.replace(".json", "_performance.png")
    plt.savefig(plot_path)
    print(f"Performance plot saved: {plot_path}")
    print("\nTraining Complete.")

if __name__ == "__main__":
    train_model()
