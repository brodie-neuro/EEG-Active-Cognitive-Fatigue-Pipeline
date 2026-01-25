# eeg_pipeline/analysis/16_regression_model.py
"""
Step 16: Integrated Predictive Model
Multiple regression predicting performance decline from neural markers.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

FEATURES_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR = pipeline_dir / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed")


def load_all_features():
    """Load and merge all feature files."""
    dfs = []
    
    # P3b features
    p3b_file = FEATURES_DIR / "p3b_features.csv"
    if p3b_file.exists():
        dfs.append(pd.read_csv(p3b_file))
    
    # Band power
    power_file = FEATURES_DIR / "band_power_features.csv"
    if power_file.exists():
        dfs.append(pd.read_csv(power_file))
    
    # Frequencies
    freq_file = FEATURES_DIR / "frequency_features.csv"
    if freq_file.exists():
        dfs.append(pd.read_csv(freq_file))
    
    # PAC
    pac_file = FEATURES_DIR / "pac_nodal_features.csv"
    if pac_file.exists():
        dfs.append(pd.read_csv(pac_file))
    
    # Connectivity
    conn_file = FEATURES_DIR / "connectivity_features.csv"
    if conn_file.exists():
        dfs.append(pd.read_csv(conn_file))
    
    if not dfs:
        return None
    
    # Merge all on subject
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='subject', how='outer')
    
    return merged


def main():
    if not SKLEARN_AVAILABLE:
        print("scikit-learn required for regression modelling")
        return
    
    print("--- Integrated Predictive Model ---")
    
    # Load all features
    df = load_all_features()
    if df is None or len(df) == 0:
        print("No features found. Run analysis steps 10-14 first.")
        return
    
    print(f"Loaded features for {len(df)} subjects")
    print(f"Features: {list(df.columns)}")
    
    # For now, we don't have behavioral Δd' so we'll demonstrate the model structure
    # In practice, you'd load behavioral data and merge it here
    
    # Create placeholder outcome (normally this would be Δd' from behavioral data)
    # For demonstration: predict theta power from other features
    outcome_col = 'theta_power' if 'theta_power' in df.columns else df.columns[1]
    
    # Select numeric predictors (excluding subject ID and outcome)
    predictor_cols = [c for c in df.columns if c not in ['subject', outcome_col, 'channel']]
    predictor_cols = [c for c in predictor_cols if df[c].dtype in [np.float64, np.int64]]
    
    if len(predictor_cols) < 2:
        print("Not enough predictors for regression")
        return
    
    # Prepare data
    X = df[predictor_cols].values
    y = df[outcome_col].values
    
    # Remove NaN rows
    valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
    X = X[valid]
    y = y[valid]
    
    if len(X) < 5:
        print(f"Only {len(X)} valid observations - need more for modelling")
        return
    
    print(f"\nModelling {outcome_col} from {len(predictor_cols)} predictors")
    print(f"Valid observations: {len(X)}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # Cross-validation
    cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    
    print(f"\nModel Results:")
    print(f"  R² (training): {model.score(X_scaled, y):.3f}")
    print(f"  R² (CV mean): {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Feature importance
    coefs = pd.DataFrame({
        'feature': predictor_cols,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nFeature Importance (top 5):")
    for _, row in coefs.head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    # Save results
    coefs.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    
    results = {
        'outcome': outcome_col,
        'n_predictors': len(predictor_cols),
        'n_observations': len(X),
        'r2_train': model.score(X_scaled, y),
        'r2_cv_mean': scores.mean(),
        'r2_cv_std': scores.std()
    }
    pd.DataFrame([results]).to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    
    print(f"\nSaved model results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
