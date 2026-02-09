"""
TB Breath Analyzer - Model Training
Random Forest classifier with baseline normalization.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from config import SENSOR_DATA_NEW_FILE, BASELINE_DATA_FILE, MODEL_FILE, SENSOR_NAMES, MODELS_DIR


def log(msg: str):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_baseline():
    """Load baseline values from file."""
    log("Loading baseline data...")
    if not BASELINE_DATA_FILE.exists():
        print(f"Error: {BASELINE_DATA_FILE} not found. Run collect_baseline.py first.")
        sys.exit(1)
    
    baseline_df = pd.read_csv(BASELINE_DATA_FILE)
    baseline = {name: baseline_df[name].values[0] for name in SENSOR_NAMES}
    
    log(f"Loaded baseline from {BASELINE_DATA_FILE}")
    for name, value in baseline.items():
        print(f"  {name}: {value:.2f}")
    print()
    return baseline


def load_data():
    """Load sensor data from CSV."""
    log("Loading sensor data...")
    if not SENSOR_DATA_NEW_FILE.exists():
        print(f"Error: {SENSOR_DATA_NEW_FILE} not found.")
        sys.exit(1)
    
    df = pd.read_csv(SENSOR_DATA_NEW_FILE)
    log(f"Loaded {len(df)} samples")
    
    log("Distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    print()
    return df


def normalize_data(df: pd.DataFrame, baseline: dict):
    """Normalize raw values using baseline."""
    log("Normalizing data...")
    normalized_df = df.copy()
    
    for name in SENSOR_NAMES:
        raw_col = f"{name}_raw"
        pct_col = f"{name}_pct"
        base = baseline[name]
        normalized_df[pct_col] = (df[raw_col] - base) / base if base > 0 else 0.0
        normalized_df = normalized_df.drop(columns=[raw_col])
    
    log("Normalization complete")
    print()
    return normalized_df


def train_model(df: pd.DataFrame):
    """Train and evaluate Random Forest classifier."""
    log("Preparing features...")
    feature_cols = [f"{name}_pct" for name in SENSOR_NAMES]
    X = df[feature_cols].values
    y = df['label'].values
    
    log("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print()
    
    log("Training Random Forest...")
    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=1
    )
    clf.fit(X_train, y_train)
    log(f"Training complete ({time.time() - start:.2f}s)")
    
    log("Evaluating...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print()
    print("=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print()
    log(f"Accuracy: {accuracy:.2%}")
    print()
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, clf.classes_)
    
    log("Feature Importance:")
    for name, imp in zip(feature_cols, clf.feature_importances_):
        print(f"  {name}: {imp:.3f}")
    print()
    
    return clf


def plot_confusion_matrix(cm: np.ndarray, classes: np.ndarray):
    """Save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    fig_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved to {fig_path}")
    plt.show()


def save_model(clf: RandomForestClassifier):
    """Save model to disk."""
    log("Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    log(f"Saved to {MODEL_FILE}")


def main():
    print("=" * 50)
    print("TB Breath Analyzer - Model Training")
    print("=" * 50)
    print()
    
    baseline = load_baseline()
    df = load_data()
    df_normalized = normalize_data(df, baseline)
    
    if df_normalized['label'].nunique() < 2:
        print("Error: Need at least 2 classes.")
        sys.exit(1)
    
    clf = train_model(df_normalized)
    save_model(clf)
    
    print()
    log("Training complete")


if __name__ == "__main__":
    main()
