"""
TB Breath Analyzer - Model Training
Random Forest classifier trained on KDE-augmented data.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, MODEL_FILE, DATA_DIR

TRAIN_AUGMENTED_FILE = DATA_DIR / "train_augmented.csv"
TEST_HIDDEN_FILE = DATA_DIR / "test_real_hidden.csv"
SENSOR_COLUMNS = ["MQ3_raw", "MQ135_raw", "MQ2_raw"]


def log(msg: str):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_training_data() -> pd.DataFrame:
    """Load augmented training dataset."""
    log("Loading training data...")
    
    if not TRAIN_AUGMENTED_FILE.exists():
        print(f"Error: {TRAIN_AUGMENTED_FILE} not found. Run augment.py first.")
        sys.exit(1)
    
    df = pd.read_csv(TRAIN_AUGMENTED_FILE)
    real = len(df[df['is_synthetic'] == 0])
    synth = len(df[df['is_synthetic'] == 1])
    
    log(f"Loaded {len(df)} samples ({real} real + {synth} synthetic)")
    
    log("Distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    print()
    
    return df


def load_test_data() -> pd.DataFrame:
    """Load real-only test dataset."""
    log("Loading test data...")
    
    if not TEST_HIDDEN_FILE.exists():
        print(f"Error: {TEST_HIDDEN_FILE} not found. Run augment.py first.")
        sys.exit(1)
    
    df = pd.read_csv(TEST_HIDDEN_FILE)
    
    if 'is_synthetic' in df.columns:
        assert df['is_synthetic'].sum() == 0
    
    log(f"Loaded {len(df)} test samples")
    
    log("Distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    print()
    
    return df


def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    log("Preparing features...")
    
    X_train = train_df[SENSOR_COLUMNS].values
    y_train = train_df['label'].values
    
    log(f"Training on {len(X_train)} samples...")
    start = time.time()
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    log(f"Training complete ({time.time() - start:.2f}s)")
    print()
    
    return clf


def evaluate_model(clf: RandomForestClassifier, test_df: pd.DataFrame) -> float:
    """Evaluate model on test set."""
    log("Evaluating...")
    
    X_test = test_df[SENSOR_COLUMNS].values
    y_test = test_df['label'].values
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print()
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print()
    
    log(f"Accuracy: {accuracy:.2%}")
    print()
    
    log("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    log("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    
    plot_confusion_matrix(cm, clf.classes_)
    
    log("Feature Importance:")
    for name, imp in zip(SENSOR_COLUMNS, clf.feature_importances_):
        print(f"  {name}: {imp:.3f}")
    print()
    
    return accuracy


def plot_confusion_matrix(cm: np.ndarray, classes: np.ndarray):
    """Save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = MODELS_DIR / "confusion_matrix_augmented.png"
    plt.savefig(fig_path, dpi=150)
    log(f"Saved to {fig_path}")
    plt.close()


def save_model(clf: RandomForestClassifier):
    """Save model to disk."""
    log("Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(clf, MODELS_DIR / "tb_model_augmented.pkl")
    joblib.dump(clf, MODEL_FILE)
    log(f"Saved to {MODEL_FILE}")


def main():
    print("=" * 60)
    print("TB Breath Analyzer - Model Training")
    print("=" * 60)
    print()
    
    train_df = load_training_data()
    test_df = load_test_data()
    clf = train_model(train_df)
    accuracy = evaluate_model(clf, test_df)
    save_model(clf)
    
    print()
    print("=" * 60)
    log("Complete")
    print("=" * 60)
    print()
    print(f"Trained on: {len(train_df)} samples")
    print(f"Tested on: {len(test_df)} samples")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
