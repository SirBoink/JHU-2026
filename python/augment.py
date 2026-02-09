"""
TB Breath Analyzer - Data Augmentation
KDE-based synthetic data generation for tabular sensor data.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, SENSOR_DATA_NEW_FILE

RANDOM_STATE = 42
TEST_SIZE = 0.2
SYNTHETIC_SAMPLES_PER_CLASS = 2000
SENSOR_COLUMNS = ["MQ3_raw", "MQ135_raw", "MQ2_raw"]

TRAIN_AUGMENTED_FILE = DATA_DIR / "train_augmented.csv"
TEST_HIDDEN_FILE = DATA_DIR / "test_real_hidden.csv"


def log(msg: str):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    """Load sensor data from CSV."""
    log("Loading sensor data...")
    
    if not SENSOR_DATA_NEW_FILE.exists():
        print(f"Error: Data file not found: {SENSOR_DATA_NEW_FILE}")
        sys.exit(1)
    
    df = pd.read_csv(SENSOR_DATA_NEW_FILE)
    log(f"Loaded {len(df)} samples")
    
    log("Class distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    print()
    
    return df


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data with stratification (must happen before augmentation)."""
    log(f"Splitting data ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label']
    )
    
    log(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print()
    return train_df, test_df


def save_hidden_test_set(test_df: pd.DataFrame):
    """Save pristine test set."""
    log(f"Saving test set to {TEST_HIDDEN_FILE}...")
    test_df.to_csv(TEST_HIDDEN_FILE, index=False)
    log(f"Saved {len(test_df)} samples")
    print()


def fit_kde_per_class(train_df: pd.DataFrame) -> dict:
    """Fit KDE estimator for each class."""
    log("Fitting KDE models...")
    
    kde_models = {}
    for class_label in train_df['label'].unique():
        class_data = train_df[train_df['label'] == class_label][SENSOR_COLUMNS]
        kde = gaussian_kde(class_data.values.T, bw_method='scott')
        kde_models[class_label] = kde
        log(f"  {class_label}: {len(class_data)} samples")
    
    print()
    return kde_models


def generate_synthetic_samples(kde_models: dict, n_samples: int) -> pd.DataFrame:
    """Generate synthetic samples from KDE distributions."""
    log(f"Generating {n_samples} samples per class...")
    
    synthetic_data = []
    for class_label, kde in kde_models.items():
        samples = kde.resample(n_samples, seed=RANDOM_STATE).T
        samples = np.maximum(samples, 0)
        samples = np.round(samples).astype(int)
        
        class_df = pd.DataFrame(samples, columns=SENSOR_COLUMNS)
        class_df['label'] = class_label
        class_df['is_synthetic'] = 1
        synthetic_data.append(class_df)
        log(f"  {class_label}: {n_samples} generated")
    
    print()
    return pd.concat(synthetic_data, ignore_index=True)


def combine_and_save(train_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Combine real and synthetic data, then save."""
    log("Combining datasets...")
    
    train_real = train_df.copy()
    train_real['is_synthetic'] = 0
    
    combined_df = pd.concat([train_real, synthetic_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    log(f"Total: {len(combined_df)} ({len(train_real)} real + {len(synthetic_df)} synthetic)")
    print()
    
    log("Final distribution:")
    for label, count in combined_df['label'].value_counts().items():
        real = len(combined_df[(combined_df['label'] == label) & (combined_df['is_synthetic'] == 0)])
        synth = len(combined_df[(combined_df['label'] == label) & (combined_df['is_synthetic'] == 1)])
        print(f"  {label}: {count} ({real} real + {synth} synthetic)")
    print()
    
    log(f"Saving to {TRAIN_AUGMENTED_FILE}...")
    combined_df.to_csv(TRAIN_AUGMENTED_FILE, index=False)
    log(f"Saved {len(combined_df)} samples")


def validate_output():
    """Validate generated files."""
    log("Validating output...")
    
    test_df = pd.read_csv(TEST_HIDDEN_FILE)
    assert 'is_synthetic' not in test_df.columns
    
    train_df = pd.read_csv(TRAIN_AUGMENTED_FILE)
    assert 'is_synthetic' in train_df.columns
    
    for col in SENSOR_COLUMNS:
        assert train_df[col].min() >= 0
    
    log(f"Test set: {len(test_df)}, Train set: {len(train_df)}")
    log("Validation passed")
    print()


def main():
    print("=" * 60)
    print("TB Breath Analyzer - KDE Data Augmentation")
    print("=" * 60)
    print()
    
    df = load_data()
    train_df, test_df = stratified_split(df)
    save_hidden_test_set(test_df)
    kde_models = fit_kde_per_class(train_df)
    synthetic_df = generate_synthetic_samples(kde_models, SYNTHETIC_SAMPLES_PER_CLASS)
    combine_and_save(train_df, synthetic_df)
    validate_output()
    
    print("=" * 60)
    log("Augmentation complete")
    print("=" * 60)
    print()
    print(f"Output: {TEST_HIDDEN_FILE}")
    print(f"Output: {TRAIN_AUGMENTED_FILE}")


if __name__ == "__main__":
    main()
