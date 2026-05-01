"""
EnviroScan - Pollution Source Classification Pipeline

Description:
- Data loading & cleaning
- Feature engineering
- Noise injection
- Hyperparameter tuning
- Model training (RF, DT, XGB)
- Evaluation
- Feature importance
- Model saving
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Import project configuration
from config import (
    FINAL_DATASET_LABELED, MODELS_DIR, ALL_FEATURES, TARGET_COLUMN,
    RANDOM_STATE, TEST_SIZE, NOISE_LEVEL, RF_PARAM_GRID, DT_PARAM_GRID, XGB_PARAM_DIST,
    RANDOM_FOREST_MODEL, DECISION_TREE_MODEL, XGBOOST_MODEL, LABEL_ENCODER
)
from utils import load_dataset, remove_missing_values, add_noise_to_features, save_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# ============================================
# Step 1: Load Dataset
# ============================================

logger.info("Loading Dataset...")
try:
    df = load_dataset(FINAL_DATASET_LABELED, required_columns=ALL_FEATURES + [TARGET_COLUMN])
    logger.info(f"Dataset Shape: {df.shape}")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    sys.exit(1)

# ============================================
# Step 2: Data Cleaning
# ============================================

logger.info("Checking Missing Values...")
missing_count = df.isnull().sum().sum()
logger.info(f"Total missing values: {missing_count}")

df = remove_missing_values(df, strategy='drop')
logger.info("Missing values removed")

# ============================================
# Step 3: Feature Selection
# ============================================

X = df[ALL_FEATURES].copy()
y = df[TARGET_COLUMN]

logger.info(f"Features: {len(ALL_FEATURES)}")
logger.info(f"Samples: {len(X)}")

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

logger.info("Class Mapping:")
for idx, class_name in enumerate(le.classes_):
    logger.info(f"  {class_name} --> {idx}")

# ============================================
# Step 4: Add Controlled Noise
# ============================================

logger.info("Adding Noise to Features...")
X = add_noise_to_features(X, noise_level=NOISE_LEVEL)
logger.info("Noise added successfully")

# ============================================
# Step 5: Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

logger.info(f"Train Size: {X_train.shape}")
logger.info(f"Test Size: {X_test.shape}")

# ============================================
# Step 6: Random Forest with Hyperparameter Tuning
# ============================================

logger.info("Training Random Forest (GridSearch)...")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    RF_PARAM_GRID,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_

logger.info(f"Best RF Parameters: {rf_grid.best_params_}")

# ============================================
# Evaluation Function
# ============================================

def evaluate_model(model, model_name):
    """Evaluate model performance and generate confusion matrix"""
    logger.info(f"Evaluating {model_name}...")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    logger.info(f"Training Accuracy: {train_acc:.4f}")
    logger.info(f"Testing Accuracy: {test_acc:.4f}")
    
    logger.info("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = IMAGES_DIR / f"{model_name.replace(' ', '_')}_Confusion_Matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {cm_path}")
    plt.show()
    
    return train_acc, test_acc

# Evaluate Random Forest
rf_train_acc, rf_test_acc = evaluate_model(rf_model, "Random Forest")

# ============================================
# Feature Importance (Random Forest)
# ============================================

logger.info("Analyzing Feature Importance...")
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()

# Save feature importance plot
fi_path = IMAGES_DIR / "Feature_Importance.png"
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
logger.info(f"Feature importance plot saved to {fi_path}")
plt.show()

# ============================================
# Step 7: Decision Tree
# ============================================

logger.info("Training Decision Tree...")

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    DT_PARAM_GRID,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_

logger.info(f"Best DT Parameters: {dt_grid.best_params_}")
dt_train_acc, dt_test_acc = evaluate_model(dt_model, "Decision Tree")

# ============================================
# Step 8: XGBoost
# ============================================

logger.info("Training XGBoost...")

xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric='mlogloss'
)

random_search = RandomizedSearchCV(
    xgb,
    param_distributions=XGB_PARAM_DIST,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

random_search.fit(X_train, y_train)
xgb_model = random_search.best_estimator_

logger.info(f"Best XGBoost Parameters: {random_search.best_params_}")
xgb_train_acc, xgb_test_acc = evaluate_model(xgb_model, "XGBoost")

# ============================================
# Step 9: Save Models
# ============================================

logger.info("Saving Models...")

# Save with timestamp (versioned)
joblib.dump(rf_model, MODELS_DIR / f"RandomForest_{timestamp}.joblib")
joblib.dump(dt_model, MODELS_DIR / f"DecisionTree_{timestamp}.joblib")
joblib.dump(xgb_model, MODELS_DIR / f"XGBoost_{timestamp}.joblib")
joblib.dump(le, MODELS_DIR / f"LabelEncoder_{timestamp}.joblib")

# Save as latest (for production use)
joblib.dump(rf_model, RANDOM_FOREST_MODEL)
joblib.dump(dt_model, DECISION_TREE_MODEL)
joblib.dump(xgb_model, XGBOOST_MODEL)
joblib.dump(le, LABEL_ENCODER)

logger.info("All models saved successfully")

# ============================================
# Step 10: Model Comparison Summary
# ============================================

logger.info("\n" + "="*50)
logger.info("MODEL COMPARISON SUMMARY")
logger.info("="*50)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree', 'XGBoost'],
    'Train Accuracy': [rf_train_acc, dt_train_acc, xgb_train_acc],
    'Test Accuracy': [rf_test_acc, dt_test_acc, xgb_test_acc],
    'Overfitting': [
        rf_train_acc - rf_test_acc,
        dt_train_acc - dt_test_acc,
        xgb_train_acc - xgb_test_acc
    ]
})

print("\n" + comparison_df.to_string(index=False))

best_model_idx = comparison_df['Test Accuracy'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
logger.info(f"\n🏆 Best Model: {best_model_name}")
logger.info(f"✅ Training completed successfully at {timestamp}")
logger.info("="*50)
