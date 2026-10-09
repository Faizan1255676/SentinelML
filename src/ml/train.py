"""
Enterprise Model Training Pipeline.

Orchestrates synthetic dataset generation, feature scaling,
LightGBM training with imbalanced cost-sensitive learning,
rigorous metric evaluation, and joblib artifact serialization.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import settings
from src.core.logger import get_logger
from src.ml.generator import FEATURE_COLUMNS, SyntheticDataGenerator

logger = get_logger("sentinelml.train")


def build_pipeline() -> Pipeline:
    """Builds an enterprise Scikit-Learn pipeline with preprocessing and LightGBM."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="passthrough",
    )

    classifier = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        class_weight="balanced",  # Robust class imbalance handling
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_evaluate() -> Pipeline:
    """Runs data generation, pipeline fitting, evaluation, and asset persistence."""
    logger.info("Initializing synthetic enterprise transaction generation...")
    generator = SyntheticDataGenerator(seed=42)
    X, y = generator.generate_baseline_dataset(n_samples=10000, fraud_ratio=0.035)

    fraud_count = int(y.sum())
    total_count = len(y)
    logger.info(
        f"Generated {total_count} transactions. Fraud cases: {fraud_count} "
        f"({(fraud_count / total_count) * 100:.2f}% prevalence)."
    )

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info("Fitting Scikit-Learn + LightGBM pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions & probabilities
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= settings.fraud_probability_threshold).astype(int)

    # Metrics computation
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("================ MODEL EVALUATION METRICS ================")
    logger.info(f"ROC-AUC Score          : {roc_auc:.4f}")
    logger.info(f"PR-AUC (Avg Precision) : {pr_auc:.4f}")
    logger.info(f"Precision              : {precision:.4f}")
    logger.info(f"Recall                 : {recall:.4f}")
    logger.info(f"F1-Score               : {f1:.4f}")
    logger.info("==========================================================")

    # Ensure model output directory exists
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    # Persist serialized model pipeline
    model_path = settings.model_file_path
    joblib.dump(pipeline, model_path)
    logger.info(f"Persisted trained pipeline to {model_path}")

    # Persist baseline features for drift detection
    baseline_path = settings.baseline_data_path
    X_train.to_parquet(baseline_path, index=False)
    logger.info(f"Persisted baseline reference data to {baseline_path}")

    return pipeline


if __name__ == "__main__":
    train_and_evaluate()
