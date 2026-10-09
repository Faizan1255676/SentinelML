"""
Machine Learning Core Unit & Integration Tests.

Validates synthetic generation statistical properties, pipeline inference,
SHAP TreeExplainer attributions, and Kolmogorov-Smirnov / PSI drift calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.config import settings
from src.domain.responses import RiskLevel
from src.ml.drift import ModelDriftDetector
from src.ml.generator import FEATURE_COLUMNS, SyntheticDataGenerator
from src.ml.inference import FraudInferenceEngine


def test_synthetic_data_generator_properties():
    """Validates properties of synthetic data generator."""
    generator = SyntheticDataGenerator(seed=99)
    X, y = generator.generate_baseline_dataset(n_samples=1000, fraud_ratio=0.04)

    assert len(X) == 1000
    assert len(y) == 1000
    assert list(X.columns) == FEATURE_COLUMNS
    assert 30 <= int(y.sum()) <= 50  # roughly 4% prevalence
    assert (X["transaction_amount"] > 0).all()
    assert (X["hour_of_day"].between(0, 23)).all()
    assert (X["device_risk_score"].between(0.0, 1.0)).all()


def test_inference_engine_singleton():
    """Verifies that FraudInferenceEngine implements a thread-safe singleton."""
    engine1 = FraudInferenceEngine.get_instance()
    engine2 = FraudInferenceEngine.get_instance()
    assert engine1 is engine2
    assert engine1.is_loaded is True


def test_inference_scoring_legitimate(sample_legit_transaction):
    """Verifies low fraud score for normal customer behavior."""
    engine = FraudInferenceEngine.get_instance()
    response = engine.predict(sample_legit_transaction)

    assert response.is_fraud is False
    assert response.risk_score < settings.fraud_probability_threshold
    assert response.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    assert response.latency_ms > 0.0


def test_inference_scoring_fraudulent(sample_fraud_transaction):
    """Verifies critical fraud score for suspicious attack patterns."""
    engine = FraudInferenceEngine.get_instance()
    response = engine.predict(sample_fraud_transaction)

    assert response.is_fraud is True
    assert response.risk_score >= settings.fraud_probability_threshold
    assert response.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]


def test_inference_sub_20ms_sla(sample_legit_transaction):
    """Benchmarks that inference comfortably meets the 20ms SLA."""
    engine = FraudInferenceEngine.get_instance()
    # Warmup
    engine.predict(sample_legit_transaction)

    # Measured run
    response = engine.predict(sample_legit_transaction)
    assert response.latency_ms < settings.target_latency_ms


def test_shap_explainability_consistency(sample_fraud_transaction):
    """Verifies that SHAP TreeExplainer outputs match feature dimensionality."""
    engine = FraudInferenceEngine.get_instance()
    explanation = engine.explain(sample_fraud_transaction)

    assert len(explanation.shap_values) == len(FEATURE_COLUMNS)
    assert len(explanation.top_risk_factors) == len(FEATURE_COLUMNS)

    # Top risk factors must be sorted descending by SHAP value
    shap_vals = [f.shap_value for f in explanation.top_risk_factors]
    assert shap_vals == sorted(shap_vals, reverse=True)

    # Fraudulent transaction should have positive SHAP for velocity or device risk
    top_feature = explanation.top_risk_factors[0]
    assert top_feature.impact_direction == "INCREASES_RISK"


def test_drift_detection_clean_baseline():
    """Verifies that a sample drawn from same distribution does not trigger false drift."""
    detector = ModelDriftDetector()
    assert detector.baseline_df is not None

    # Sample from the baseline itself
    clean_sample = detector.baseline_df.sample(n=300, random_state=42)
    report = detector.evaluate_batch(clean_sample)

    assert report.overall_drift_detected is False
    assert report.drifted_features_count == 0


def test_drift_detection_drifted_batch():
    """Verifies that anomalous distribution shift is flagged accurately."""
    detector = ModelDriftDetector()
    generator = SyntheticDataGenerator(seed=101)

    drifted_batch = generator.generate_drifted_batch(
        n_samples=400, drift_type="velocity_and_device"
    )
    report = detector.evaluate_batch(drifted_batch)

    assert report.overall_drift_detected is True
    assert report.drifted_features_count > 0

    # Specifically check velocity_60m or device_risk_score
    feature_map = {r.feature: r for r in report.feature_reports}
    assert feature_map["velocity_60m"].is_drift_detected is True
    assert feature_map["velocity_60m"].psi > 0.10
