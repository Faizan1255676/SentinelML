"""
Real-Time Inference & Explainability Core.

Thread-safe Singleton inference engine featuring sub-20ms SLA latency,
TreeExplainer SHAP integration, and calibrated risk level classification.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import shap

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.responses import (
    ExplanationResponse,
    FeatureContribution,
    PredictionResponse,
    RiskLevel,
)
from src.domain.transaction import TransactionInput
from src.ml.generator import FEATURE_COLUMNS

logger = get_logger("sentinelml.inference")


class FraudInferenceEngine:
    """Production ML Inference & SHAP Explainability Engine."""

    _instance: Optional["FraudInferenceEngine"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.pipeline = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.is_loaded = False
        self._load_pipeline()

    @classmethod
    def get_instance(cls) -> "FraudInferenceEngine":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_pipeline(self) -> None:
        """Loads serialized pipeline and initializes TreeExplainer."""
        model_path = settings.model_file_path
        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}. Train the model first.")
            return

        try:
            logger.info(f"Loading pipeline from {model_path}...")
            self.pipeline = joblib.load(model_path)
            
            # Extract classifier and preprocessor for SHAP TreeExplainer
            classifier = self.pipeline.named_steps.get("classifier")
            if classifier is not None:
                # Initialize TreeExplainer for instantaneous local attributions
                self.explainer = shap.TreeExplainer(classifier)
                logger.info("Initialized SHAP TreeExplainer successfully.")

            self.is_loaded = True
            logger.info("FraudInferenceEngine initialized and ready for production serving.")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}", exc_info=True)
            raise e

    def _prepare_dataframe(
        self,
        transaction_input: Union[TransactionInput, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Converts domain model or dictionary to correctly ordered DataFrame."""
        if isinstance(transaction_input, TransactionInput):
            data = transaction_input.to_feature_dict()
        elif isinstance(transaction_input, dict):
            data = {k: transaction_input[k] for k in FEATURE_COLUMNS if k in transaction_input}
        else:
            raise ValueError(f"Unsupported transaction input type: {type(transaction_input)}")

        return pd.DataFrame([data], columns=FEATURE_COLUMNS)

    def predict(
        self,
        transaction: Union[TransactionInput, Dict[str, Any]],
        transaction_id: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Executes real-time risk scoring meeting sub-20ms SLA.
        
        Args:
            transaction: Input feature payload.
            transaction_id: Optional tracking identifier.
            
        Returns:
            PredictionResponse object with probabilities, decision, and latency.
        """
        start_time = time.perf_counter()

        if not self.is_loaded or self.pipeline is None:
            self._load_pipeline()
            if not self.is_loaded:
                raise RuntimeError("Inference pipeline is not loaded.")

        tx_id = transaction_id
        if tx_id is None:
            if isinstance(transaction, TransactionInput):
                tx_id = transaction.transaction_id
            elif isinstance(transaction, dict) and "transaction_id" in transaction:
                tx_id = str(transaction["transaction_id"])
            else:
                tx_id = "tx-local"

        df_input = self._prepare_dataframe(transaction)

        # High-speed pipeline scoring
        probabilities = self.pipeline.predict_proba(df_input)[0]
        risk_score = float(probabilities[1])

        threshold = settings.fraud_probability_threshold
        is_fraud = risk_score >= threshold

        # Assign risk tier
        if risk_score >= settings.high_risk_threshold:
            risk_level = RiskLevel.CRITICAL if risk_score >= 0.90 else RiskLevel.HIGH
        elif risk_score >= settings.medium_risk_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return PredictionResponse(
            transaction_id=tx_id,
            risk_score=round(risk_score, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            threshold_applied=threshold,
            latency_ms=latency_ms,
        )

    def explain(
        self,
        transaction: Union[TransactionInput, Dict[str, Any]],
        transaction_id: Optional[str] = None,
    ) -> ExplanationResponse:
        """
        Extracts local SHAP feature attributions and ranks primary risk drivers.
        
        Args:
            transaction: Input transaction data.
            transaction_id: Optional tracking identifier.
            
        Returns:
            ExplanationResponse containing feature attributions and base value.
        """
        start_time = time.perf_counter()

        # Run inference first
        pred = self.predict(transaction, transaction_id=transaction_id)
        df_input = self._prepare_dataframe(transaction)

        if self.explainer is None:
            raise RuntimeError("SHAP TreeExplainer is not initialized.")

        # Transform inputs using pipeline preprocessor
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_scaled = preprocessor.transform(df_input)

        # Compute SHAP values
        raw_shap_values = self.explainer.shap_values(X_scaled)

        # Handle binary classification shapes from LightGBM / TreeExplainer
        if isinstance(raw_shap_values, list) and len(raw_shap_values) > 1:
            # Positive class (fraud) SHAP values
            shap_row = raw_shap_values[1][0]
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            shap_row = raw_shap_values[0, :, 1]
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2:
            shap_row = raw_shap_values[0]
        else:
            shap_row = np.array(raw_shap_values).flatten()

        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
        else:
            base_value = float(expected_val)

        # Map to feature names
        shap_dict: Dict[str, float] = {}
        contributions: List[FeatureContribution] = []

        for col, s_val in zip(FEATURE_COLUMNS, shap_row):
            val_rounded = round(float(s_val), 4)
            shap_dict[col] = val_rounded
            actual_val = float(df_input[col].iloc[0])
            impact = "INCREASES_RISK" if val_rounded > 0 else "DECREASES_RISK"

            contributions.append(
                FeatureContribution(
                    feature=col,
                    value=actual_val,
                    shap_value=val_rounded,
                    impact_direction=impact,
                )
            )

        # Sort contributions descending by SHAP value (highest risk drivers first)
        contributions.sort(key=lambda x: x.shap_value, reverse=True)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return ExplanationResponse(
            transaction_id=pred.transaction_id,
            risk_score=pred.risk_score,
            base_value=round(base_value, 4),
            shap_values=shap_dict,
            top_risk_factors=contributions,
            latency_ms=latency_ms,
        )
