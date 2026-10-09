"""
FastAPI Route Handlers.

Exposes RESTful endpoints for real-time risk scoring, SHAP explainability,
statistical drift analysis, and health monitoring.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.responses import (
    DriftReportResponse,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
)
from src.domain.transaction import BatchTransactionInput, TransactionInput
from src.ml.drift import ModelDriftDetector
from src.ml.inference import FraudInferenceEngine

logger = get_logger("sentinelml.api.routes")
router = APIRouter()

# Record service launch timestamp
SERVICE_START_TIME = time.time()


def get_inference_engine() -> FraudInferenceEngine:
    """Dependency injection providing the singleton inference engine."""
    engine = FraudInferenceEngine.get_instance()
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is currently unavailable. Ensure training has completed.",
        )
    return engine


def get_drift_detector() -> ModelDriftDetector:
    """Dependency injection providing the dataset drift detector."""
    detector = ModelDriftDetector()
    if detector.baseline_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Baseline dataset is missing. Run model training to establish baseline.",
        )
    return detector


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check & System Status",
    tags=["Observability"],
)
async def health_check() -> HealthResponse:
    """Returns microservice readiness, loaded pipeline assets, and uptime metrics."""
    engine = FraudInferenceEngine.get_instance()
    detector = ModelDriftDetector()

    uptime = time.time() - SERVICE_START_TIME

    return HealthResponse(
        status="HEALTHY" if engine.is_loaded else "DEGRADED",
        app_name=settings.app_name,
        version="1.0.0",
        model_loaded=engine.is_loaded,
        model_version="lgbm-v1.0",
        baseline_loaded=detector.baseline_df is not None,
        uptime_seconds=round(uptime, 2),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-Time Transaction Risk Scoring",
    tags=["Inference"],
)
async def predict_transaction(
    payload: TransactionInput,
    engine: FraudInferenceEngine = Depends(get_inference_engine),
) -> PredictionResponse:
    """
    Evaluates risk probability for an incoming transaction in sub-20ms.
    Returns probability, categorical risk tier, and binary fraud flag.
    """
    try:
        response = engine.predict(payload, transaction_id=payload.transaction_id)
        return response
    except Exception as exc:
        logger.error(f"Prediction failed for transaction {payload.transaction_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(exc)}",
        )


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Local Explainability via SHAP Values",
    tags=["Explainability"],
)
async def explain_transaction(
    payload: TransactionInput,
    engine: FraudInferenceEngine = Depends(get_inference_engine),
) -> ExplanationResponse:
    """
    Generates local SHAP attributions explaining why the transaction was scored
    as legitimate or fraudulent, ranking top risk contributors.
    """
    try:
        response = engine.explain(payload, transaction_id=payload.transaction_id)
        return response
    except Exception as exc:
        logger.error(f"SHAP explanation failed for transaction {payload.transaction_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability computation error: {str(exc)}",
        )


@router.post(
    "/drift-check",
    response_model=DriftReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Dataset Distribution Drift Analysis",
    tags=["Observability"],
)
async def check_data_drift(
    batch: BatchTransactionInput,
    detector: ModelDriftDetector = Depends(get_drift_detector),
) -> DriftReportResponse:
    """
    Compares incoming batch feature distributions against baseline reference data
    using the Kolmogorov-Smirnov test and Population Stability Index (PSI).
    """
    try:
        report = detector.evaluate_batch(batch)
        return report
    except Exception as exc:
        logger.error(f"Drift analysis failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift calculation error: {str(exc)}",
        )
