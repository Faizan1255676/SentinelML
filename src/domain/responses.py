"""
API Response Domain Models.

Defines serialized outputs for prediction, SHAP explainability,
data drift observability, and service health checks.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Categorical risk assessment level based on model probability."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FeatureContribution(BaseModel):
    """Local attribution of an individual feature to the risk score."""
    feature: str = Field(..., description="Feature name.")
    value: float = Field(..., description="Observed feature value.")
    shap_value: float = Field(..., description="SHAP contribution value.")
    impact_direction: str = Field(..., description="'INCREASES_RISK' or 'DECREASES_RISK'.")


class PredictionResponse(BaseModel):
    """Standard payload returned by /predict endpoint."""
    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Model fraud probability.")
    is_fraud: bool = Field(..., description="Binary decision based on operational threshold.")
    risk_level: RiskLevel = Field(..., description="Categorized risk bracket.")
    threshold_applied: float = Field(..., description="Probability threshold applied for decision.")
    latency_ms: float = Field(..., description="Model inference latency in milliseconds.")


class ExplanationResponse(BaseModel):
    """Detailed explainability payload returned by /explain endpoint."""
    transaction_id: str
    risk_score: float
    base_value: float = Field(..., description="Expected baseline model output log-odds/probability.")
    shap_values: Dict[str, float] = Field(..., description="Raw SHAP attribution per feature.")
    top_risk_factors: List[FeatureContribution] = Field(
        ...,
        description="Features ranked by positive contribution to fraud risk.",
    )
    latency_ms: float


class FeatureDriftDetail(BaseModel):
    """Statistical summary for a single feature under drift analysis."""
    feature: str
    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov test statistic D.")
    ks_p_value: float = Field(..., description="Two-sample KS test p-value.")
    psi: float = Field(..., description="Population Stability Index value.")
    is_drift_detected: bool = Field(..., description="Flag indicating statistically significant drift.")
    alert_severity: str = Field(..., description="'NORMAL', 'MODERATE_DRIFT', or 'CRITICAL_DRIFT'.")


class DriftReportResponse(BaseModel):
    """Payload returned by /drift-check endpoint."""
    sample_size: int
    baseline_sample_size: int
    overall_drift_detected: bool
    drifted_features_count: int
    feature_reports: List[FeatureDriftDetail]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check payload returned by /health endpoint."""
    status: str
    app_name: str
    version: str
    model_loaded: bool
    model_version: Optional[str] = None
    baseline_loaded: bool
    uptime_seconds: float
