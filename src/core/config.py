"""
Application Configuration Module.

Leverages Pydantic v2 Settings for strict environment variable parsing,
type validation, and centralized defaults following 12-factor application design.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Core settings and runtime configuration for SentinelML."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    app_name: str = "SentinelML"
    app_env: Literal["development", "staging", "production"] = "production"
    debug: bool = False
    log_level: str = "INFO"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    model_dir: Path = base_dir / "models"
    model_file_path: Path = model_dir / "fraud_pipeline.joblib"
    baseline_data_path: Path = model_dir / "baseline_data.parquet"

    # ML Inference & Observability Thresholds
    fraud_probability_threshold: float = 0.50
    high_risk_threshold: float = 0.75
    medium_risk_threshold: float = 0.40
    drift_p_value_threshold: float = 0.05
    drift_psi_threshold: float = 0.20

    # Latency SLAs
    target_latency_ms: float = 20.0


# Global cached config instance
settings = AppConfig()
