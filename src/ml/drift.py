"""
Distributional Drift & ML-Observability Engine.

Implements statistical hypothesis testing (Two-sample Kolmogorov-Smirnov test)
and Population Stability Index (PSI) to track data drift across incoming inference batches.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.responses import DriftReportResponse, FeatureDriftDetail
from src.domain.transaction import BatchTransactionInput, TransactionInput
from src.ml.generator import FEATURE_COLUMNS

logger = get_logger("sentinelml.drift")


class ModelDriftDetector:
    """Enterprise Statistical Drift Detector for Covariate Shift Monitoring."""

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None) -> None:
        self.baseline_df = baseline_df
        if self.baseline_df is None:
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Loads baseline reference dataset from disk."""
        baseline_path = settings.baseline_data_path
        if baseline_path.exists():
            try:
                self.baseline_df = pd.read_parquet(baseline_path)
                logger.info(
                    f"Loaded baseline reference dataset with {len(self.baseline_df)} records."
                )
            except Exception as e:
                logger.error(f"Error loading baseline parquet from {baseline_path}: {e}")
                self.baseline_df = None
        else:
            logger.warning(f"Baseline file does not exist at {baseline_path}.")

    @staticmethod
    def calculate_ks_test(
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Executes two-sample Kolmogorov-Smirnov test.
        
        Returns:
            Tuple of (ks_statistic, p_value)
        """
        res = stats.ks_2samp(baseline, current)
        return float(res.statistic), float(res.pvalue)

    @staticmethod
    def calculate_psi(
        baseline: np.ndarray,
        current: np.ndarray,
        num_buckets: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """
        Calculates Population Stability Index (PSI).
        
        PSI < 0.10: No significant distribution change.
        0.10 <= PSI < 0.20: Moderate distribution shift.
        PSI >= 0.20: Substantial distribution drift requiring attention.
        """
        # Ensure finite non-null values
        b_clean = baseline[np.isfinite(baseline)]
        c_clean = current[np.isfinite(current)]

        if len(b_clean) == 0 or len(c_clean) == 0:
            return 0.0

        # Create quantile bin edges based on baseline
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bin_edges = np.percentile(b_clean, percentiles)
        bin_edges = np.unique(bin_edges)  # handle non-unique edges

        if len(bin_edges) < 2:
            return 0.0

        # Adjust minimum and maximum edges to capture all current values
        bin_edges[0] = min(bin_edges[0], c_clean.min()) - 1e-5
        bin_edges[-1] = max(bin_edges[-1], c_clean.max()) + 1e-5

        # Compute empirical counts
        b_counts, _ = np.histogram(b_clean, bins=bin_edges)
        c_counts, _ = np.histogram(c_clean, bins=bin_edges)

        # Convert to relative proportions with epsilon smoothing
        b_prop = (b_counts + epsilon) / (len(b_clean) + epsilon * len(b_counts))
        c_prop = (c_counts + epsilon) / (len(c_clean) + epsilon * len(c_counts))

        # Compute PSI component summation
        psi_value = np.sum((c_prop - b_prop) * np.log(c_prop / b_prop))
        return float(np.clip(psi_value, 0.0, 10.0))

    def evaluate_batch(
        self,
        batch_data: Union[pd.DataFrame, BatchTransactionInput, List[Dict]],
    ) -> DriftReportResponse:
        """
        Evaluates a batch of transactions against baseline reference data.
        
        Returns:
            DriftReportResponse with statistical tests and drift verdicts.
        """
        if self.baseline_df is None:
            self._load_baseline()
            if self.baseline_df is None:
                raise RuntimeError("Baseline dataset is not available for drift detection.")

        # Convert input into DataFrame
        if isinstance(batch_data, pd.DataFrame):
            df_current = batch_data[FEATURE_COLUMNS].copy()
        elif isinstance(batch_data, BatchTransactionInput):
            rows = [t.to_feature_dict() for t in batch_data.transactions]
            df_current = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        elif isinstance(batch_data, list):
            df_current = pd.DataFrame(batch_data, columns=FEATURE_COLUMNS)
        else:
            raise ValueError(f"Unsupported batch_data type: {type(batch_data)}")

        feature_reports: List[FeatureDriftDetail] = []
        drifted_count = 0

        p_val_thresh = settings.drift_p_value_threshold
        psi_thresh = settings.drift_psi_threshold

        for col in FEATURE_COLUMNS:
            base_vals = self.baseline_df[col].to_numpy()
            curr_vals = df_current[col].to_numpy()

            ks_stat, p_val = self.calculate_ks_test(base_vals, curr_vals)
            psi = self.calculate_psi(base_vals, curr_vals)

            # Drift detected if KS reject null hypothesis (p < threshold) or PSI indicates shift
            is_drift = bool(p_val < p_val_thresh or psi >= psi_thresh)
            if is_drift:
                drifted_count += 1

            # Determine alert severity
            if psi >= 0.25 or (p_val < 0.001 and psi >= 0.15):
                severity = "CRITICAL_DRIFT"
            elif is_drift:
                severity = "MODERATE_DRIFT"
            else:
                severity = "NORMAL"

            feature_reports.append(
                FeatureDriftDetail(
                    feature=col,
                    ks_statistic=round(ks_stat, 4),
                    ks_p_value=round(p_val, 6),
                    psi=round(psi, 4),
                    is_drift_detected=is_drift,
                    alert_severity=severity,
                )
            )

        overall_drift = drifted_count > 0

        return DriftReportResponse(
            sample_size=len(df_current),
            baseline_sample_size=len(self.baseline_df),
            overall_drift_detected=overall_drift,
            drifted_features_count=drifted_count,
            feature_reports=feature_reports,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
