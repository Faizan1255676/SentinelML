"""
Synthetic Financial Dataset Generator.

Generates realistic enterprise financial transaction records featuring realistic
statistical distributions, real-world class imbalance (~2-5% fraud rate),
and realistic syndicate fraud patterns.
"""

from typing import Tuple
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "transaction_amount",
    "hour_of_day",
    "distance_from_home",
    "user_age",
    "velocity_60m",
    "device_risk_score",
]


class SyntheticDataGenerator:
    """Enterprise-grade synthetic financial transaction generator."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_baseline_dataset(
        self,
        n_samples: int = 10000,
        fraud_ratio: float = 0.035,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates realistic baseline dataset with legitimate and fraudulent transactions.
        
        Args:
            n_samples: Total number of rows to generate.
            fraud_ratio: Target fraud prevalence (~3.5% default).
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        # --- Legitimate Transactions ---
        legit_amounts = self.rng.lognormal(mean=3.8, sigma=0.8, size=n_legit)  # median ~$45
        legit_hours = self.rng.choice(
            np.arange(24),
            size=n_legit,
            p=self._get_diurnal_probabilities(),
        )
        legit_distance = self.rng.exponential(scale=6.5, size=n_legit)  # mostly < 20km
        legit_age = self.rng.normal(loc=41, scale=12, size=n_legit).clip(18, 90).astype(int)
        legit_velocity = self.rng.poisson(lam=0.8, size=n_legit).clip(0, 4)
        legit_device = self.rng.beta(a=1.5, b=8.0, size=n_legit)  # strongly skewed towards 0.0-0.2

        df_legit = pd.DataFrame({
            "transaction_amount": legit_amounts,
            "hour_of_day": legit_hours,
            "distance_from_home": legit_distance,
            "user_age": legit_age,
            "velocity_60m": legit_velocity,
            "device_risk_score": legit_device,
            "is_fraud": 0,
        })

        # --- Fraudulent Transactions ---
        # Spikes in high amounts, night hours, high distance, high velocity, high device risk
        fraud_amounts = self.rng.lognormal(mean=6.2, sigma=1.1, size=n_fraud)  # median ~$500
        fraud_hours = self.rng.choice(
            np.arange(24),
            size=n_fraud,
            p=self._get_fraud_hour_probabilities(),
        )
        fraud_distance = self.rng.exponential(scale=45.0, size=n_fraud) + 20.0  # long distance
        fraud_age = self.rng.normal(loc=35, scale=14, size=n_fraud).clip(18, 90).astype(int)
        fraud_velocity = self.rng.poisson(lam=5.5, size=n_fraud).clip(1, 15)
        fraud_device = self.rng.beta(a=6.0, b=2.0, size=n_fraud)  # skewed towards 0.7-1.0

        df_fraud = pd.DataFrame({
            "transaction_amount": fraud_amounts,
            "hour_of_day": fraud_hours,
            "distance_from_home": fraud_distance,
            "user_age": fraud_age,
            "velocity_60m": fraud_velocity,
            "device_risk_score": fraud_device,
            "is_fraud": 1,
        })

        # Combine and shuffle
        df = pd.concat([df_legit, df_fraud], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Rounding and formatting
        df["transaction_amount"] = df["transaction_amount"].round(2)
        df["distance_from_home"] = df["distance_from_home"].round(2)
        df["device_risk_score"] = df["device_risk_score"].round(4)

        X = df[FEATURE_COLUMNS]
        y = df["is_fraud"]
        return X, y

    def generate_drifted_batch(
        self,
        n_samples: int = 500,
        drift_type: str = "velocity_and_device",
    ) -> pd.DataFrame:
        """
        Generates an inference batch exhibiting distributional covariate shift / drift.
        """
        # Baseline normal generation
        X, _ = self.generate_baseline_dataset(n_samples=n_samples, fraud_ratio=0.03)

        df_drift = X.copy()
        if drift_type == "velocity_and_device":
            # Simulate coordinated credential stuffing / bot attack
            df_drift["velocity_60m"] = self.rng.poisson(lam=4.2, size=n_samples).clip(0, 15)
            df_drift["device_risk_score"] = self.rng.beta(a=5.0, b=3.0, size=n_samples).round(4)
        elif drift_type == "amount_inflation":
            # Extreme economic or black swan purchase spikes
            df_drift["transaction_amount"] = (df_drift["transaction_amount"] * 3.5).round(2)
        elif drift_type == "night_owl":
            # Shifts in hours of activity
            df_drift["hour_of_day"] = self.rng.choice(np.arange(0, 6), size=n_samples)

        return df_drift

    @staticmethod
    def _get_diurnal_probabilities() -> np.ndarray:
        """Diurnal distribution with peak daytime transactions (hours 9-20)."""
        probs = np.array([
            0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # 00:00 - 05:00
            0.02, 0.04, 0.06, 0.08, 0.08, 0.08,      # 06:00 - 11:00
            0.08, 0.07, 0.07, 0.07, 0.07, 0.08,      # 12:00 - 17:00
            0.07, 0.06, 0.05, 0.03, 0.02, 0.015     # 18:00 - 23:00
        ])
        return probs / probs.sum()

    @staticmethod
    def _get_fraud_hour_probabilities() -> np.ndarray:
        """Fraud probability skewed towards midnight and early morning hours."""
        probs = np.array([
            0.08, 0.09, 0.10, 0.11, 0.09, 0.07,      # 00:00 - 05:00 (peak fraud)
            0.03, 0.02, 0.02, 0.03, 0.03, 0.03,      # 06:00 - 11:00
            0.03, 0.03, 0.03, 0.03, 0.03, 0.04,      # 12:00 - 17:00
            0.04, 0.05, 0.06, 0.06, 0.07, 0.07       # 18:00 - 23:00
        ])
        return probs / probs.sum()
