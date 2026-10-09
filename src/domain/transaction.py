"""
Transaction Domain Schemas.

Defines Pydantic v2 data models for financial transactions with strict validation,
type constraints, and documentation examples.
"""

from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Represents a single financial transaction submitted for risk evaluation."""

    transaction_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the transaction (UUID v4).",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    transaction_amount: float = Field(
        ...,
        gt=0.0,
        description="Transaction monetary amount (must be positive).",
        examples=[149.99],
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour when the transaction occurred (0 to 23).",
        examples=[14],
    )
    distance_from_home: float = Field(
        ...,
        ge=0.0,
        description="Distance from customer home in kilometers.",
        examples=[4.2],
    )
    user_age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Age of the account holder (18 to 120).",
        examples=[34],
    )
    velocity_60m: int = Field(
        ...,
        ge=0,
        description="Number of transactions initiated by user in the past 60 minutes.",
        examples=[2],
    )
    device_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Device telemetry fraud propensity score between 0.0 and 1.0.",
        examples=[0.12],
    )

    @field_validator("transaction_amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("transaction_amount must be strictly greater than 0.")
        return round(v, 2)

    def to_feature_dict(self) -> dict:
        """Converts model to dictionary containing raw feature vector."""
        return {
            "transaction_amount": self.transaction_amount,
            "hour_of_day": self.hour_of_day,
            "distance_from_home": self.distance_from_home,
            "user_age": self.user_age,
            "velocity_60m": self.velocity_60m,
            "device_risk_score": self.device_risk_score,
        }


class BatchTransactionInput(BaseModel):
    """Payload for batch scoring or drift detection analysis."""

    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        description="List of transaction items to process.",
    )
