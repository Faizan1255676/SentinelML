"""
Domain Layer Unit Tests.

Validates Pydantic v2 domain schemas, constraints, value ranges,
and error handling for corrupted inputs.
"""

import pytest
from pydantic import ValidationError

from src.domain.transaction import BatchTransactionInput, TransactionInput


def test_valid_transaction_creation(sample_legit_transaction_dict):
    """Verifies that legitimate transaction passes validation."""
    tx = TransactionInput(**sample_legit_transaction_dict)
    assert tx.transaction_amount == 45.50
    assert tx.hour_of_day == 14
    assert tx.distance_from_home == 3.2
    assert tx.user_age == 38
    assert tx.velocity_60m == 1
    assert tx.device_risk_score == 0.08
    assert len(tx.transaction_id) > 0


def test_invalid_negative_amount(sample_legit_transaction_dict):
    """Verifies that non-positive amount raises ValidationError."""
    corrupted = sample_legit_transaction_dict.copy()
    corrupted["transaction_amount"] = -10.0
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)


def test_invalid_zero_amount(sample_legit_transaction_dict):
    """Verifies that zero amount raises ValidationError."""
    corrupted = sample_legit_transaction_dict.copy()
    corrupted["transaction_amount"] = 0.0
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)


def test_invalid_hour_boundary(sample_legit_transaction_dict):
    """Verifies that hour outside 0-23 raises ValidationError."""
    corrupted = sample_legit_transaction_dict.copy()
    corrupted["hour_of_day"] = 25
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)

    corrupted["hour_of_day"] = -1
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)


def test_invalid_user_age(sample_legit_transaction_dict):
    """Verifies that minor or extreme ages fail validation."""
    corrupted = sample_legit_transaction_dict.copy()
    corrupted["user_age"] = 16  # below 18
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)

    corrupted["user_age"] = 150  # above 120
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)


def test_invalid_device_risk_score(sample_legit_transaction_dict):
    """Verifies device risk score outside [0.0, 1.0] raises ValidationError."""
    corrupted = sample_legit_transaction_dict.copy()
    corrupted["device_risk_score"] = 1.2
    with pytest.raises(ValidationError):
        TransactionInput(**corrupted)


def test_to_feature_dict(sample_legit_transaction):
    """Verifies conversion of model to raw feature dictionary."""
    features = sample_legit_transaction.to_feature_dict()
    assert isinstance(features, dict)
    assert set(features.keys()) == {
        "transaction_amount",
        "hour_of_day",
        "distance_from_home",
        "user_age",
        "velocity_60m",
        "device_risk_score",
    }


def test_batch_transaction_empty():
    """Verifies that an empty batch list raises ValidationError."""
    with pytest.raises(ValidationError):
        BatchTransactionInput(transactions=[])
