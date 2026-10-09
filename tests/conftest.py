"""
PyTest Test Configuration and Fixtures.

Provides reusable mock transactions, batch payloads, and HTTP clients
for integration and unit testing.
"""

from typing import AsyncGenerator, Dict
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.domain.transaction import BatchTransactionInput, TransactionInput


@pytest.fixture
def sample_legit_transaction_dict() -> Dict:
    """Fixture providing typical legitimate transaction attributes."""
    return {
        "transaction_amount": 45.50,
        "hour_of_day": 14,
        "distance_from_home": 3.2,
        "user_age": 38,
        "velocity_60m": 1,
        "device_risk_score": 0.08,
    }


@pytest.fixture
def sample_fraud_transaction_dict() -> Dict:
    """Fixture providing high-risk fraudulent transaction attributes."""
    return {
        "transaction_amount": 3200.00,
        "hour_of_day": 3,
        "distance_from_home": 280.0,
        "user_age": 29,
        "velocity_60m": 9,
        "device_risk_score": 0.94,
    }


@pytest.fixture
def sample_legit_transaction(sample_legit_transaction_dict: Dict) -> TransactionInput:
    """Fixture providing validated Pydantic model for legitimate transaction."""
    return TransactionInput(**sample_legit_transaction_dict)


@pytest.fixture
def sample_fraud_transaction(sample_fraud_transaction_dict: Dict) -> TransactionInput:
    """Fixture providing validated Pydantic model for fraud transaction."""
    return TransactionInput(**sample_fraud_transaction_dict)


@pytest.fixture
def sample_batch_transactions(
    sample_legit_transaction_dict: Dict,
    sample_fraud_transaction_dict: Dict,
) -> BatchTransactionInput:
    """Fixture providing small batch for testing drift and batch endpoints."""
    items = []
    # 20 normal items
    for _ in range(20):
        items.append(TransactionInput(**sample_legit_transaction_dict))
    # 5 fraud items
    for _ in range(5):
        items.append(TransactionInput(**sample_fraud_transaction_dict))
    return BatchTransactionInput(transactions=items)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provides async HTTP client bound to FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
