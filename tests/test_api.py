"""
FastAPI Microservice Integration Tests.

Validates HTTP REST endpoints, status codes, headers, and validation behaviors
using httpx AsyncClient.
"""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Verifies that GET /health returns 200 OK with healthy status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["baseline_loaded"] is True
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    """Verifies that GET / returns 200 OK with service info."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "SentinelML" in data["name"]


@pytest.mark.asyncio
async def test_predict_endpoint_success(async_client, sample_legit_transaction_dict):
    """Verifies POST /predict handles legitimate transactions successfully."""
    response = await async_client.post("/predict", json=sample_legit_transaction_dict)
    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert "is_fraud" in data
    assert data["is_fraud"] is False
    assert data["risk_level"] in ["LOW", "MEDIUM"]
    assert "latency_ms" in data

    # Observability headers check
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


@pytest.mark.asyncio
async def test_predict_endpoint_fraud(async_client, sample_fraud_transaction_dict):
    """Verifies POST /predict correctly flags attack patterns as fraud."""
    response = await async_client.post("/predict", json=sample_fraud_transaction_dict)
    assert response.status_code == 200
    data = response.json()

    assert data["is_fraud"] is True
    assert data["risk_score"] >= 0.50
    assert data["risk_level"] in ["HIGH", "CRITICAL"]


@pytest.mark.asyncio
async def test_predict_endpoint_validation_error(async_client, sample_legit_transaction_dict):
    """Verifies POST /predict returns 422 with structured error when payload is invalid."""
    corrupted = sample_legit_transaction_dict.copy()
    corrupted["transaction_amount"] = -500.0  # Invalid negative amount

    response = await async_client.post("/predict", json=corrupted)
    assert response.status_code == 422
    data = response.json()

    assert data["error"] == "Validation Error"
    assert "details" in data
    assert "request_id" in data


@pytest.mark.asyncio
async def test_explain_endpoint_success(async_client, sample_fraud_transaction_dict):
    """Verifies POST /explain returns local SHAP values and risk factor breakdown."""
    response = await async_client.post("/explain", json=sample_fraud_transaction_dict)
    assert response.status_code == 200
    data = response.json()

    assert "shap_values" in data
    assert "top_risk_factors" in data
    assert len(data["shap_values"]) == 6
    assert len(data["top_risk_factors"]) == 6
    assert "base_value" in data
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_drift_check_endpoint_success(async_client, sample_batch_transactions):
    """Verifies POST /drift-check calculates statistical drift for batches."""
    payload = sample_batch_transactions.model_dump()
    response = await async_client.post("/drift-check", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "sample_size" in data
    assert data["sample_size"] == 25
    assert "baseline_sample_size" in data
    assert "feature_reports" in data
    assert len(data["feature_reports"]) == 6
    assert "overall_drift_detected" in data
