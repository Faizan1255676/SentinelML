"""
FastAPI Microservice Application Entrypoint.

Configures lifespan startup/shutdown lifecycle, custom JSON exception handling,
observability middlewares, CORS, and endpoint mounting.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.logger import get_logger
from src.api.middleware import ObservabilityMiddleware
from src.api.routes import router
from src.ml.inference import FraudInferenceEngine
from src.ml.drift import ModelDriftDetector

logger = get_logger("sentinelml.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan manager executing pre-warming and graceful shutdown."""
    logger.info("Initializing SentinelML microservice assets during startup...")
    # Pre-warm model singleton & baseline data
    try:
        engine = FraudInferenceEngine.get_instance()
        detector = ModelDriftDetector()
        logger.info(
            f"Assets loaded successfully. Model loaded: {engine.is_loaded}, "
            f"Baseline loaded: {detector.baseline_df is not None}"
        )
    except Exception as exc:
        logger.warning(f"Startup asset pre-load error: {exc}")

    yield

    logger.info("SentinelML microservice gracefully shutting down.")


def create_app() -> FastAPI:
    """Application factory for FastAPI instance."""
    app = FastAPI(
        title="SentinelML - Real-Time Financial Fraud & ML-Observability Engine",
        description=(
            "Enterprise-grade financial transaction risk engine powered by LightGBM, "
            "SHAP explainability, and statistical distribution drift detection."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Observability & latency middleware
    app.add_middleware(ObservabilityMiddleware)

    # Custom exception handlers for strict enterprise responses
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            f"Validation error on {request.method} {request.url.path}: {exc.errors()}",
            extra={"request_id": request_id, "status_code": 422},
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "request_id": request_id,
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"Global unhandled error on {request.method} {request.url.path}: {exc}",
            extra={"request_id": request_id, "status_code": 500},
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "request_id": request_id,
                "message": "An unexpected error occurred processing your request.",
            },
        )

    # Mount routes directly at root for /health, /predict, /explain, /drift-check
    app.include_router(router)

    @app.get("/", tags=["Root"])
    async def root_overview():
        return {
            "name": "SentinelML Fraud & Observability Engine",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
