from fastapi import FastAPI, HTTPException

from app.predictor import Predictor
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelSchemaResponse,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Default Prediction API",
    description="API para predicción de riesgo de Default de clientes.",
    version="1.0.0",
)


# El modelo se carga una sola vez cuando inicia la aplicación.
predictor = Predictor()


@app.get("/health", response_model=HealthResponse)
def health():
    """Comprueba el estado del servicio y del modelo."""
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Realiza una predicción para un cliente."""
    try:
        return predictor.predict(
            request.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(request: BatchPredictionRequest):
    """Realiza predicciones para múltiples clientes."""
    try:
        instances = [
            instance.model_dump()
            for instance in request.instances
        ]

        predictions = predictor.predict_batch(instances)

        return {
            "predictions": predictions,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.get(
    "/model/schema",
    response_model=ModelSchemaResponse,
)
def model_schema():
    """Devuelve las features esperadas por el modelo."""
    return predictor.get_schema()