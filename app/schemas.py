from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Edad: int = Field(..., ge=18, le=100)
    Nivel_Educacional: str = Field(..., min_length=1)
    Años_Trabajando: int = Field(..., ge=0, le=80)
    Ingresos: int = Field(..., ge=0)
    Deuda_Comercial: float = Field(..., ge=0)
    Deuda_Credito: float = Field(..., ge=0)
    Otras_Deudas: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float = Field(..., ge=0, le=1)


class BatchPredictionRequest(BaseModel):
    instances: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelSchemaResponse(BaseModel):
    features: list[str]
    feature_types: dict[str, str]
    numeric_features: list[str]
    categorical_features: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
