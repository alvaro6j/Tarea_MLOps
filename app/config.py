import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def get_path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)

    if value:
        return Path(value).expanduser()

    return default


@dataclass(frozen=True)
class Settings:
    app_env: str
    api_title: str
    api_description: str
    api_version: str
    model_path: Path
    metadata_path: Path


settings = Settings(
    app_env=os.getenv("APP_ENV", "development"),
    api_title=os.getenv(
        "API_TITLE",
        "Default Prediction API",
    ),
    api_description=os.getenv(
        "API_DESCRIPTION",
        "API para predicción de riesgo de Default de clientes.",
    ),
    api_version=os.getenv("API_VERSION", "1.0.0"),
    model_path=get_path_from_env(
        "MODEL_PATH",
        BASE_DIR / "models" / "model.joblib",
    ),
    metadata_path=get_path_from_env(
        "METADATA_PATH",
        BASE_DIR / "models" / "metadata.json",
    ),
)
