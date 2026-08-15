import json

import joblib
import pandas as pd

from app.config import settings


class Predictor:
    def __init__(self):
        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {settings.model_path}"
            )

        if not settings.metadata_path.exists():
            raise FileNotFoundError(
                f"No se encontró la metadata en: {settings.metadata_path}"
            )

        self.model = joblib.load(settings.model_path)

        with open(settings.metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.features = self.metadata["features"]

    def is_loaded(self) -> bool:
        """Indica si el modelo fue cargado correctamente."""
        return self.model is not None

    def predict(self, data: dict) -> dict:
        dataframe = pd.DataFrame([data])

        dataframe = dataframe[self.features]

        prediction = int(self.model.predict(dataframe)[0])

        probability = float(self.model.predict_proba(dataframe)[0][1])

        return {
            "prediction": prediction,
            "probability": probability,
        }

    def predict_batch(self, data: list[dict]) -> list[dict]:
        dataframe = pd.DataFrame(data)

        dataframe = dataframe[self.features]

        predictions = self.model.predict(dataframe)

        probabilities = self.model.predict_proba(dataframe)[:, 1]

        return [
            {
                "prediction": int(prediction),
                "probability": float(probability),
            }
            for prediction, probability in zip(
                predictions,
                probabilities,
            )
        ]

    def get_schema(self) -> dict:
        return {
            "features": self.metadata["features"],
            "numeric_features": self.metadata["numeric_features"],
            "categorical_features": self.metadata["categorical_features"],
        }
