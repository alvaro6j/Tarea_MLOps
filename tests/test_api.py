from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "Edad": 34,
    "Nivel_Educacional": "SupInc",
    "Años_Trabajando": 11,
    "Ingresos": 68,
    "Deuda_Comercial": 11.9,
    "Deuda_Credito": 3.37,
    "Otras_Deudas": 4.73,
}


def test_health_returns_service_and_model_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_schema_returns_expected_features():
    response = client.get("/model/schema")

    assert response.status_code == 200

    body = response.json()

    assert "Edad" in body["features"]
    assert "Nivel_Educacional" in body["categorical_features"]
    assert "Ingresos" in body["numeric_features"]


def test_predict_returns_valid_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability"] <= 1


def test_batch_predict_returns_one_result_per_instance():
    second_instance = deepcopy(VALID_PAYLOAD)
    second_instance["Edad"] = 45
    second_instance["Ingresos"] = 120

    response = client.post(
        "/predict/batch",
        json={"instances": [VALID_PAYLOAD, second_instance]},
    )

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == 2
    assert all(item["prediction"] in [0, 1] for item in predictions)
    assert all(0 <= item["probability"] <= 1 for item in predictions)


def test_predict_rejects_missing_required_field():
    invalid_payload = deepcopy(VALID_PAYLOAD)
    invalid_payload.pop("Ingresos")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_rejects_age_below_minimum():
    invalid_payload = deepcopy(VALID_PAYLOAD)
    invalid_payload["Edad"] = 17

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_age_type():
    invalid_payload = deepcopy(VALID_PAYLOAD)
    invalid_payload["Edad"] = "no-es-un-numero"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_rejects_empty_education_level():
    invalid_payload = deepcopy(VALID_PAYLOAD)
    invalid_payload["Nivel_Educacional"] = ""

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_rejects_negative_income():
    invalid_payload = deepcopy(VALID_PAYLOAD)
    invalid_payload["Ingresos"] = -1

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_batch_rejects_invalid_instance():
    invalid_instance = deepcopy(VALID_PAYLOAD)
    invalid_instance["Deuda_Credito"] = -10

    response = client.post(
        "/predict/batch",
        json={"instances": [invalid_instance]},
    )

    assert response.status_code == 422


def test_predict_accepts_minimum_age():
    payload = deepcopy(VALID_PAYLOAD)
    payload["Edad"] = 18

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability"] <= 1


def test_predict_accepts_maximum_age():
    payload = deepcopy(VALID_PAYLOAD)
    payload["Edad"] = 100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200


def test_predict_rejects_age_above_maximum():
    payload = deepcopy(VALID_PAYLOAD)
    payload["Edad"] = 101

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_accepts_zero_income():
    payload = deepcopy(VALID_PAYLOAD)
    payload["Ingresos"] = 0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability"] <= 1


def test_predict_accepts_zero_credit_debt():
    payload = deepcopy(VALID_PAYLOAD)
    payload["Deuda_Credito"] = 0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability"] <= 1
