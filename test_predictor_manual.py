from app.predictor import Predictor

predictor = Predictor()

sample = {
    "Edad": 35,
    "Nivel_Educacional": "SupCom",
    "Años_Trabajando": 10,
    "Ingresos": 80,
    "Deuda_Comercial": 5.2,
    "Deuda_Credito": 2.1,
    "Otras_Deudas": 3.4,
}

result = predictor.predict(sample)

print("Predicción individual:")
print(result)

print("\nSchema:")
print(predictor.get_schema())

batch = [
    sample,
    {
        "Edad": 50,
        "Nivel_Educacional": "Med",
        "Años_Trabajando": 20,
        "Ingresos": 120,
        "Deuda_Comercial": 10.0,
        "Deuda_Credito": 5.0,
        "Otras_Deudas": 4.0,
    },
]

print("\nPredicciones batch:")
print(predictor.predict_batch(batch))
