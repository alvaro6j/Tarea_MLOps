# Tarea MLOps – Predicción de riesgo de incumplimiento de pagos de créditos

## Integrantes

- Germán Ruiz
- Camilo Silva
- Álvaro Gómez

---

## 1. Descripción

El trabajo corresponde a un proyecto de **Machine Learning Operations (MLOps)** orientado a la predicción del cumplimiento de pago de créditos otorgados a clientes.

Las variables consideradas son:

- **Edad:** campo cuantitativo que indica la edad del cliente.
- **Nivel Educacional:** campo categórico que indica el nivel educacional del cliente.
- **Años Trabajando:** campo cuantitativo con los años de experiencia laboral del cliente.
- **Ingresos:** campo cuantitativo correspondiente al monto de ingresos del cliente.
- **Deuda Comercial:** campo cuantitativo correspondiente a la deuda comercial del cliente.
- **Deuda Crédito:** campo cuantitativo correspondiente a la deuda de consumo del cliente.
- **Otras Deudas:** campo cuantitativo correspondiente a otras deudas del cliente.
- **Default:** variable objetivo binaria. El valor `1` representa **cumplimiento** y el valor `0` representa **incumplimiento**.

El sistema permite entrenar un modelo de Machine Learning a partir de datos históricos de clientes, evaluar su desempeño sobre datos no vistos y exponer el modelo mediante una API REST desarrollada con FastAPI.

La solución está contenerizada mediante Docker y cuenta con automatización de pruebas, linting, entrenamiento, validación del modelo, quality gate y pruebas de humo mediante GitHub Actions.

### Arquitectura general

El flujo principal de la solución es:

```text
Dataset
   │
   ▼
training/train.py
   │
   ├── Preprocesamiento
   ├── Entrenamiento
   ├── Evaluación
   └── Artefactos
          │
          ├── models/model.joblib
          └── models/metadata.json
                    │
                    ▼
                 FastAPI
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       /health   /predict   /predict/batch
                    │
                    ▼
                 Docker
                    │
                    ▼
              GitHub Actions
```

---

## Estructura del proyecto

```text
MLOps_Trabajo/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── predictor.py
│   └── config.py
├── data/
├── models/
│   ├── model.joblib
│   └── metadata.json
├── tests/
├── training/
│   ├── train.py
│   └── evaluate.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── audit_dataset.py
├── body.json
├── test_predictor_manual.py
├── README.md
└── Informe_Proyecto_MLOps.pdf
```

---

## 2. Datos

El modelo utiliza un dataset histórico de clientes.

La variable objetivo es:

```text
Default
```

Las variables utilizadas como entrada son:

| Variable | Tipo |
|---|---|
| Edad | Integer |
| Nivel_Educacional | String |
| Años_Trabajando | Integer |
| Ingresos | Integer |
| Deuda_Comercial | Number |
| Deuda_Credito | Number |
| Otras_Deudas | Number |

Durante el entrenamiento se eliminan duplicados considerando las variables utilizadas por el modelo.

El dataset original contiene **12.356 registros**. Después de eliminar duplicados quedan **1.500 registros**.

La separación de datos utilizada es:

- **70%** para entrenamiento.
- **30%** para prueba.
- Separación estratificada según la variable objetivo.

Se utiliza una semilla fija:

```text
random_seed = 0
```

---

## 3. Modelo

El modelo utilizado es:

```text
HistGradientBoostingClassifier
```

El entrenamiento se implementa en:

```text
training/train.py
```

El procesamiento se realiza mediante un pipeline de **scikit-learn**.

### Variables numéricas

- Edad
- Años_Trabajando
- Ingresos
- Deuda_Comercial
- Deuda_Credito
- Otras_Deudas

Para estas variables se utiliza imputación mediante la mediana.

### Variable categórica

- Nivel_Educacional

Para esta variable se utiliza:

```python
SimpleImputer(strategy="most_frequent")
```

seguido de:

```python
OneHotEncoder(handle_unknown="ignore")
```

El preprocesamiento y el modelo quedan integrados en un único pipeline y son serializados como:

```text
models/model.joblib
```

Este archivo contiene el pipeline entrenado que posteriormente es cargado por la API para realizar inferencias.

---

## 4. Resultados

La evaluación se realiza sobre un conjunto de prueba que no participa en el entrenamiento del modelo.

Los resultados obtenidos son:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7222 |
| F1 | 0.7906 |
| ROC-AUC | 0.7772 |

El proyecto incorpora además un **quality gate** que verifica que el modelo alcance los valores mínimos definidos para las métricas F1 y ROC-AUC.

Los metadatos del entrenamiento se almacenan en:

```text
models/metadata.json
```

Estos incluyen:

- Modelo utilizado.
- Versión de Python.
- Versión de scikit-learn.
- Fecha de entrenamiento.
- Semilla utilizada.
- Tamaño del conjunto de prueba.
- Features utilizadas.
- Tipos de features.
- Métricas obtenidas.

---

## 5. API REST

La API está implementada utilizando **FastAPI**.

La documentación interactiva queda disponible en:

```text
http://localhost:8000/docs
```

### Endpoints

#### GET `/health`

Comprueba el estado del servicio y si el modelo fue cargado correctamente.

Ejemplo:

```powershell
curl.exe http://localhost:8000/health
```

Respuesta:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

#### GET `/model/schema`

Devuelve las features esperadas por el modelo y sus tipos.

Ejemplo:

```powershell
curl.exe http://localhost:8000/model/schema
```

Respuesta:

```json
{
  "features": [
    "Edad",
    "Nivel_Educacional",
    "Años_Trabajando",
    "Ingresos",
    "Deuda_Comercial",
    "Deuda_Credito",
    "Otras_Deudas"
  ],
  "feature_types": {
    "Edad": "integer",
    "Nivel_Educacional": "string",
    "Años_Trabajando": "integer",
    "Ingresos": "integer",
    "Deuda_Comercial": "number",
    "Deuda_Credito": "number",
    "Otras_Deudas": "number"
  },
  "numeric_features": [
    "Edad",
    "Años_Trabajando",
    "Ingresos",
    "Deuda_Comercial",
    "Deuda_Credito",
    "Otras_Deudas"
  ],
  "categorical_features": [
    "Nivel_Educacional"
  ]
}
```

---

#### POST `/predict`

Realiza una predicción individual.

Ejemplo:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@body.json"
```

Con el ejemplo utilizado en el proyecto, la respuesta obtenida es:

```json
{
  "prediction": 1,
  "probability": 0.9867107096150026
}
```

Donde:

- `prediction = 1` indica **cumplimiento**.
- `prediction = 0` indica **incumplimiento**.
- `probability` corresponde a la probabilidad estimada por el modelo.

---

#### POST `/predict/batch`

Permite realizar predicciones para múltiples clientes en una sola solicitud.

La entrada y salida de los endpoints son validadas mediante modelos **Pydantic**.

Las entradas inválidas generan respuestas HTTP `422` con información sobre el error.

---

## 6. Ejecución con Docker

El proyecto incluye:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

La imagen utiliza:

```text
python:3.12-slim
```

Las dependencias se encuentran fijadas por versión en:

```text
requirements.txt
```

El contenedor ejecuta la aplicación utilizando un usuario no-root y cuenta con un `HEALTHCHECK`.

### Requisitos

- Docker Desktop instalado.
- Docker Desktop ejecutándose.

### Levantar la aplicación

Desde la raíz del proyecto:

```powershell
docker compose up --build
```

La API estará disponible en:

```text
http://localhost:8000
```

La documentación interactiva estará disponible en:

```text
http://localhost:8000/docs
```

### Detener el servicio

```powershell
docker compose down
```

---

## 7. Ejecución del entrenamiento

El entrenamiento puede ejecutarse localmente mediante:

```powershell
python training/train.py
```

Esto genera o actualiza:

```text
models/model.joblib
models/metadata.json
```

El script muestra por consola las métricas obtenidas sobre el conjunto de prueba.

---

## 8. Pruebas automatizadas

El proyecto utiliza **pytest** para las pruebas automatizadas.

Para ejecutar las pruebas:

```powershell
pytest -q
```

Las pruebas cubren, entre otros:

- Contrato de la API.
- Endpoint `/health`.
- Endpoint `/predict`.
- Endpoint `/predict/batch`.
- Endpoint `/model/schema`.
- Validación de entradas.
- Valores fuera de rango.
- Casos borde.
- Errores HTTP `422`.
- Quality gate del modelo.

Las pruebas se ejecutan sin necesidad de conexión a servicios externos ni credenciales.

---

## 9. Calidad de código

Se utiliza **Ruff** como linter y formateador.

Para comprobar el linting:

```powershell
ruff check .
```

Para comprobar el formato:

```powershell
ruff format --check .
```

---

## 10. CI/CD

El proyecto utiliza **GitHub Actions** para automatizar el proceso de integración continua y entrega.

El pipeline contempla las siguientes etapas:

```text
Lint
  ↓
Tests
  ↓
Entrenamiento + Quality Gate
  ↓
Build Docker
  ↓
Smoke Test
```

La prueba de humo levanta el contenedor y verifica que la API responda correctamente en:

- `GET /health`
- `POST /predict`

Además, el proyecto contempla la publicación de la imagen Docker en **GitHub Container Registry (GHCR)** durante el proceso de release.

---

## 11. Variables de configuración

Las variables de configuración se documentan en:

```text
.env.example
```

Ejemplo:

```env
APP_ENV=development
API_TITLE=Default Prediction API
API_DESCRIPTION=API para predicción de riesgo de Default de clientes.
API_VERSION=1.0.0
MODEL_PATH=models/model.joblib
METADATA_PATH=models/metadata.json
```

No se incluyen credenciales ni secretos reales en el repositorio.

---

## 12. Limitaciones conocidas

El proyecto presenta algunas limitaciones:

- El modelo se entrena con un dataset histórico y su rendimiento puede disminuir frente a datos futuros con características diferentes.
- La cantidad de datos disponibles después de eliminar duplicados es limitada.
- No se implementa actualmente monitoreo de **data drift** o **model drift** en producción.
- No existe un sistema de registro y comparación de múltiples versiones de modelos.
- La API actualmente se ejecuta como un único servicio y no contempla escalamiento horizontal.
- Las métricas utilizadas corresponden a una única separación train/test y no a validación cruzada.

---

## 13. Trabajo futuro

Con más tiempo, el equipo podría incorporar:

- Tracking de experimentos con herramientas como MLflow.
- Registro y versionado formal de modelos.
- Monitoreo de drift.
- Monitoreo de métricas de producción.
- Reentrenamiento automático.
- Validación cruzada.
- Despliegue en infraestructura cloud.
- Escalamiento de la API.
- Autenticación y autorización para ambientes productivos.

---

## 14. Integrantes y contribuciones

### Álvaro Gómez

Diseño e implementación inicial del proyecto, entrenamiento del modelo, pipeline de Machine Learning, API FastAPI, configuración del predictor y contenerización Docker. Colaboración en el informe.

### Germán Ruiz

Desarrollo y ampliación de pruebas automatizadas, validación de entradas, casos borde, pruebas de endpoints y quality gate. Colaboración y subida del informe.

### Camilo Silva

Diseño del modelo de Machine Learning, revisión y mejoras de CI/CD, pruebas de smoke test, documentación del proyecto y README. Colaboración en el informe.

---

## Informe del proyecto

El informe final del proyecto se encuentra disponible en:

```text
Informe_Proyecto_MLOps.pdf
```

Nota: Durante el desarrollo del proyecto se utilizaron asistentes de Inteligencia Artificial como herramienta de apoyo al trabajo técnico y documental.

ChatGPT: utilizado como apoyo para resolver dudas sobre Python, FastAPI, Docker, Git/GitHub, GitHub Actions y pruebas automatizadas. Fue útil para guiarnos en revisar errores de ejecución; orientar la implementación de casos borde y apoyar la revisión y redacción de la documentación del proyecto.

Visual Studio Code y herramientas de desarrollo: utilizadas para implementar, revisar y ejecutar el código del proyecto.

La IA se utilizó como herramienta de apoyo y consulta, adicionalmente del material visto en clases, principalmente "MLOps_Pipeline_CICD_Documentacion". Las decisiones de implementación, integración, ejecución de pruebas, validación del sistema y revisión final de los resultados fueron realizadas por el equipo.