# diabetes-mlops
Tratamiento de las fases de MLOps con el dataset diabetes.csv
##  Arquitectura del Sistema
El proyecto está dividido en componentes independientes bajo las mejores prácticas que indicaron los papers aplicados:
1. **Fase de Datos (`preprocess.py`):** Limpieza reproducible (imputación de medianas reales usando `NaN`) y partición estratificada.
2. **Fase de Entrenamiento (`train.py`):** Pipelines reproducibles, comparación de modelos (Logistic Regression vs. Random Forest), análisis de errores y tracking con **MLflow**.
3. **Fase de Despliegue (`deploy/`):** API en **Flask** empaquetada en un contenedor **Docker** y expuesta mediante **Swagger UI**.
4. **Fase de Monitoreo (`monitoring/`):** Almacenamiento de predicciones en vivo listo para un posterior analisis

---

## Requisitos Previos
Antes de iniciar, asegúrate de tener instalado en tu computadora:
* Python 3.10 o superior
* Docker Desktop (Con el motor de Docker corriendo)

---

## Guía de Ejecución Paso a Paso

### 1. Preparar el Entorno

git clone [https://github.com/tu-usuario/diabetes-mlops.git](https://github.com/tu-usuario/diabetes-mlops.git)
cd diabetes-mlops

# Crear e instalar las dependencias locales
pip install -r requirements.txt

### 2. Fase de experimentacion y tracking con MlFlow
# Ejecutar el pipeline de entrenamiento completo
python train.py
# Para visualizar y comparar de forma gráfica la Regresión Logística contra el Random Forest, enciende el servidor local de MLflow UI:
mlflow ui --backend-store-uri sqlite:///mlflow_proyectos.db

### 3. Fase de despliegue con Docker y Swagger
# Construir la imagen de Docker usando el archivo Dockerfile personalizado
docker build -t diabetes-api:v1 -f deploy/Dockerfile .
# Levantar el contenedor mapeando el puerto 8000 y el volumen local de monitoreo
docker run -d -p 8000:8000 --name diabetes_api_server -v //"$(pwd)/monitoring:/app/monitoring" diabetes-api:v1
# Ahora abre la terminal en el siguiente puerto que te desplegara la interfaz de swagger para que tu mismo puedas hacer tus propias predicciones
http://localhost:8000/docs/

### 4. Fase de monitoreo continuo 
Cada consulta realizada exitosamente a través de la interfaz de Swagger se guarda automáticamente de forma incremental en el archivo local:
