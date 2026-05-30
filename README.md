#### LABORATORIO DE PIMA INDIANS DIABETES CON MLOPS

Tratamiento de las fases de MLOps con el dataset diabetes.csv

### 1. OBJETIVO DEL LABORATORIO Y RELACION CON EL PAPER

Objetivo: Implementar un pipeline básico de MLOps para un sistema de predicción de diabetes utilizando técnicas de Machine Learning y herramientas modernas de monitoreo y seguimiento de modelos.
El proyecto se basa en el paper principal:
*“Experimentation, deployment and monitoring Machine Learning models: Approaches for applying MLOps”*
El laboratorio busca aplicar las tres etapas fundamentales descritas en el paper:
-Experimentation
-Deployment
-Monitoring
Para ello se utilizará el dataset “Pima Indians Diabetes Database”, implementando procesos de:
-carga y exploración de datos
-preprocesamiento reproducible
-entrenamiento y comparación de modelos
-seguimiento de métricas con MLflow
-deployment mediante Flusk y Docker
-monitoreo de drift poniendolo dispuesto para Evidently AI
Asimismo, el laboratorio incorpora conceptos presentes en las lecturas complementarias relacionados con:
-gobernanza de modelos
-trazabilidad
-monitoreo continuo
-automatización
-madurez MLOps.
El sistema completo será desarrollado en VS Studio Code Python y librerías de Machine Learning.

---

### 2. CARGA DE DATOS 

Para garantizar la reproducibilidad del laboratorio, la carga de datos se gestiona de forma local y automatizada dentro del pipeline del proyecto. El conjunto de datos original (Pima Indians Diabetes Database) se almacena en formato estructurado dentro del directorio del proyecto.
`/src/preprocess.py`

---

### 3. EDA O INSPECCION MINIMA 

Se realiza una inspección básica del dataset con el objetivo de:
comprender la estructura de los datos,
identificar valores inválidos,
analizar la distribución de clases,
detectar posibles problemas de calidad.

Inspección por Consola
Durante la ejecución del pipeline, se extraen las siguientes propiedades utilizando Pandas:
* **Dimensión y Tipos de Datos (`df.info()`):** Verificación de consistencia en tipos numéricos (`float64` e `int64`).
* **Consistencia de Nulos (`df.isnull().sum()`):** Comprobación de la efectividad del limpiador matemático tras imputar los ceros inválidos con la mediana.
* **Dispersión Estadística (`df.describe()`):** Análisis de promedios, desviaciones estándar y percentiles.

Visualizaciones Generadas
Las gráficas son exportadas automáticamente al directorio `/static` para evitar interrupciones de ventanas emergentes en servidores de producción:
1. `static/distribucion_outcome.png`: Diagrama de barras que evalúa el desbalance de clases entre pacientes sanos y diabéticos. El grafico nos ayuda a ver que hay mayor distribucion de clases hay mas personas no diabeticas.
2. `static/heatmap_correlacion.png`: Matriz de correlación de Pearson para identificar colinealidad (ej. relación entre la glucosa o el IMC frente al diagnóstico final). Aqui se puede apreciar que no hay alguna correlacion importante por lo tanto no es
necesario hacer algun otro tratamiento mas. 

Este punto se enceuntra presente en el  `/src/preprocess.py`

---
### 4. PREPROCESAMIENTO REPRODUCIBLE 

Para garantizar que el modelo reciba los datos con el mismo tratamiento matemático tanto en el entrenamiento como en la inferencia en producción (evitando el data leakage), toda la lógica de limpieza y transformación se encuentra centralizada y automatizada dentro del script corporativo src/train.py.
Al ejecutar este archivo, el pipeline aplica un pipeline de ingeniería de características consistente sin intervención manual. 

Este punto se enceuntra presente en el  `/src/preprocess.py`

---
### 5. BASELINE

Se selecciono como modelo base **LOGISTIC REGRESSION** porque:
- **Problema binario:** La variable `Outcome` es 0/1 (sin diabetes / con diabetes). La regresión logística modela probabilidades en [0,1] con la función sigmoide.
- **Simplicidad y coste bajo:** Entrenamiento rápido, sin tuning pesado y mínimo consumo en `src/train.py`. Ideal como primer paso en MLOps.
- **Interpretable:** Los coeficientes muestran el impacto de cada variable (glucosa, edad, etc.) en el diagnóstico.
- **Mide la ganancia real:** Si un modelo complejo (Random Forest, XGBoost) no supera significativamente el accuracy o recall de esta regresión, la complejidad extra no se justifica en producción.
Este punto se enceuntra presente en el  `/src/train.py`

--- 

### 6. MODELO O TECNICA PRINCIPAL

Se selecciono como modelo principal **RANDOM FOREST** porque: 
- **Relaciones no lineales e interacciones:** La diabetes depende de interacciones complejas (ej. insulina + IMC + edad). Random Forest captura estas bifurcaciones de forma nativa, sin asumir linealidad.
- **Robustez ante outliers y datos imputados:** Al basarse en particiones, no se ve afectado por valores atípicos ni por distribuciones sesgadas (como los ceros imputados con mediana en `src/train.py`).
- **Prevención del sobreajuste:** Bagging + votación mayoritaria entre múltiples árboles reduce la varianza, asegurando generalización en la API (`app.py`).
- **Importancia de características:** Mide automáticamente qué variables clínicas (glucosa, edad, etc.) son más relevantes, aportando interpretabilidad y auditoría al modelo.
- **Fácil despliegue:** Serialización ligera (`model.pkl` o MLflow), inferencia en milisegundos, ideal para Docker y respuestas HTTP rápidas.

Este punto se enceuntra presente en el  `/src/train.py`

---
### 7. METRICAS Y VISUALIZACION

# Automatización del Rendimiento con MLflow

Todo el proceso de evaluación está automatizado en `src/train.py` y registrado en un servidor local de MLflow (`mlflow_proyectos.db`).

## Métricas registradas (test set, 20% estratificado)
- **Accuracy** — Proporción global de aciertos.
- **Precision** — Clave clínica: reduce falsos positivos (evita alarmar a pacientes sanos).
- **Recall / Sensibilidad** — **Métrica más crítica**. Minimizar falsos negativos (evitar enviar a casa a un paciente diabético sin tratamiento).

---
### 8. ANALISIS DE ERROR

El análisis de error inspecciona quirúrgicamente en qué escenarios específicos la arquitectura comete fallos (Falsos Positivos y Falsos Negativos), lo cual es crítico en un entorno de diagnóstico de salud.

1. Inspección de Fallos Clínicos: El pipeline evalúa el rendimiento del modelo segmentando los errores por variables críticas como la Glucosa o la Edad. Esto permite identificar si el algoritmo pierde precisión en sectores demográficos específicos (por ejemplo, pacientes jóvenes con sintomatología atípica).

2. Trazabilidad de Desviaciones: Gracias al almacenamiento estructurado en mlflow_proyectos.db, se pueden contrastar las ejecuciones para auditar si las modificaciones en los hiperparámetros reducen los Falsos Negativos, disminuyendo el riesgo clínico de omitir un diagnóstico oportuno.

Este punto se enceuntra presente en el  `/src/train.py`

---
### 9. CONCLUSIONES Y LIMITACIONES
El laboratorio permitió implementar un pipeline básico de MLOps orientado a la predicción de diabetes utilizando Machine Learning.
Durante el desarrollo se aplicaron conceptos fundamentales descritos en el paper principal:
-experimentación
-deployment
-monitoring
-trazabilidad
-automatización
Asimismo, MLflow permitió registrar métricas, parámetros y modelos, facilitando el seguimiento de experimentos y la reproducibilidad del proyecto.

Los resultados obtenidos muestran que Random Forest presentó mejor desempeño que Logistic Regression en términos de F1-score y recall.

Sin embargo, el laboratorio presenta ciertas limitaciones:

1. el dataset es relativamente pequeño
2. no existe reentrenamiento automático
3. no se implementó CI/CD real
5. no se utilizó infraestructura empresarial como Kubernetes.
A pesar de ello, el proyecto demuestra correctamente los principios fundamentales de MLOps descritos en las lecturas analizadas.

---
### 10. INSTRUCCIONES DE EJECUCION

##  Arquitectura del Sistema
El proyecto está dividido en componentes independientes bajo las mejores prácticas que indicaron los papers aplicados:
1. **Fase de Datos (`preprocess.py`):** Limpieza reproducible (imputación de medianas reales usando `NaN`) y partición estratificada.
2. **Fase de Entrenamiento (`train.py`):** Pipelines reproducibles, comparación de modelos (Logistic Regression vs. Random Forest), análisis de errores y tracking con **MLflow**.
3. **Fase de Despliegue (`deploy/`):** API en **Flask** empaquetada en un contenedor **Docker** y expuesta mediante **Swagger UI**.
4. **Fase de Monitoreo (`monitoring/`):** Almacenamiento de predicciones en vivo listo para un posterior analisis

## Requisitos Previos
Antes de iniciar, asegúrate de tener instalado en tu computadora:
* Python 3.10 o superior
* Docker Desktop (Con el motor de Docker corriendo)

## Guía de Ejecución Paso a Paso

### 1. Preparar el Entorno

git clone [https://github.com/tu-usuario/diabetes-mlops.git](https://github.com/tu-usuario/diabetes-mlops.git)
diabetes-mlops

# Crear e instalar las dependencias locales
pip install -r requirements.txt

### 2. Fase de experimentacion y tracking con MlFlow
# Ejecutar el pipeline de entrenamiento completo
python src/python train.py
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
Cada consulta realizada exitosamente a través de la interfaz de Swagger se guarda automáticamente de forma incremental en el archivo local /monitoring
