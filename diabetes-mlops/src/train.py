import os
import joblib  
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import load_and_clean_data, prepare_splits

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def run_experiment():
    data_path = os.path.join("data", "diabetes.csv") 
    
    if not os.path.exists(data_path):
        print(f"error: No se encontró el archivo en '{data_path}'. Revisa su ubicación.")
        return

    df = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = prepare_splits(df) 

    mlflow.set_tracking_uri("sqlite:///mlflow_proyectos.db")
    mlflow.set_experiment("Diabetes_MLOps_Local")
    
    # === EXPERIMENTO 1: REGRESIÓN LOGÍSTICA ===
    #3 PIPELINE RERPODUCIBLE Y TRACKING DE EXPERIMENTOS: se implementa un pipeline de entrenamiento 
    #que incluye tanto la estandarización de características como el modelo de regresión logística que es seleccionado MODELO BASELINE.
    with mlflow.start_run(run_name="Logistic_Regression"):
        baseline_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])
        
        baseline_pipeline.fit(X_train, y_train)
        preds = baseline_pipeline.predict(X_test)
        acc, prec, rec, f1 = evaluate_metrics(y_test, preds)
        #metricas trackeadas en mlflow para comparacion futura con el modelo de random forest
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(baseline_pipeline, "logistic_model")
        print("Model 1/2 (Logistic Regression) trackeado con éxito.")
        
    # === EXPERIMENTO 2: RANDOM FOREST ===
    #4 MODELO MEJORADO Y EXPORTACIÓN PARA PRODUCCIÓN: se implementa un segundo modelo utilizando Random Forest, 
    # el cual es trackeado en mlflow y exportado como 'model.pkl' para su posterior despliegue en producción.
    #fue seleccionado por su capacidad para manejar relaciones no lineales y mejorar el rendimiento predictivo.
    with mlflow.start_run(run_name="Random_Forest"):
        rf_pipeline = Pipeline([
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        rf_pipeline.fit(X_train, y_train)
        preds = rf_pipeline.predict(X_test)
        acc, prec, rec, f1 = evaluate_metrics(y_test, preds)
        #metricas trackeadas en mlflow para comparacion futura con el modelo baseline de regresion logistica
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(rf_pipeline, "random_forest_model")
        print("Model 2/2 (Random Forest) trackeado con éxito.")
        #5 ANÁLISIS DE ERROR Y MONITOREO PRODUCTIVO: se implementa un análisis de error detallado para el modelo de Random Forest,
        #identificando las instancias mal clasificadas en el conjunto de prueba y exportando esta información a un archivo CSV para su monitoreo continuo en producción.
        print("\n Realizando Análisis de Error para Random Forest...")
        errors = X_test.copy()
        errors["Actual"] = y_test.values
        errors["Predicted"] = preds
        
        misclassified = errors[errors["Actual"] != errors["Predicted"]]
        print(f"Cantidad de errores detectados en Test: {len(misclassified)}")
        os.makedirs("deploy", exist_ok=True)
        joblib.dump(rf_pipeline, "model.pkl")         
        print("Archivo 'model.pkl' exportado para producción correctamente.")

if __name__ == "__main__":
    run_experiment()