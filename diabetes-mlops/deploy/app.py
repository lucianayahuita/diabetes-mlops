import os
import json  
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

SWAGGER_URL = '/docs'  
API_URL = '/swagger.json' 

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "API de Diabetes (MLOps)"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/swagger.json', methods=['GET'])
def swagger_spec():
    posibles_rutas = ["swagger.json", "static/swagger.json", "deploy/static/swagger.json"]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            with open(ruta, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
    return jsonify({"error": "No se encontró el archivo swagger.json en el servidor"}), 404

MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("¡Modelo cargado desde la raíz con éxito!")
elif os.path.exists("deploy/model.pkl"):
    model = joblib.load("deploy/model.pkl")
    print("¡Modelo cargado desde la carpeta deploy!")
else:
    model = None
    print("ERROR: El archivo model.pkl no existe en el contenedor.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API corriendo exitosamente", "model_loaded": model is not None})
# FUNCION CRITICA: revisa si el modelo existe antes de intentar predecir, si no existe devuelve error
# luego, recibe los datos en un JSON y los convierte en un Df para hacer la prediccion,
# finalmente, guarda la prediccion y su probabilidad en un CSV para monitoreo continuo y devuelve el resultado al cliente.
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no disponible en el servidor"}), 500
    
    try:
        data = request.get_json()
        df_features = pd.DataFrame([data])
        prediction = int(model.predict(df_features)[0])
        probability = float(model.predict_proba(df_features)[0][1])
        
        df_features['Prediction'] = prediction
        df_features['Probability'] = probability
        monitor_path = "/app/monitoring/live_predictions.csv"
        if not os.path.exists(monitor_path):
            df_features.to_csv(monitor_path, index=False)
        else:
            df_features.to_csv(monitor_path, mode='a', header=False, index=False)
            
        return jsonify({
            "diabetes_prediction": prediction,
            "probability_of_diabetes": round(probability, 4)
        })
        
    except Exception as e:
        return jsonify({"error_de_ejecucion": str(e)}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)