import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#1 EDA Y PREPROCESAMIENTO REPRODUCIBLE:se implementa una etapa de limpieza y transformación de datos orientada a garantizar reproducibilidad y consistencia durante el entrenamiento de modelos.
#Los valores inválidos serán reemplazados utilizando la mediana de cada atributo.
def load_and_clean_data(filepath):
    """Carga el dataset y reemplaza los valores 0 inválidos por la mediana."""
    df = pd.read_csv(filepath)
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        
    return df
#2 PREPARACION DE SPLITS ESTRATIFICADOS: se implementa una función que separa las características del target 
#y realiza un split estratificado para asegurar que la distribución de clases se mantenga tanto en el conjunto de entrenamiento como en el de prueba.
def prepare_splits(df, target_col="Outcome", test_size=0.2, random_state=42):
    """Separa las características del target y realiza el split estratificado."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test