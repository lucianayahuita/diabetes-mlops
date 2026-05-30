import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---- EDA O INSPECCIÓN MÍNIMA ----
def perform_eda(df, output_dir="static"):
    """Realiza la inspección mínima del dataset y guarda los gráficos generados."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print(" INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*50)
    
    print("\nℹ Información del dataset:")
    print(df.info())
    
    print("\n Valores nulos por columna:")
    print(df.isnull().sum())
    
    print("\n Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n" + "="*50)
    print(" Generando y guardando gráficos del EDA...")
    print("="*50)
    
    # Gráfico 1: Distribución de la variable objetivo (Outcome)
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Outcome", data=df, palette="Set2")
    plt.title("Distribución de pacientes con diabetes")
    plt.xlabel("Resultado (0 = No Diabético, 1 = Diabético)")
    plt.ylabel("Cantidad de Registros")
    plot_path_dist = os.path.join(output_dir, "distribucion_outcome.png")
    plt.savefig(plot_path_dist, bbox_inches='tight')
    plt.close() 
    print(f" Gráfico de distribución guardado en: {plot_path_dist}")

    # Gráfico 2: Mapa de calor de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Correlación entre variables")
    plot_path_corr = os.path.join(output_dir, "heatmap_correlacion.png")
    plt.savefig(plot_path_corr, bbox_inches='tight')
    plt.close()
    print(f" Mapa de calor de correlación guardado en: {plot_path_corr}")
    
    print("\n✨ EDA completado con éxito.\n")


# ---- PREPROCESAMIENTO REPRODUCIBLE ----
def load_and_clean_data(filepath):
    """Carga el dataset y reemplaza los valores 0 inválidos por la mediana."""
    df = pd.read_csv(filepath)
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        
    return df

# ---- PREPARACIÓN DE SPLITS ESTRATIFICADOS ----
def prepare_splits(df, target_col="Outcome", test_size=0.2, random_state=42):
    """Separa las características del target y realiza el split estratificado."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---- FLUJO PRINCIPAL DE EJECUCIÓN ----
if __name__ == "__main__":
    # Ruta de tus datos reales
    DATA_PATH = "data/diabetes.csv"
    df_clean = load_and_clean_data(DATA_PATH)
    perform_eda(df_clean, output_dir="static")

    X_train, X_test, y_train, y_test = prepare_splits(df_clean)
    print(f"Splits listos. Entrenamiento: {X_train.shape[0]} muestras, Prueba: {X_test.shape[0]} muestras.")