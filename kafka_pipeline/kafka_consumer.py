from kafka import KafkaConsumer
import json
import os
import pandas as pd
import psycopg2
import joblib
from sklearn.metrics import r2_score

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../"))
CREDENTIALS_PATH = os.path.join(ROOT_DIR, "credentials.json")   
# Conexión a PostgreSQL
def connect_postgres():
    with open (CREDENTIALS_PATH, "r", encoding="utf-8") as file:
        credentials = json.load(file)
        
    db_host = credentials["db_host"]
    db_name = credentials["db_name"]
    db_user = credentials["db_user"]
    db_password = credentials["db_password"]

    conn = psycopg2.connect(
        host=db_host,
        dbname=db_name,
        user=db_user,
        password=db_password
    )

    return conn

def run_consumer():
    consumer = KafkaConsumer(
        'ml-features',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='ml-consumer-group'
    )

    model_path = os.path.join(ROOT_DIR, "model", "trained_model.pkl")
    model = joblib.load(model_path)

    true_values = []
    predicted_values = []

    for message in consumer:
        data = message.value

        y_true = data.pop('true_score')

        # Convertir en DataFrame para predicción
        X_input = pd.DataFrame([data])
        y_pred = model.predict(X_input)[0]

        true_values.append(y_true)
        predicted_values.append(y_pred)

        r2 = r2_score(true_values, predicted_values)

        save_to_postgres(data, y_true, y_pred, r2)

        print(f"Guardado en BD - R² acumulado: {r2:.4f}")

# Guardado en PostgreSQL con columnas individuales
def save_to_postgres(features_dict, y_true, y_pred, r2_value):
    conn = connect_postgres()
    cursor = conn.cursor()

    # Crear tabla si no existe (ajustada con columnas específicas)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        country TEXT,
        gdp_per_capita REAL,
        social_support REAL,
        healthy_life_expectancy REAL,
        freedom REAL,
        y_true REAL,
        y_pred REAL,
        r2 REAL
    );
    """)

    cursor.execute("""
        INSERT INTO predictions (
            country, gdp_per_capita, social_support,
            healthy_life_expectancy, freedom, y_true, y_pred, r2
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        features_dict.get('country'),
        features_dict.get('gdp_per_capita'),
        features_dict.get('social_support'),
        features_dict.get('healthy_life_expectancy'),
        features_dict.get('freedom'),
        y_true,
        y_pred,
        r2_value
    ))

    conn.commit()
    cursor.close()
    conn.close()
