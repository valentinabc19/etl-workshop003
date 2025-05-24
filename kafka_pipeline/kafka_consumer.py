from kafka import KafkaConsumer
import json
import os
import pandas as pd
import psycopg2
import joblib

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))
CREDENTIALS_PATH = os.path.join(ROOT_DIR, "credentials.json")  
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

        save_to_postgres(data, y_true, y_pred)

        print(f"Guardado en BD - Prediccion: {y_pred}")

# Guardado en PostgreSQL con columnas individuales
def save_to_postgres(features_dict, y_true, y_pred):
    conn = connect_postgres()
    cursor = conn.cursor()

    # Crear tabla si no existe (ajustada con columnas específicas)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        region TEXT,
        gdp_per_capita REAL,
        social_support REAL,
        healthy_life_expectancy REAL,
        freedom REAL,
        y_true REAL,
        y_pred REAL
    );
    """)

    cursor.execute("""
        INSERT INTO predictions (
            region, gdp_per_capita, social_support,
            healthy_life_expectancy, freedom, y_true, y_pred
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        str(features_dict.get('region')),
        float(features_dict.get('gdp_per_capita')),
        float(features_dict.get('social_support')),
        float(features_dict.get('healthy_life_expectancy')),
        float(features_dict.get('freedom')),
        float(y_true),
        float(y_pred)
    ))

    conn.commit()
    cursor.close()
    conn.close()
