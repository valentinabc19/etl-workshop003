from kafka import KafkaConsumer
import json
import os
import pandas as pd
import psycopg2
import joblib

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))
CREDENTIALS_PATH = os.path.join(ROOT_DIR, "credentials.json")  
def connect_postgres():
    """Establish a connection to a PostgreSQL database using credentials from a JSON file.

        Returns
        -------
        psycopg2.connection
            A connection object to the PostgreSQL database.

        Raises
        ------
        FileNotFoundError
            If the credentials file is not found at the specified path.
        KeyError
            If required credentials (db_host, db_name, db_user, db_password) are missing in the JSON file.
        psycopg2.Error
            If the connection to the PostgreSQL database fails.
    """
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
    """
    Run a Kafka consumer to process messages from the 'ml-features' topic, make predictions using a pre-trained model,
    and save the results to a PostgreSQL database.

    The function loads a pre-trained model, consumes messages containing feature data and true happiness scores,
    predicts happiness scores, and stores both true and predicted values in a database.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the trained model file is not found at the specified path.
        KafkaError
            If there is an error connecting to or consuming from the Kafka topic.
    """
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

        X_input = pd.DataFrame([data])
        y_pred = model.predict(X_input)[0]

        true_values.append(y_true)
        predicted_values.append(y_pred)

        save_to_postgres(data, y_true, y_pred)

        print(f"Guardado en BD - Prediccion: {y_pred}")

def save_to_postgres(features_dict, y_true, y_pred):
    """
    Save prediction results along with features to a PostgreSQL database.

        Parameters
        ----------
        features_dict : dict
            Dictionary containing the feature values (region, gdp_per_capita, social_support, healthy_life_expectancy, freedom).
        y_true : float
            The actual happiness score.
        y_pred : float
            The predicted happiness score.

        Returns
        -------
        None

        Raises
        ------
        psycopg2.Error
            If there is an error executing the SQL commands or connecting to the database.
    """
    conn = connect_postgres()
    cursor = conn.cursor()

    
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
