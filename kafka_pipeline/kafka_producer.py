from kafka import KafkaProducer
import pandas as pd
import json
import time

def run_producer(X_test: pd.DataFrame, y_test: pd.Series):
    """
    Envía datos de test (features y valor real) a un tópico de Kafka.

    Parámetros
    ----------
    X_test : pd.DataFrame
        Conjunto de features a usar para predicción.
    y_test : pd.Series
        Valores reales correspondientes a X_test.
    """
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    topic = 'ml-features'

    for index, row in X_test.iterrows():
        message = row.to_dict()
        message['true_score'] = y_test.loc[index]
        producer.send(topic, message)
        time.sleep(0.1)  # Simula streaming
        print("message sent")

    producer.flush()
    producer.close()
