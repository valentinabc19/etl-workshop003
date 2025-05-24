from kafka import KafkaProducer
import pandas as pd
import json
import time

def run_producer(X_test: pd.DataFrame, y_test: pd.Series):
    """
    Sends test data (features and actual values) to a Kafka topic.

        Parameters
        ----------
        X_test : pd.DataFrame
            Feature set to be used for prediction.
        y_test : pd.Series
            Actual values corresponding to X_test.
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
        time.sleep(0.1)
        print("message sent")

    producer.flush()
    producer.close()
