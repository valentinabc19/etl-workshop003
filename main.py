from kafka_pipeline.kafka_producer import run_producer
from kafka_pipeline.kafka_consumer import run_consumer
from kafka_pipeline.feature_selection import prepare_test_set

if __name__ == "__main__":

    X_test, y_test = prepare_test_set()
    run_producer(X_test, y_test)
    run_consumer()