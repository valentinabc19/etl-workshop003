# ***Workshop 003 - Machine Learning Regression Model***
This project aims to utilize 5 datasets containing happiness score information from different countries over the years to train a machine learning regression model that predicts happiness scores based on given data.

The project analyzes factors influencing the global happiness index score, focusing on comparing the predictive value of countries versus regions.

Additionally, a data streaming pipeline using Kafka is implemented to transmit the training data of the selected model, ultimately loading the selected features, predicted happiness score, and actual happiness score into a database.

## ***Technologies Used***

- Python
- Jupiter Notebook
- PostgreSQL
- Kafka
- CSV files
- Docker
- Git and GitHub

***Python dependencies***

- Pandas
- Matplotlib
- NumPy
- Seaborn
- Scikit-learn
- Psycopg2

## ***Project Structure***

```bash
.
├── data
│   ├── 2015.csv
│   ├── 2016.csv
│   ├── 2017.csv
│   ├── 2018.csv
│   ├── 2019.csv
│   └── merged_data.csv
├── kafka_pipeline
│   ├── __pycache__/
│   ├── feature_selection.py
│   ├── kafka_consumer.py
│   └── kafka_producer.py
├── model
│   └── trained_model.pkl
├── notebooks
│   ├── 001_merge_eda.ipynb
│   └── 002_model_training.ipynb
├── docker-compose.yml
├── venv
├── .gitignore
├── credentials.json
├── docker-compose.yml
├── main.py
├── README.md
└── requirements.txt
```
## ***Workflow***

This workshop was developed in two main work streams:

- Using notebooks for EDA and model training
- Using Python scripts to implement the Kafka data streaming pipeline and make predictions with the chosen model

### ***Exploratory Data Analysis - Happiness Data***

The EDA process involved several key steps:

#### ***Data Preparation and Cleaning***

- Standardization of country and region names
- Handling missing values
- Feature normalization
- Outlier detection and treatment

#### ***Feature Selection***

Based on the analysis, the following features were selected for ML model training:

- **gdp_per_capita**
- **social_support**
- **healthy_life_expectancy**
- **freedom**
- **region**
- **country**

Country and region variables were evaluated separately during training to determine the better predictor.

### ***Model Training and Selection***

Three models were evaluated using 5-fold cross-validation:

- Linear Regression
- Ridge Regression (with CV)
- Random Forest Regressor

Despite country-based models showing better performance, the region-based Random Forest model (R² = 0.8337) was selected for better generalization capability.

### ***Kafka Data Streaming Pipeline***

The pipeline consists of three main components:

#### ***Feature Selection Module***

Handles data preparation, cleaning, and test set generation.

#### ***Kafka Producer***

Streams test set records to the Kafka topic for real-time processing.

#### ***Kafka Consumer***

Processes incoming data, makes predictions, and stores results in PostgreSQL.

## ***Setup Instructions***

### ***Clone the repository***

Execute the following command to clone the repository

```bash
git clone https://github.com/valentinabc19/etl_workshop003

```
> From this point on all processes are done in Visual Studio Code

### ***Create Virtual Environment***
```bash
python -m venv venv
source venv/bin/activate  #On Windows: venv\Scripts\activate
```

### ***Credentials***
To make a connection to the database you must have the database credentials in a JSON file called credentials. So this file must be created in the project folder, with the following syntax:

```bash
{
    "dbname": "DB_NAME",
    "user": "DB_USER",
    "password": "DB_PASSWORD",
    "host": "DB_HOST",
    "port": "DB_PORT"    
}
```
>Ensure this file is included in `.gitignore`.

### ***Installing the dependencies***
The necessary dependencies are stored in a file named requirements.txt. To install the dependencies you can use the command
```bash
pip install -r requirements.txt
```


## ***How to Run***

### ***Initialize kafka***

Open a terminal in Visual Studio Code and start docker
```bash
docker-compose up -d --build
```
Enter the kafka-test container using this command
```bash
docker exec -it kafka-test bash
```
Create a new topic
```bash
kafka-topics --bootstrap-server kafka-test:9092 --create --topic ml-features
```
Then run this command in a terminal to initialize the consumer
```bash
kafka-console-consumer --bootstrap-server kafka-test:9092 --topic ml-features --property print.offset=true
```

Open a new terminal located in the project root folder and run this command to initialize the producer
```bash
python3 main.py
```
