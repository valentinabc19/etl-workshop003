import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Tuple

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))

def load_and_clean_yearly_data(file_path: str, year: int, drop_cols: list, rename_dict: dict) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df = df.drop(columns=drop_cols)
    df = df.rename(columns= rename_dict)
    df["year"] = year
    return df

def preliminar_eda_and_merge(data_dir: str = "data") -> pd.DataFrame:
    """Carga, limpia y combina los datasets de felicidad de 2015 a 2019.

    Parameters
    ----------
    data_dir : str
        Directorio que contiene los archivos CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame combinado y preprocesado.
    """
    rename_dict = {
        "country or region": "country",
        "score": "happiness_score",
        "happiness score": "happiness_score",
        "happiness.score": "happiness_score",
        "economy (gdp per capita)": "gdp_per_capita",
        "gdp per capita": "gdp_per_capita",
        "economy..gdp.per.capita.": "gdp_per_capita",
        "family": "social_support",
        "social support": "social_support",
        "health (life expectancy)": "healthy_life_expectancy",
        "healthy life expectancy": "healthy_life_expectancy",
        "health..life.expectancy.": "healthy_life_expectancy",
        "freedom to make life choices": "freedom",
        "trust (government corruption)": "perceptions_of_corruption",
        "trust..government.corruption.": "perceptions_of_corruption",
        "perceptions of corruption": "perceptions_of_corruption"
    }

    drop_dict = {
        2015: ["happiness rank", "standard error", "dystopia residual", "region"],
        2016: ["happiness rank", "lower confidence interval", "upper confidence interval", "dystopia residual", "region"],
        2017: ["happiness.rank", "whisker.high", "whisker.low", "dystopia.residual"],
        2018: ["overall rank"],
        2019: ["overall rank"]
    }

    dataframes = []
    for year in range(2015, 2020):
        file_path = os.path.join(ROOT_DIR, data_dir, f"{year}.csv")
        df_year = load_and_clean_yearly_data(file_path, year, drop_dict[year], rename_dict)
        dataframes.append(df_year)

    df = pd.concat(dataframes, ignore_index=True)
    return df

def data_transformations(df: pd.DataFrame) -> pd.DataFrame:

    df["country"] = df["country"].str.lower()

    country_replacements = {
    'macedonia': 'north macedonia',
    'trinidad & tobago': 'trinidad and tobago',
    'taiwan province of china': 'taiwan',
    'north cyprus': 'northern cyprus',
    'hong kong s.a.r., china': 'hong kong'
    }

    df['country'] = df['country'].replace(country_replacements)

    df['perceptions_of_corruption'] = df.groupby('country')['perceptions_of_corruption'].transform(lambda x: x.fillna(x.median()))

    return df

def prepare_test_set(test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa y retorna únicamente el conjunto de prueba (X_test, y_test) a partir del dataframe dado.
    Útil para casos en los que el conjunto de entrenamiento será procesado por otro sistema (e.g. Kafka).

    Parámetros
    ----------

    test_size : float, optional
        Proporción del conjunto de datos usado como test (por defecto 0.2).

    random_state : int, optional
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    df = preliminar_eda_and_merge()
    df = data_transformations(df)

    X = df.drop(['happiness_score', 'perceptions_of_corruption', 'generosity', 'year'], axis=1)
    y = df['happiness_score']

    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_test, y_test

