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
        2015: ["happiness rank", "standard error", "dystopia residual"],
        2016: ["happiness rank", "lower confidence interval", "upper confidence interval", "dystopia residual"],
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

    region_replacements = {
    # Australia and New Zealand
    'australia': 'australia and new zealand',
    'new zealand': 'australia and new zealand',
    
    # Central and Eastern Europe
    'albania': 'central and eastern europe',
    'armenia': 'central and eastern europe',
    'azerbaijan': 'central and eastern europe',
    'belarus': 'central and eastern europe',
    'bosnia and herzegovina': 'central and eastern europe',
    'bulgaria': 'central and eastern europe',
    'croatia': 'central and eastern europe',
    'czech republic': 'central and eastern europe',
    'estonia': 'central and eastern europe',
    'georgia': 'central and eastern europe',
    'hungary': 'central and eastern europe',
    'kazakhstan': 'central and eastern europe',
    'kosovo': 'central and eastern europe',
    'kyrgyzstan': 'central and eastern europe',
    'latvia': 'central and eastern europe',
    'lithuania': 'central and eastern europe',
    'moldova': 'central and eastern europe',
    'montenegro': 'central and eastern europe',
    'north macedonia': 'central and eastern europe',
    'poland': 'central and eastern europe',
    'romania': 'central and eastern europe',
    'russia': 'central and eastern europe',
    'serbia': 'central and eastern europe',
    'slovakia': 'central and eastern europe',
    'slovenia': 'central and eastern europe',
    'tajikistan': 'central and eastern europe',
    'turkmenistan': 'central and eastern europe',
    'ukraine': 'central and eastern europe',
    'uzbekistan': 'central and eastern europe',
    
    # Eastern Asia
    'china': 'eastern asia',
    'hong kong': 'eastern asia',
    'japan': 'eastern asia',
    'mongolia': 'eastern asia',
    'south korea': 'eastern asia',
    'taiwan': 'eastern asia',
    
    # Latin America and Caribbean
    'argentina': 'latin america and caribbean',
    'belize': 'latin america and caribbean',
    'bolivia': 'latin america and caribbean',
    'brazil': 'latin america and caribbean',
    'chile': 'latin america and caribbean',
    'colombia': 'latin america and caribbean',
    'costa rica': 'latin america and caribbean',
    'dominican republic': 'latin america and caribbean',
    'ecuador': 'latin america and caribbean',
    'el salvador': 'latin america and caribbean',
    'guatemala': 'latin america and caribbean',
    'haiti': 'latin america and caribbean',
    'honduras': 'latin america and caribbean',
    'jamaica': 'latin america and caribbean',
    'mexico': 'latin america and caribbean',
    'nicaragua': 'latin america and caribbean',
    'panama': 'latin america and caribbean',
    'paraguay': 'latin america and caribbean',
    'peru': 'latin america and caribbean',
    'puerto rico': 'latin america and caribbean',
    'suriname': 'latin america and caribbean',
    'trinidad and tobago': 'latin america and caribbean',
    'uruguay': 'latin america and caribbean',
    'venezuela': 'latin america and caribbean',
    
    # Middle East and Northern Africa
    'algeria': 'middle east and northern africa',
    'bahrain': 'middle east and northern africa',
    'egypt': 'middle east and northern africa',
    'iran': 'middle east and northern africa',
    'iraq': 'middle east and northern africa',
    'israel': 'middle east and northern africa',
    'jordan': 'middle east and northern africa',
    'kuwait': 'middle east and northern africa',
    'lebanon': 'middle east and northern africa',
    'libya': 'middle east and northern africa',
    'morocco': 'middle east and northern africa',
    'oman': 'middle east and northern africa',
    'palestinian territories': 'middle east and northern africa',
    'qatar': 'middle east and northern africa',
    'saudi arabia': 'middle east and northern africa',
    'syria': 'middle east and northern africa',
    'tunisia': 'middle east and northern africa',
    'turkey': 'middle east and northern africa',
    'united arab emirates': 'middle east and northern africa',
    'yemen': 'middle east and northern africa',
    
    # North America
    'canada': 'north america',
    'united states': 'north america',
    
    # Southeastern Asia
    'cambodia': 'southeastern asia',
    'indonesia': 'southeastern asia',
    'laos': 'southeastern asia',
    'malaysia': 'southeastern asia',
    'myanmar': 'southeastern asia',
    'philippines': 'southeastern asia',
    'singapore': 'southeastern asia',
    'thailand': 'southeastern asia',
    'vietnam': 'southeastern asia',
    
    # Southern Asia
    'afghanistan': 'southern asia',
    'bangladesh': 'southern asia',
    'bhutan': 'southern asia',
    'india': 'southern asia',
    'nepal': 'southern asia',
    'pakistan': 'southern asia',
    'sri lanka': 'southern asia',
    
    # Sub-Saharan Africa
    'angola': 'sub-saharan africa',
    'benin': 'sub-saharan africa',
    'botswana': 'sub-saharan africa',
    'burkina faso': 'sub-saharan africa',
    'burundi': 'sub-saharan africa',
    'cameroon': 'sub-saharan africa',
    'central african republic': 'sub-saharan africa',
    'chad': 'sub-saharan africa',
    'comoros': 'sub-saharan africa',
    'congo (brazzaville)': 'sub-saharan africa',
    'congo (kinshasa)': 'sub-saharan africa',
    'djibouti': 'sub-saharan africa',
    'ethiopia': 'sub-saharan africa',
    'gabon': 'sub-saharan africa',
    'ghana': 'sub-saharan africa',
    'guinea': 'sub-saharan africa',
    'ivory coast': 'sub-saharan africa',
    'kenya': 'sub-saharan africa',
    'lesotho': 'sub-saharan africa',
    'liberia': 'sub-saharan africa',
    'madagascar': 'sub-saharan africa',
    'malawi': 'sub-saharan africa',
    'mali': 'sub-saharan africa',
    'mauritania': 'sub-saharan africa',
    'mauritius': 'sub-saharan africa',
    'mozambique': 'sub-saharan africa',
    'namibia': 'sub-saharan africa',
    'niger': 'sub-saharan africa',
    'nigeria': 'sub-saharan africa',
    'rwanda': 'sub-saharan africa',
    'senegal': 'sub-saharan africa',
    'sierra leone': 'sub-saharan africa',
    'somalia': 'sub-saharan africa',
    'somaliland region': 'sub-saharan africa',
    'south africa': 'sub-saharan africa',
    'south sudan': 'sub-saharan africa',
    'sudan': 'sub-saharan africa',
    'swaziland': 'sub-saharan africa',
    'tanzania': 'sub-saharan africa',
    'togo': 'sub-saharan africa',
    'uganda': 'sub-saharan africa',
    'zambia': 'sub-saharan africa',
    'zimbabwe': 'sub-saharan africa',
    
    # Western Europe
    'austria': 'western europe',
    'belgium': 'western europe',
    'cyprus': 'western europe',
    'denmark': 'western europe',
    'finland': 'western europe',
    'france': 'western europe',
    'germany': 'western europe',
    'greece': 'western europe',
    'iceland': 'western europe',
    'ireland': 'western europe',
    'italy': 'western europe',
    'luxembourg': 'western europe',
    'malta': 'western europe',
    'netherlands': 'western europe',
    'northern cyprus': 'western europe',
    'norway': 'western europe',
    'portugal': 'western europe',
    'spain': 'western europe',
    'sweden': 'western europe',
    'switzerland': 'western europe',
    'united kingdom': 'western europe'
    }

    df['region'] = df['country'].str.lower().map(region_replacements)

    df['perceptions_of_corruption'] = df.groupby('country')['perceptions_of_corruption'].transform(lambda x: x.fillna(x.median()))

    df = df.dropna()

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

    X = df.drop(['happiness_score', 'perceptions_of_corruption', 'generosity', 'year', 'country'], axis=1)
    y = df['happiness_score']

    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_test, y_test

