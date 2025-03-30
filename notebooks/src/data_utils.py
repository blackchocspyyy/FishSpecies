import pandas as pd
from typing import List, Dict

def load_fish_csvs(file_list: List[str], base_path: str = "../data/") -> Dict[str, pd.DataFrame]:
    """
    Load multiple hydroacoustic fish detection CSVs into separate Pandas DataFrames.

    Args:
        file_list (List[str]): List of CSV filenames.
        base_path (str): Path to the folder containing the CSVs.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping file names (without .csv) to their DataFrames.
    """
    dataframes = {}
    for dataset in file_list:
        var_name = dataset.replace(".csv", "")
        dataframes[var_name] = pd.read_csv(base_path + dataset)
    return dataframes


def merge_and_parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:

    """
    Merges date and time columns into a unified datetime column.

    Args:
        df (pd.DataFrame): Raw dataframe with 'dateProcessed' and 'Ping_time' columns.

    Returns:
        pd.DataFrame: Updated DataFrame with combined 'Ping_time' as datetime.
    """
    df["dateProcessed"] = pd.to_datetime(df["dateProcessed"])
    df["Ping_time"] = pd.to_datetime(df["Ping_time"].str.strip(), format="%H:%M:%S.%f").dt.time
    df["Ping_time"] = df.apply(lambda row: pd.Timestamp.combine(row["dateProcessed"], row["Ping_time"]), axis=1)
    return df
