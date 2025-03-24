import pandas as pd

def load_filtered_data(filepath):
    """
    Loads a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath)