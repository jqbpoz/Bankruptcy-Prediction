import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_data(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path, index=False)

def label_data(df: pd.DataFrame) -> pd.DataFrame:
    df_labeled = df.copy()
    labels = ["Bankrupt"] + [f"X{i}" for i in range(1, 96)]
    df_labeled.columns = labels
    return df_labeled

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    len_before = len(df)
    df_cleaned = df.dropna()
    len_after = len(df_cleaned)
    logging.info(f"Rows before cleaning: {len_before}")
    logging.info(f"Rows after cleaning: {len_after}")
    return df_cleaned

def extract_significant_correlations(df: pd.DataFrame, threshold:float) -> pd.DataFrame:
    """
   Creates a DataFrame of correlations greater than a specified threshold.

   This function computes the correlation matrix for the given DataFrame and filters
   correlations that are greater than the specified threshold (`value`). It then extracts
   the upper triangle of the matrix (to avoid duplicates of the same correlation values)
   and stacks the remaining values into a new DataFrame with columns 'Variable 1',
   'Variable 2', and 'Correlation'.

   Args:
       df (pd.DataFrame): The input DataFrame containing numerical data to calculate correlations.
       value (float): The correlation threshold above which to keep the values.

   Returns:
       pd.DataFrame: A DataFrame containing the variable pairs and their correlation values
                      that are greater than the specified `value`.
   """
    corr_matrix_above = df[df > threshold]
    # Create a mask to only show the upper triangle
    upper_triangle_mask = np.triu(np.ones(corr_matrix_above.shape), k=1).astype(bool)
    # Apply the mask to the correlation matrix
    upper_triangle = corr_matrix_above.where(upper_triangle_mask)
    #Stack the upper triangle to create a list of correlations
    correlation_list = upper_triangle.stack().reset_index()
    correlation_list.columns = ['Variable 1', 'Variable 2', 'Correlation']
    return correlation_list

def remove_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes outliers from the specified columns of a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to check for outliers.

    Returns:
        pd.DataFrame: A DataFrame with rows containing outliers in the specified columns removed.
    """
    df_cleaned = df.copy()
    print(df_cleaned.columns)
    for column in columns:
        # Calculate Q1, Q3, and IQR for the column
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove rows with outliers in the current column
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

    return df_cleaned

def normalize(df: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    features = df.drop(columns=target_columns)
    targets = df[target_columns]

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    df_scaled = pd.concat([features_scaled, targets.reset_index(drop=True)], axis=1)
    return df_scaled

