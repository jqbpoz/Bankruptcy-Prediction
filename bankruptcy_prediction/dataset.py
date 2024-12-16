import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


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

