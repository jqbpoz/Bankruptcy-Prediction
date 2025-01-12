from docutils.nodes import label
from sympy.physics.quantum.gate import normalized

from bankruptcy_prediction.dataset_methods import load_data, clean_data, label_data, save_data, normalize, \
    extract_significant_correlations, remove_outliers

if __name__ == "__main__":
    raw_data_path = "../data/raw/dataset.csv"
    df = load_data(raw_data_path)
    df_labeled = label_data(df)
    save_data(df_labeled, "../data/processed/dataset_raw_labeled.csv")

    #1. Clean the data
    df_normalized = normalize(df_labeled,['Bankrupt'])
    save_data(df_normalized, "../data/processed/dataset_normalized.csv")

    #2. Remove correlated features
    df_correlation_reduced = df_normalized.copy()
    corr_matrix = df_correlation_reduced.corr()
    corr_pairs_df = extract_significant_correlations(corr_matrix, threshold=0.35)
    to_remove_unique = corr_pairs_df['Variable 1'].unique()
    df_correlation_reduced = df_correlation_reduced.drop(columns=to_remove_unique)
    save_data(df_correlation_reduced, "../data/processed/dataset_correlation_reduced.csv")












