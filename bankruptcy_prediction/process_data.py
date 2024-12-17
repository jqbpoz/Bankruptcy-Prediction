from docutils.nodes import label

from bankruptcy_prediction.dataset_methods import load_data, clean_data, label_data, save_data

if __name__ == "__main__":
    raw_data_path = "../data/raw/dataset.csv"
    df = load_data(raw_data_path)
    df_labeled = label_data(df)
    save_data(df_labeled, "../data/processed/dataset_labeled.csv")


