from bankruptcy_prediction.dataset import load_data, clean_data

if __name__ == "__main__":
    raw_data_path = "../data/raw/dataset.csv"
    df = load_data(raw_data_path)

    df_cleaned = clean_data(df)

    processed_data_path = "../data/processed/dataset_cleaned.csv"
    df_cleaned.to_csv(processed_data_path, index=False)
    print(f"Przetworzone dane zapisane w: {processed_data_path}")
