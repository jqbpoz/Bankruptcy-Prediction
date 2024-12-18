from nltk import accuracy
from sklearn.model_selection import train_test_split

import Model
from bankruptcy_prediction.dataset_methods import load_data
from bankruptcy_prediction.default_hyperparameters import DEFAULT_PARAMS_MAPPING

raw_data_path = "../data/processed/dataset_normalized.csv"
df = load_data(raw_data_path)



model = Model.Model(algorithm="LightGBM", **DEFAULT_PARAMS_MAPPING["LightGBM"])



X = df.drop(columns=['Bankrupt'])
y = df['Bankrupt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.train(X_train, y_train)

ev = model.evaluate(X_test, y_test)
print(ev['accuracy'])
print(ev['report'])


