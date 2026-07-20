"""
Treinamento de modelo de exemplo usando um dataset CSV (ex.: california housing).
Salva o modelo treinado em disco.
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(path):
    df = pd.read_csv(path)
    return df


def default_preprocess(df):
    # Remover colunas não numéricas simples e preencher NA
    df = df.select_dtypes(include=["number"]).copy()
    df = df.fillna(df.median())
    return df


def train(train_csv, output_path, test_size=0.2, random_state=42):
    df = load_data(train_csv)
    df = default_preprocess(df)

    if "median_house_value" in df.columns:
        target_col = "median_house_value"
    else:
        # Escolhe a última coluna numérica como target por convenção
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    print(f"Validation MSE: {mse:.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Modelo salvo em {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo de exemplo")
    parser.add_argument("--train-csv", required=True, help="Caminho para o CSV de treino")
    parser.add_argument("--output", required=True, help="Caminho para salvar o modelo (p.ex. models/model.pkl)")
    args = parser.parse_args()

    train(args.train_csv, args.output)
