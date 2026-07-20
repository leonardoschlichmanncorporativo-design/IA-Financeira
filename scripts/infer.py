"""
Script de inferência simples: carrega um modelo salvo e gera previsões para um CSV de entrada.
"""
import argparse
import os
import joblib
import pandas as pd


def main(model_path, input_csv, output_csv):
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    X = df.select_dtypes(include=["number"]).fillna(df.median())
    preds = model.predict(X)
    result = df.copy()
    result["prediction"] = preds
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz inferência usando um modelo salvo")
    parser.add_argument("--model", required=True, help="Caminho para o modelo salvo (p.ex. models/model.pkl)")
    parser.add_argument("--input", required=True, help="CSV de entrada para gerar previsões")
    parser.add_argument("--output", required=True, help="CSV de saída com previsões")
    args = parser.parse_args()

    main(args.model, args.input, args.output)
