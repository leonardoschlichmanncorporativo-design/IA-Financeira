#!/usr/bin/env python3
"""
Treina um modelo baseline (RandomForestRegressor) com sample_data/california_housing_train.csv
Saída: imprime MAE e salva modelo em models/rf_baseline.joblib
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = Path("sample_data/california_housing_train.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "rf_baseline.joblib"

if not DATA_PATH.exists():
    print(f"Erro: arquivo de dados não encontrado em {DATA_PATH}", file=sys.stderr)
    sys.exit(1)

print("Carregando dados de:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Colunas detectadas:", list(df.columns))

# Detectar coluna alvo comum
target_candidates = ["median_house_value", "median_house_value.", "median_house_value\n", "target", "median_value"]
target_col = next((c for c in df.columns if c in target_candidates), None)
if target_col is None:
    # heurística: última coluna numeric pode ser target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 1:
        target_col = numeric_cols[-1]
    else:
        raise RuntimeError("Não encontrei coluna alvo no CSV")

print("Usando coluna alvo:", target_col)
X = df.drop(columns=[target_col])
y = df[target_col]

# Simples pré-processamento
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_features:
    raise RuntimeError("Não foram encontradas colunas numéricas para treinar")

X_num = X[numeric_features].copy()

imputer = SimpleImputer(strategy="median")
X_num_imputed = imputer.fit_transform(X_num)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_imputed)

X_prepared = X_num_scaled

X_train, X_test, y_train, y_test = train_test_split(
    X_prepared, y, test_size=0.2, random_state=42
)

print("Treinando RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Avaliando...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE (RandomForest baseline): {mae:.4f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "imputer": imputer, "scaler": scaler, "features": numeric_features}, MODEL_PATH)
print(f"Modelo salvo em: {MODEL_PATH}")
