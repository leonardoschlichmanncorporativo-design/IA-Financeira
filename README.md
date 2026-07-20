# IA Financeira

Este repositório fornece conjuntos de dados de exemplo (California housing, MNIST e Anscombe) destinados a experimentos de ML/IA relacionados ao projeto "IA Financeira".

Este commit adiciona um README com instruções básicas e um script de exemplo para treinar um modelo simples usando o dataset `california_housing_train.csv` presente em `sample_data/`.

## Uso rápido

1. Clone o repositório:

```bash
git clone https://github.com/leonardoschlichmanncorporativo-design/IA-Financeira.git
cd IA-Financeira
```

2. (Opcional) crie um ambiente virtual e instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

3. Execute o script de exemplo que treina um modelo baseline:

```bash
python train_baseline.py
```

O script carrega `sample_data/california_housing_train.csv`, treina um RandomForestRegressor simples e imprime o erro absoluto médio (MAE). Ele também salva o modelo em `models/rf_baseline.joblib`.

## Conteúdo relevante

- `sample_data/` — datasets (CSV/JSON)
- `train_baseline.py` — script de exemplo para treinar um modelo baseline
- `requirements.txt` — dependências mínimas para rodar o script

---

License: MIT
