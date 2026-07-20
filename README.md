# IA Financeira

Este repositório contém código inicial para um projeto de IA aplicada a problemas financeiros e de previsão. Inclui dados de amostra em `sample_data/`.

Objetivo imediato

- Fornecer uma estrutura mínima para treinar um modelo de regressão sobre dados de exemplo.

Como usar (exemplo rápido)

1. Crie um ambiente virtual e instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

2. Treinar o modelo de exemplo usando o dataset de habitação (exemplo):

```bash
python src/ia_financeira/train.py --train-csv sample_data/california_housing_train.csv --output models/model.pkl
```

3. Fazer previsões com o modelo treinado:

```bash
python scripts/infer.py --model models/model.pkl --input sample_data/california_housing_test.csv --output predictions.csv
```

Próximos passos sugeridos

- Adaptar os scripts para dados financeiros (preços, séries temporais, features econômicas).
- Adicionar testes unitários e CI (GitHub Actions).
- Documentar os pipelines de dados e objetivos de negócio.

