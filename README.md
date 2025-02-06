misinfo-env\Scripts\activate     # Windows


run using this
python -m src.model_training.data_preprocessing
python -m src.model_training.sarcasm_sentiment.train_sarcasm

python src/data_pipeline/api_clients/news_api.py

uvicorn main:app --reload