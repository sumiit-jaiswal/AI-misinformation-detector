misinfo-env\Scripts\activate     # Windows


run using this
python -m src.model_training.data_preprocessing
python -m src.model_training.sarcasm_sentiment.train_sarcasm

python src/data_pipeline/api_clients/news_api.py

uvicorn main:app --reload

pip freeze > requirements.txt 

# Train BERT
python src/model_training/train_llm.py --model-type bert

# Train GPT-2
python src/model_training/train_llm.py --model-type gpt2

python -m src.model_training.train_llm --model-type bert
python -m src.model_training.train_llm --model-type gpt2
