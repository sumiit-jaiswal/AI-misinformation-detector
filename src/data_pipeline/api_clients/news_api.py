# news_api.py
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os

load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))

def fetch_news(query="misinformation"):
    articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy")
    return articles["articles"]