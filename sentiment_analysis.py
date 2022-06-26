import pandas as pd
from pathlib import Path
import ipywidgets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

analyst_ratings_processed = pd.read_csv ('D:/DATASETS/PROGRAMES_SCRAPPERS/bitmex_historical_scraper/DATA/BTC_Tweets/Bitcoin_tweets.csv')
analyst_ratings_processed['user_description'].fillna("", inplace=True)

positive = []
negative = []
neutral = []
for headline in analyst_ratings_processed['user_description']:
    inputs = tokenizer(headline, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive.append(predictions.detach().numpy()[0][0])
    negative.append(predictions.detach().numpy()[0][1])
    neutral.append(predictions.detach().numpy()[0][2])
    print(index)

analyst_ratings_processed['positive'] = positive
analyst_ratings_processed['negative'] = negative
analyst_ratings_processed['neutral'] = neutral

analyst_ratings_processed.head()

analyst_ratings_processed.to_csv('BTC_sentiment.csv', index=False)
