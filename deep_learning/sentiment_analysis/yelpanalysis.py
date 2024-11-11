from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import math


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors="pt")
    result = model(tokens)
    return int(torch.argmax(result.logits))+1  # returns a score from 1-5
# function for calculating sentiment score


# loading hugging face models
tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

link = input("Enter the link to the Yelp business: ")

r = requests.get(link + "#reviews")
soup = BeautifulSoup(r.text, "html.parser")
regex = re.compile(".*comment.*")
results = soup.find_all("p", {"class": regex})
reviews = [
    result.find("span").text for result in results if result.find("span")
]

df = pd.DataFrame(np.array(reviews), columns=["review"])
df["sentiment"] = df["review"].apply(lambda x: sentiment_score(x[:512]))
avg = df["sentiment"].mean()
rounded_avg = round(avg, 2 - int(math.floor(math.log10(abs(avg)))))

print("Based on the reviews, we rate the restaurant: ",
      rounded_avg, "/ 5 stars!")
