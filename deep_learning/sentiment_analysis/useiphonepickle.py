import keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model(
    "deep_learning/sentiment_analysis/sentiment_analysis_model_2.h5"
)
with open(
    "deep_learning/sentiment_analysis/tokenizer2.pickle", "rb"
) as handle:
    tokenizer = pickle.load(handle)


def predict_sentiment(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)
    predicted_rating = model.predict(text_sequence)[0]
    if np.argmax(predicted_rating) == 0:
        return "Negative"
    elif np.argmax(predicted_rating) == 1:
        return "Neutral"
    else:
        return "Positive"


text_input = "I hated this iPhone. It sucks."
predicted_sentiment = predict_sentiment(text_input)
print(predicted_sentiment)

text_input = "This iPhone is average, nothing too much."
predicted_sentiment = predict_sentiment(text_input)
print(predicted_sentiment)

text_input = "This iPhone is amazing! I absolutely love it!"
predicted_sentiment = predict_sentiment(text_input)
print(predicted_sentiment)
