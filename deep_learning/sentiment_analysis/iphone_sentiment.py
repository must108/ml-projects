import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout
import pickle

df = pd.read_csv(
    "deep_learning/sentiment_analysis/text_data/iphone.csv"
)

df["review"] = df["reviewTitle"] + " " + df["reviewDescription"]
df = df[["ratingScore", "review"]]
df["sentiment"] = df["ratingScore"].apply(
    lambda x: "positive" if x > 3 else "negative" if x < 3 else "neutral"
)

df = df[["review", "sentiment"]]
df = df.sample(frac=1).reset_index(drop=True)
df["review"] = df["review"].astype(str)

print(df.head())

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["review"])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df["review"])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating="post")

sentiment_labels = pd.get_dummies(df["sentiment"]).values

x_train, x_test, y_train, y_test = train_test_split(
    padded_sequences, sentiment_labels, test_size=0.2
)

# create neural network
model = Sequential()
model.add(Embedding(5000, 100, input_length=100))
model.add(Conv1D(64, 5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy",
                        metrics=["accuracy"])
model.summary()

model.fit(x_train, y_train, epochs=10, batch_size=32,
          validation_data=(x_test, y_test))
y_pred = np.argmax(model.predict(x_test), axis=-1)
print("Accuracy: ", accuracy_score(np.argmax(y_test, axis=-1), y_pred))

model.save("sentiment_analysis_model_2.h5")
with open("tokenizer2.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
