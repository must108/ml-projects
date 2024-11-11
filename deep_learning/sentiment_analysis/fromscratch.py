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
    "deep_learning/sentiment_analysis/text_data/tripadvisor_hotel_reviews.csv"
)

df = df[["Review", "Rating"]]  # gets only these two columns
df["sentiment"] = df["Rating"].apply(lambda x: "positive" if x > 3
                                     else "negative" if x < 3
                                     else "neutral")
# determines sentiment by checking review value

df = df[["Review", "sentiment"]]  # essentially drops rating
df = df.sample(frac=1).reset_index(drop=True)
# randomly distributes data for better use

print(df.head())  # prints the first few lines

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
# converts words to integer sequences
tokenizer.fit_on_texts(df["Review"])
# creates vocab, where words are assigned integers based on freq.
word_index = tokenizer.word_index  # gets the dictionary of words to values
sequences = tokenizer.texts_to_sequences(df["Review"])
# converts reviews to integers
padded_sequences = pad_sequences(sequences, maxlen=100, truncating="post")
# pads integer sequences to have a length of 100

sentiment_labels = pd.get_dummies(df["sentiment"]).values
# makes it categorical

x_train, x_test, y_train, y_test = train_test_split(
    padded_sequences, sentiment_labels, test_size=0.2
)  # 80/20 train test split

# create neural network
model = Sequential()  # allows linear adding of layers
model.add(
    Embedding(5000, 100, input_length=100)
)  # converts word to 100-dimensional vector (embedding)
model.add(
    Conv1D(64, 5, activation="relu")
)  # adds convolutional layer with 64 filters, kernel size 5,
# and reLU activation function
model.add(GlobalMaxPooling1D())  # reduces data to a single vector
model.add(Dense(32, activation="relu"))  # fully connected layer with relu
model.add(Dropout(0.5))  # introduces regularization, prevents overfitting,
# randomly drops neurons
model.add(Dense(3, activation="softmax"))  # adds a layer with softmax function
model.compile(optimizer="adam", loss="categorical_crossentropy",
                        metrics=["accuracy"])
# compiles model ^ and prints summary V
model.summary()

# train model
model.fit(x_train, y_train, epochs=10, batch_size=32,
          validation_data=(x_test, y_test))

y_pred = np.argmax(model.predict(x_test), axis=-1)
print("Accuracy: ", accuracy_score(np.argmax(y_test, axis=-1), y_pred))

# save model for future use:
model.save('sentiment_analysis_model.h5')
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# opens a pickle file and dumps into it
