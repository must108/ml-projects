from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler

samples = pd.read_csv('unsupervised_learning/wine.csv')

model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)

df = pd.DataFrame({'labels': labels,
                   'varieties': varieties})

# feature variance = feature influence on dataset

# scaler code:
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy = True, with_mean = True, with_std = True)
samples_scaled = scaler.transform(samples)

# with pipeline:
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
