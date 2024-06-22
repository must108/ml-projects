from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

samples = 1

iris = load_iris()
model = KMeans(n_clusters=3)
model.fit(samples)
labels = model.predict(samples)

new_samples = 1
new_labels = model.predict(new_samples)

xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()