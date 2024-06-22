import pandas as pd

labels = "hello"
species = "goodbye"
samples = 1
# sample variables for demonstration

df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

# inertia measures how spread out a cluster is/distance of each sample to the centroid
# a cluster of data shows patterns/trends without supervised metrics

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)

#########
# simple script for plotting inertia values!

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#########