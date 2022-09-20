import pandas as pd

# Read a csv file and show the records
features = pd.read_csv('data/house.price.csv')
features.head(10)

# We will use the K-Means algorithm
from sklearn.cluster import KMeans
# We will only consider 2 features and see if we get a pattern
cluster_Xs = features[['Area', 'Locality']]
# How many clusters we want to find
NUM_CLUSTERS = 3
# Build the K Means Clusters model
model = KMeans(n_clusters=NUM_CLUSTERS)