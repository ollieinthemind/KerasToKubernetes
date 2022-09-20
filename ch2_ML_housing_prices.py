import pandas as pd
from sklearn.cluster import KMeans
# Read a csv file and show the records
features = pd.read_csv('data/house.price.csv')
features.head(10)

# We will use the K-Means algorithm

# We will only consider 2 features and see if we get a pattern
cluster_Xs = features[['Area', 'Locality']]
# How many clusters we want to find
NUM_CLUSTERS = 3
# Build the K Means Clusters model
model = KMeans(n_clusters=NUM_CLUSTERS)
model.fit(cluster_Xs)
# Predict and get cluster labels - 0, 1, 2 ... NUM_CLUSTERS
predictions = model.predict(cluster_Xs)
# Add predictions to the features data frame
features['cluster'] = predictions
features.head(10)

features_sorted = features.sort_values('cluster')
print(features_sorted)

# Separate first 8 points as Validation set (0-7)
X_train = features[["Area","Locality"]].values[:7]
Y_train = features[["Price"]].values[:7]
# Separate last 2 points as Validation set
X_test = features[["Area","Locality"]].values[7:]
Y_test = features[["Price"]].values[7:]