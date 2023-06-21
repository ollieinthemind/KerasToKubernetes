import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Read a csv file and show the records
features = pd.read_csv('../data/house.price.csv')
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

# Use Scikit-Learn's built-in function to fit Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
# print("Model weights are: ", model.coef_)
# print("Model intercept is: ", model.intercept_)

# Predict for one point from Test set
# print('Predicting for ', X_test[0])
# print('Expected value ', Y_test[0])
# print('Predicted value ', model.predict([[95,5]]))

# Separate first 8 points as Validation set (0-7)
X_train = features[["Area","Locality","Price"]].values[:8]
Y_train = features["Buy"].values[:8]
# Separate last 2 points as Validation set (0-7)
X_test = features[["Area","Locality","Price"]].values[8:]
Y_test = features["Buy"].values[8:]

model = LogisticRegression()
model.fit(X_train, Y_train)

#make a prediction on test data
Y_pred = model.predict(X_test)

#print expected results
print(Y_test)
#print the predictions
print(Y_pred)

# separate last 2 points as validation set (0-7)
X_test = features[["Area", "Locality", "Price",]].values[8:]
print(Y_test)