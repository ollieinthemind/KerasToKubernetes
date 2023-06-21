import time
import pydot
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier



# Read a csv file and show the records
features = pd.read_csv('../data/winequality-red.csv', sep=';')
features.describe()

#separate the Xs and Ys
#

X = features # all features
X = X.drop(['quality'],axis=1) #remove the quality which is a Y
Y = features[['quality']]
print("X features (Inputs) : ", X.columns)
print("Y features (Outputs) : ", Y.columns)

from sklearn.model_selection import train_test_split
#split the data into training and test datasets -> 80-20 split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print("Training features: X", X_train.shape, " Y", Y_train.shape)
print("Test features: X", X_test.shape, " Y", Y_test.shape)



#
t_start = time.time()
#




#
# model = LogisticRegression()
# model.fit(X_train,Y_train.values.ravel())
# Y_pred = model.predict(X_train)
# print("Precision for LogisticRegression on Training data: ", precision_score(Y_train, Y_pred, average='micro'))
# #make prediction on testing data and get precision
# Y_pred = model.predict(X_test)
# print("Precision for LogisticRegression on Testing data: ", precision_score(Y_test,Y_pred, average='micro'))
#

# #train the KNN model
# model = KNeighborsClassifier(n_neighbors=20)
# model.fit(X_train.values, Y_train)
# #predict for X_test
# Y_pred = model.predict(X_test)
# #compare with Y-Test
#
# print("Precision for KNN: ", precision_score(Y_test, Y_pred, average='micro'))


model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
# #predict for the test
Y_pred = model.predict(X_train)
print("Precision for Decision Tree on Training data: ", precision_score(Y_train, Y_pred, average='micro'))
Y_pred = model.predict(X_test)
print("Precision for Decision Tree on Testing data: ", precision_score(Y_test, Y_pred, average='micro'))

# #find the precision of how good it is
# print("Precision for Decision Tree: ", precision_score(Y_test, Y_pred, average='micro'))
# #export as dot file
# export_graphviz(model,
#                 out_file='tree.dot',
#                 feature_names=X_train.columns,
#                 class_names=str(range(6)),
#                 rounded=True, proportion = False,
#                 precision = 1, filled = True)

# #build the model with 100 random trees
# model = RandomForestClassifier(n_estimators=100)
# #ffit your training data
# model.fit(X_train, Y_train.values.ravel())
# #make prediction for testing data
# Y_pred = model.predict(X_test)
# #show the precision value
# print("Precision for Random Forest: ", precision_score(Y_test, Y_pred, average='micro'))
#

t_end = time.time()

t_tot = (t_end-t_start)
print("Time to run operation: ", t_tot)

