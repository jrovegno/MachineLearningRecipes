import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print iris.feature_names
print iris.target_names
type(iris.target)
iris.target.shape
iris.data.shape
test_idx = [0, 50, 100]
# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
# print testing data
print 'Test Data: ' + str(test_data)
print 'Test Target: ' + str(test_target)
# 
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print 'Predicted: ' + str(clf.predict(test_data))
