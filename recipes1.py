import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
features_dict = {1:'smooth', 0:'bumpy'}
labels_dict = {1:'orange', 0:'apple'}

features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print clf.predict([[150, 0]])
