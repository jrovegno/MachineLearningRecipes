from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.5)
print 'Training data dim: ' + str(X_train.shape)
print 'Testing data dim: ' + str(X_test.shape)

from sklearn.tree import DecisionTreeClassifier
my_classifier = DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print 'DecisionTreeClassifier Accuracy: ' + str(accuracy_score(y_test, predictions))

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)
print 'KNeighborsClassifier Accuracy: ' + str(accuracy_score(y_test, predictions))
