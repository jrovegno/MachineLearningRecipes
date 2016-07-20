from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        return [self.closest(row) for row in X_test]
    
    def closest(self, row):
        dist = [euc(row, b) for b in self.X_train]
        best_index = dist.index(min(dist))
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.5)
print X_train.shape, X_test.shape

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
