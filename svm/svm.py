#training samples
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)

#Predicting new values
clf.predict([[2., 2.]])

#Support vectors
clf.support_vectors_ #get support vectors
clf.support_ #get indices of support vectors
clf.n_support_ #get number of support vectors for each class.

##Multiclass classification
#one-against-one approach
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC()
clf.fit(X, Y)
dec = clf.decision_function([[1]])
dec.shape[1]

#one-vs-the-rest
lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
dec = lin_clf.decision_function([[1]])
dec.shape[1]

