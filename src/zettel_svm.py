import zettel_preprocessor as process
import zettel_KE
from sklearn import svm
from sklearn.metrics import accuracy_score



# Build dictionary of words from email documents from training set.
# Consider the most common 3000 words.
# For each document in training set, create a frequency matrix for these words in dictionary and corresponding labels.
# [spam email file names start with prefix “spmsg”.


# TRAIN_DIR =
# TEST_DIR =

# features_matrix, labels = extract_features(TRAIN_DIR)
# test_feature_matrix, test_labels = extract_features(TEST_DIR)

#
# model = svm.SVC()
#
# model.fit(features_matrix, labels)  # 2-d features_matrix --> list(list(x,y))  # labels = target (keyword or not)  # TRAIN_DATA
#
# predicted_labels = model.predict(test_feature_matrix)  # 2-d test matrix  # TEST_DATA
#
# print(accuracy_score(test_lables, predicted_labels))

movies = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"
clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

import datetime
print(datetime.datetime.now())

z_process = process.ZettelPreProcessor()
zettels = z_process.get_zettels_from_clean_directory(movies)

ke = zettel_KE.KE(zettels)
suggested_keywords = ke.run(min_freq=1)

index = 0
for zettel in suggested_keywords:
    print("\nSuggested Keywords for Zettel " + str(index) + ": ")
    inner_i = 1
    for item in zettel:
        print(str(inner_i) + ": ")
        print(item)
        inner_i += 1
    index += 1

print("Done.")
print(datetime.datetime.now())


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
print(X)  # sepalLengthCM, sepalWidthCM
y = iris.target
print(y)  # labels -- setosa = 0, versicolor = 1, virginica = 2

# we create an instance of SVM and fit out data. We do not scale our data since we want to plot the support vectors
svc = svm.SVC(kernel='linear', C=1).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
print(np.c_[xx.ravel(), yy.ravel()])
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()