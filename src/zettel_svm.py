import zettel_preprocessor as process
import zettel_KE as zke
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def get_lables(tokens, tags):
    labels = []
    all_tags = [tag for zettel in tags for tag in zettel]
    for zettel in tokens:
        for word in zettel:
            if word[0] in all_tags:
                labels.append(1)
            else:
                labels.append(0)
    return labels

def extract_features(directory):
    z_process = process.ZettelPreProcessor()
    zettels = z_process.get_zettels_from_clean_directory(directory)
    ke = zke.KE(zettels)
    ke.run()
    feature_1 = ke.tf_idf_scores
    feature_2 = ke.all_scores_dict
    feature_matrix = []
    for zettel in ke.lemma_tokens:
        for word in zettel:
            feature_matrix.append((feature_1[word[0]], feature_2[word[0]]))
    return feature_matrix, get_lables(ke.lemma_tokens, ke.tags), ke.lemma_tokens

def get_keywords(predicted, tokens):
    w_index = 0
    all_predicted_tokens = []
    for zettel in tokens:
        z_predicted = []
        for word in zettel:
            if predicted[w_index] == 1:
                z_predicted.append(word[0])
        all_predicted_tokens.append(z_predicted)
    return all_predicted_tokens


TRAIN_DIR = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"
TEST_DIR = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

import datetime
print(datetime.datetime.now())

# 2-d features_matrix --> list(list(x,y))  # labels = target (keyword or not)
features_matrix, labels, tokens = extract_features(TRAIN_DIR)
test_features_matrix, test_labels, test_tokens = extract_features(TEST_DIR)

model = svm.SVC(kernel='linear', C=1).fit(features_matrix, labels)

predicted_labels = model.predict(test_features_matrix)
for item in predicted_labels:
    print(item)

print(accuracy_score(test_labels, predicted_labels))
print(get_keywords(predicted_labels, test_tokens))

# https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
x_ = []
y_ = []
for item in features_matrix:
    x_.append(item[0])
    y_.append(item[1])

x_.sort()
x_min = float(x_[0])
x_.sort(reverse=True)
x_max = float(x_[0] + 1)

y_.sort()
y_min = float(y_[0])
y_.sort(reverse=True)
y_max = float(y_[0] + .3)

# create a mesh to plot in
h = (x_max / x_min)/1000
xx, yy = np.meshgrid(np.arange(0.0, x_max, h),
                     np.arange(0.0, y_max, h))

plt.subplot(1, 1, 1)
Z = []
Z_index = 0
for list in xx:
    new_list = []
    for item in list:
        new_list.append(predicted_labels[Z_index])
        Z_index += 1
    Z.append(new_list)

plt.contourf(xx, yy, Z, alpha=0.8)

plt.scatter(x_, y_, c=labels)
plt.xlabel('TF * IDF')
plt.ylabel('Total Score')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

print("Done.")
print(datetime.datetime.now())
