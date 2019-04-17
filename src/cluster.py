from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn import cluster
import numpy as np
#from sklearn.naive_bayes import GaussianNB
#from skmultilearn.problem_transform import BinaryRelevance



class Cluster:
    # from https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    def hclust(self, data, title):
        data = np.array(data)
        plt.figure(figsize=(10, 10))
        plt.title(title)
        dend = sch.dendrogram(sch.linkage(data, method='complete'))

        new_cluster = cluster.AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        new_cluster.fit_predict(data)

        plt.figure(figsize=(10, 10))
        plt.scatter(data[:, 0], data[:, 1], c=new_cluster.labels_, cmap='rainbow')
        plt.title(title)
        # plt.xlabel('xx')
        # plt.ylabel('yy')
        plt.show()

    def k_means(self, data, title):
        data = np.array(data)

        kmeans = cluster.KMeans(n_clusters=3, n_init=1)
        kmeans.fit_predict(data)    # TODO  fit, fit_predict...?

        plt.figure(figsize=(10, 10))
        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.title(title)
        # plt.xlabel('xx')
        # plt.ylabel('yy')
        plt.show()

    # def multi_label_classify(self):