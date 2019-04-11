from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


class Cluster:
    # from https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    def hclust(self, data):
        plt.figure(figsize=(10, 10))
        plt.title("Custom Dendograms")
        dend = sch.dendrogram(sch.linkage(data, method='complete'))

        cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
        cluster.fit_predict(data)

        plt.figure(figsize=(10, 10))
        plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')

        plt.show()


