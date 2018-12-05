import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


'''
Clustering class
'''
class Clusterer(object):
    def __init__(self):
        self.clustering_df = pickle.load(open("clustering_df.pickle", "rb"))
        self.data = pickle.load(open("data.pickle", "rb"))

    def run_clusterer(self):
        # (using 3 clusters because of the elbow method)
        num_clusters = 3

        # Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=1, init="k-means++")
        kmeans.fit(self.clustering_df)

        # Plotting the result (we plot Indnum against the Quality of Life Importance column)
        plt.scatter(self.clustering_df.iloc[:, 0], self.clustering_df.iloc[:, 28], c=kmeans.labels_, cmap="rainbow")
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 28], color="black")
        # uncomment to see clusters
        # plt.show()

        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = self.data
        cluster_map['cluster'] = kmeans.labels_

        print(cluster_map)
        print(kmeans.labels_)
