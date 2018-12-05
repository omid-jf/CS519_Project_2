import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


'''
Elbow method to find the number of clusters
'''
class Elbow(object):
    def __init__(self):
        self.clustering_df = pickle.load(open("clustering_df.pickle", "rb"))

    def run_elbow(self):
        sse_list = []

        for k in range(1, 15):
            km = KMeans(n_clusters=k, init="k-means++", max_iter=300, random_state=0)
            km.fit(self.clustering_df)
            sse_list.append(km.inertia_)

        plt.plot(range(1, 15), sse_list)
        plt.title("Finding the number of clusters (elbow method)")
        plt.ylabel("SSE")
        plt.xlabel("k")
        plt.show()
        plt.savefig('elbow.png')

