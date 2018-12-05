import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
from preprocessor import Preprocessor
from elbow import Elbow
from clusterer import Clusterer
from regressor import Regressor


'''
Find the household with the higher qoli from a singe group
'''
def find_happiest(cluster):
    max = 0
    max_index = -1
    for i in range (len(cluster)):
        if cluster[-1] > max:
            max = cluster[i]
            max_index = i
    return max, max_index


'''
Find the household with the lowest carbon emissions from a singe group
'''
def find_greenest(cluster):
    min = 100000000
    min_index = -1
    for i in range (len(cluster)):
        if sum(cluster[1:len(cluster)-1]) < min:
            min = cluster[i]
            min_index = i
    return min, min_index


# Preprocessing
preprocessor = Preprocessor()
preprocessor.run_preprocessor()

# Elbow method
elbow = Elbow()
elbow.run_elbow()

# Clustering
clusterer = Clusterer()
clusterer.run_clusterer()

# Regression
regressor = Regressor()
regressor.run_regressor()
