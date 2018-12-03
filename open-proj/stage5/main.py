import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# # Step 1: Load data
# df = pd.read_csv("raw_data.csv", header=0)
# cfdf = pd.read_csv("raw_cf.csv", header=0)
#
# # replace NaN with 0
# df = df.fillna(0)
# cfdf = cfdf.fillna(0)
#
# dataset_size = len(df.index)
# dataset_columns = len(df.columns)
# cf_dataset_size = len(cfdf.index)
# cf_dataset_columns = len(cfdf.columns)
# num_activities = 27
# num_equipments = 11
#
# df.insert(loc=dataset_columns, column="CarbonFootprint", value=0)
#
# # Step 2: Multiplying the equipment of each activity (zeros and ones) by the cells in the second sheet.
# equipments = []
# for i in range(num_equipments):
#     equipments.append(cfdf.columns.values[i+3])
#
# for row in range(dataset_size):
#     carbons = []
#
#     for equipment in equipments:
#         if (row % num_activities) > 22:
#             df.loc[row, equipment] = float(cfdf.loc[row % num_activities, equipment])
#         else:
#             df.loc[row, equipment] = int(df.loc[row, equipment]) * float(cfdf.loc[row % num_activities, equipment])
#
#         if df.loc[row, equipment] != 0:
#             carbons.append(df.loc[row, equipment])
#
#     # Step 3: Computing the carbon footprint of each activity
#     if (len(carbons) != 0) and (df.loc[row, "Consumption"] != 0):
#         df.loc[row, "CarbonFootprint"] = sum(carbons) / len(carbons) * df.loc[row, "Consumption"]
#     else:
#         df.loc[row, "CarbonFootprint"] = 0
#
# df.to_csv("dataset.csv", sep="\t", header=True, index=False)
#
# # Step 4: We need to have a cell for each individual showing their total quality of life (average of the qualities of all activities).
# individual_qols = []
# for i in range(0, dataset_size-1, num_activities):
#     column_consumption = list(map(float, df.iloc[i:i+num_activities, 4]))
#     column_qol = list(map(float, df.iloc[i:i+num_activities, 5]))
#     # Multiply each QoLI with consumption
#     mult = [x * y for x, y in zip(column_consumption, column_qol)]
#     # Average all the above values for that user
#     avg = sum(mult)/num_activities
#     individual_qols.append(avg)
#
# print(individual_qols)
#
# # Step 5: Creating the clustering dataset (Each row containing the consumptions patterns of the individual)
# col_headers = ["Indnum"]
# col_headers.extend(list(df.loc[0:num_activities-1, "Activity"]))
# data = []
# row = 1
# for i in range(0, dataset_size-1, num_activities):
#     column_consumption = list(map(float, df.loc[i:i + num_activities - 1, "Consumption"]))
#     data.append([row] + column_consumption)
#     row += 1
#
# clustering_df = pd.DataFrame(data=data, columns=col_headers)
# clustering_df.insert(loc=len(clustering_df.columns), column="QoLI", value=individual_qols)
#
# clustering_df.to_csv("clustering_ds.csv", sep="\t", header=True, index=False)
#
# # Step 6: Clustering (using 10 clusters for now)
# kmeans = KMeans(n_clusters=10, random_state=1, init="k-means++")
# kmeans.fit(clustering_df)
#
# # Plotting the result (we plot Indnum against the Quality of Life Importance column)
# plt.scatter(clustering_df.iloc[:, 0], clustering_df.iloc[:, 28], c=kmeans.labels_, cmap="rainbow")
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 28], color="black")
# plt.show()

#######################
# TEMP
#######################
import pickle
# pickle.dump(clustering_df, open("clustering_df.pickle", "wb"))
clustering_df = pickle.load(open("clustering_df.pickle", "rb"))
#######################


# Predict any attribute (building a regression model for each attribute)
regression_df = clustering_df.drop(clustering_df.columns[0], axis=1)  # Removing individual ID column
alpha_params = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
regressors = []  # a regressor for each attribute

for col in range(regression_df.shape[1]):  # Looping through columns
    print("\n\n*** Regression model for \"%s\" ***" % regression_df.columns[col])

    y = regression_df.iloc[:, col]  # Labels (target)
    X = regression_df.drop(regression_df.columns[col], axis=1)  # Features

    # Standardizing
    sc_X = StandardScaler()
    X_std = sc_X.fit_transform(X)

    sc_y = StandardScaler()
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

    # Splitting train and test data
    X_std_tr, X_std_ts, y_std_tr, y_std_ts = train_test_split(X_std, y_std, test_size=0.3, random_state=1)

    # Parameter tuning
    base_model = Ridge(random_state=1)  # Base model (Ridge regressor)
    grid = GridSearchCV(estimator=base_model, param_grid=dict(alpha=alpha_params), cv=10, iid=False)  # Grid search
    grid.fit(X_std_tr, y_std_tr)

    print("-- Best parameters values --")
    print(grid.best_params_)

    # Evaluation
    best_grid = grid.best_estimator_
    y_std_pred = best_grid.predict(X_std_ts)
    print("-- Mean squared error:\t%.3f%% --" % mean_squared_error(y_std_ts, y_std_pred))

    regressors.append(best_grid)

print("\n\nTotal number of regressors: %d" % len(regressors))
