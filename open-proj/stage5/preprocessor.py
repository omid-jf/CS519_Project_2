import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle


'''
This class prepares the data by moving around values 
and combining the two sheets into one. The result is saved
in dataset.csv
It also aggregates the data of each individual in one row,
meaning that each row would contain the the consumptions 
patterns of the individual and save the result to clustering_df.pickle
'''
class Preprocessor(object):

    def run_preprocessor(self):
        num_activities = 27
        num_equipments = 11

        print("Preparing data")
        # Step 1: Load data
        df = pd.read_csv("raw_data.csv", header=0)
        cfdf = pd.read_csv("raw_cf.csv", header=0)

        # replace NaN with 0
        df = df.fillna(0)
        cfdf = cfdf.fillna(0)
        dataset_size = len(df.index)
        dataset_columns = len(df.columns)
        cf_dataset_size = len(cfdf.index)
        cf_dataset_columns = len(cfdf.columns)
        # num_activities = 27
        # num_equipments = 11

        df.insert(loc=dataset_columns, column="CarbonFootprint", value=0)

        # Step 2: Multiplying the equipment of each activity (zeros and ones) by the cells in the second sheet.
        equipments = []
        for i in range(num_equipments):
            equipments.append(cfdf.columns.values[i + 3])

        for row in range(dataset_size):
            carbons = []

            for equipment in equipments:
                if (row % num_activities) > 22:
                    df.loc[row, equipment] = float(cfdf.loc[row % num_activities, equipment])
                else:
                    df.loc[row, equipment] = int(df.loc[row, equipment]) * float(
                        cfdf.loc[row % num_activities, equipment])

                if df.loc[row, equipment] != 0:
                    carbons.append(df.loc[row, equipment])

            # Step 3: Computing the carbon footprint of each activity
            if (len(carbons) != 0) and (df.loc[row, "Consumption"] != 0):
                df.loc[row, "CarbonFootprint"] = sum(carbons) / len(carbons) * df.loc[row, "Consumption"]
            else:
                df.loc[row, "CarbonFootprint"] = 0

        df.to_csv("dataset.csv", sep="\t", header=True, index=False)

        print("Combining QolIs")
        # Step 4: We need to have a cell for each individual showing their total quality of life (average of the qualities of all activities).
        individual_qols = []
        for i in range(0, dataset_size - 1, num_activities):
            column_consumption = list(map(float, df.iloc[i:i + num_activities, 4]))
            column_qol = list(map(float, df.iloc[i:i + num_activities, 5]))
            # Multiply each QoLI with consumption
            mult = [x * y for x, y in zip(column_consumption, column_qol)]
            # Average all the above values for that user
            avg = sum(mult) / num_activities
            individual_qols.append(avg)
        # print(individusal_qols)

        df = df.fillna(0)
        cfdf = cfdf.fillna(0)
        dataset_size = len(df.index)
        dataset_columns = len(df.columns)
        cf_dataset_size = len(cfdf.index)
        cf_dataset_columns = len(cfdf.columns)
        print ("Running KMeans Clustering")
        # Step 5: Creating the clustering dataset (Each row containing the consumptions patterns of the individual)
        col_headers = ["Indnum"]
        col_headers.extend(list(df.loc[0:num_activities-1, "Activity"]))
        data = []
        row = 1
        for i in range(0, dataset_size-1, num_activities):
            column_consumption = list(map(float, df.loc[i:i + num_activities - 1, "Consumption"]))
            data.append([row] + column_consumption)
            row += 1

        clustering_df = pd.DataFrame(data=data, columns=col_headers)
        clustering_df.insert(loc=len(clustering_df.columns), column="QoLI", value=individual_qols)
        clustering_df.to_csv("clustering_ds.csv", sep="\t", header=True, index=False)

        pickle.dump(data, open("data.pickle", "wb"))
        pickle.dump(clustering_df, open("clustering_df.pickle", "wb"))
