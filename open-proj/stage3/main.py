import sys
import pandas as pd
import numpy as np
import math

# Step 1: Load data
df = pd.read_csv("data.csv", header=None)
cfdf = pd.read_csv("cf.csv", header=None)

# replace NaN with 0
df = df.fillna(0)
cfdf = cfdf.fillna(0)

dataset_size = len(df.index)
dataset_columns = len(df.columns)
cf_dataset_size = len(cfdf.index)
cf_dataset_columns = len(cfdf.columns)
num_activities = 27

x1 = df.iloc[1:dataset_size, 0:dataset_columns].values
x2 = cfdf.iloc[0:cf_dataset_size, 0:cf_dataset_columns].values

# Step 2- Multiplying the equipment of each activity (zeros and ones) by the cells in the second sheet. 
for i in range (0, dataset_size-2):
	for j in range (6, dataset_columns):
		x1[i+1][j] = int(x1[i+1][j]) * float(x2[(i % num_activities)+1][j - 3])

# Step 3- We need to have a cell for each individual showing their total quality of life (average of the qualities of all activities).
individual_qols = [] 
for i in range(0, dataset_size-1, num_activities):
	column_consumption = list(map(float, x1[i:i+num_activities, 4]))
	column_qol = list(map(float, x1[i:i+num_activities, 5]))
	# Multiply each QoLI with consumption
	mult = [x * y for x, y in zip(column_consumption, column_qol)]
	# Average all the above values for that user
	avg = sum(mult)/num_activities
	individual_qols.append(avg)

print (individual_qols)