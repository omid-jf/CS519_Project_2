from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle


class Regressor(object):
    def __init__(self):
        self.clustering_df = pickle.load(open("clustering_df.pickle", "rb"))
        # Predict any attribute (building a regression model for each attribute)
        self.regression_df = self.clustering_df.drop(self.clustering_df.columns[0], axis=1)  # Removing individual ID column
        self.alpha_params = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
        self.regressors = []  # a regressor for each attribute

    def run_regressor(self):
        for col in range(self.regression_df.shape[1]):  # Looping through columns
            print("\n\n*** Regression model for \"%s\" ***" % self.regression_df.columns[col])

            y = self.regression_df.iloc[:, col]  # Labels (target)
            X = self.regression_df.drop(self.regression_df.columns[col], axis=1)  # Features

            # Standardizing
            sc_X = StandardScaler()
            X_std = sc_X.fit_transform(X)

            sc_y = StandardScaler()
            y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

            # Splitting train and test data
            X_std_tr, X_std_ts, y_std_tr, y_std_ts = train_test_split(X_std, y_std, test_size=0.3, random_state=1)

            # Parameter tuning
            base_model = Ridge(random_state=1)  # Base model (Ridge regressor)
            grid = GridSearchCV(estimator=base_model, param_grid=dict(alpha=self.alpha_params), cv=10, iid=False)  # Grid search
            grid.fit(X_std_tr, y_std_tr)

            print("-- Best parameters values --")
            print(grid.best_params_)

            # Evaluation
            best_grid = grid.best_estimator_
            y_std_pred = best_grid.predict(X_std_ts)
            print("-- Mean squared error:\t%.3f%% --" % mean_squared_error(y_std_ts, y_std_pred))

            self.regressors.append(best_grid)

        print("\n\nTotal number of regressors: %d" % len(self.regressors))
