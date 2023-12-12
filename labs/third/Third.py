import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class Third:

    def __init__(self):
        self.start_work()

    def start_work(self):
        df_training = pd.read_csv("data/winequality_white.csv", sep=";")

        X = df_training.iloc[:, :-2]
        Y = df_training.iloc[:, -1]

        x_train, x_test, y_train, y_test = self.custom_train_test_split(X.values, Y.values)
        self.learn_train_model(
            x_train,
            y_train,
            x_test,
            y_test
        )

        self.polynomial_feature(
            x_train,
            y_train,
            x_test,
            y_test
        )

        self.regression_model(
            x_train,
            y_train,
            x_test,
            y_test
        )

    def learn_train_model(self, x_train, y_train, x_test, y_test):
        model = LinearRegression()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        R = r2_score(y_test, y_pred)

        print(f"Среднеквадратичная ошибка (MSE): {mse}")
        print(f"Кэф детерминизации: {R}")

    def custom_train_test_split(self, x: Series, y: Series, test_size=0.2, random_state=None, shuffle: bool = False):
        if random_state is not None:
            np.random.seed(random_state)

        total_samples = len(x)
        test_samples = int(total_samples * test_size)
        indices = np.arange(total_samples)

        if shuffle:
            np.random.shuffle(indices)

        test_indices = indices[:test_samples]
        train_indices = indices[test_samples:]

        X_train = x[train_indices]
        X_test = x[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    def polynomial_feature(self, x_train, y_train, x_test, y_test):
        degrees = range(1, 3)
        train_errors = []
        test_errors = []
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x_train)
            model = LinearRegression()
            model.fit(x_poly, y_train)

            y_predict_train = model.predict(poly.transform(x_train))
            y_predict_test = model.predict(poly.transform(x_test))

            mse_train = r2_score(y_train, y_predict_train)
            mse_test = r2_score(y_test, y_predict_test)

            train_errors.append(mse_train)
            test_errors.append(mse_test)

        plt.plot(degrees, train_errors, label='Train')
        plt.plot(degrees, test_errors, label='Test')
        plt.xlabel('Polymomial function')
        plt.ylabel('R2')
        plt.legend()
        plt.show()

    def regression_model(self, x_train, y_train, x_test, y_test):
        alphas = np.logspace(-6, 6, 13)
        test_errors = []
        train_errors = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(x_train, y_train)

            y_predict_train = ridge.predict(x_train)
            y_predict_test = ridge.predict(x_test)

            train_errors.append(r2_score(y_train, y_predict_train))
            test_errors.append(r2_score(y_test, y_predict_test))

        plt.plot(alphas, train_errors, label='Train')
        plt.plot(alphas, test_errors, label='Test')
        plt.xlabel('Ridge')
        plt.ylabel('R2')
        plt.legend()
        plt.show()
