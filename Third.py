import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class Third:

    def __init__(self):
        self.start_work()

    def start_work(self):
        df_training = pd.read_csv("winequality-red.csv", sep=";")

        # X = df_training[
        #     ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
        #      "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]].values
        X = df_training["free sulfur dioxide"].values
        Y = df_training["quality"].values

        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        x_train, x_test, y_train, y_test = self.custom_train_test_split(X, Y)
        self.polynomial_feature(
            x_train.reshape((-1, 1)),
            y_train,
            x_test.reshape((-1, 1)),
            y_test
        )

    def learn_train_model(self, x_train, y_train, x_test, y_test):
        model = LinearRegression()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        R = model.score(y_test.reshape((-1, 1)), y_pred)

        print(f"Среднеквадратичная ошибка (MSE): {mse}")
        print(f"Кэф детерминизации: {R}")

        plt.scatter(x_train, y_train, color="black")
        plt.plot(x_test, y_pred, color="blue", linewidth=3)
        plt.show()

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

    def polynomial_feature(self, X_train, y_train, X_test, y_test):
        train_errors = []
        test_errors = []
        degrees = np.arange(1, 2, 5)

        for degree in degrees:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            pipeline = Pipeline(
                [
                    ("polynomial_features", poly),
                    ("linear_regression", model),
                ]
            )

            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly)

            train_error = mean_squared_error(y_train, y_train_pred)
            test_error = mean_squared_error(y_test, y_test_pred)

            train_errors.append(train_error)
            test_errors.append(test_error)

        plt.figure(figsize=(10, 6))
        plt.plot(degrees, train_errors, label='Train Error', marker='o')
        plt.plot(degrees, test_errors, label='Test Error', marker='o')
        plt.title('Зависимость ошибки от степени полинома')
        plt.xlabel('Степень полинома')
        plt.ylabel('Среднеквадратичная ошибка')
        plt.legend()
        plt.show()

    def regression_model(self, x_train, y_train, x_test, y_test):
        alphas = np.logspace(-6, 6, 13)

        train_scores = []
        test_scores = []

        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)

            train_scores.append(mse_train)
            test_scores.append(mse_test)

        plt.figure(figsize=(10, 6))
        plt.semilogx(alphas, train_scores, label='Train MSE')
        plt.semilogx(alphas, test_scores, label='Test MSE')
        plt.xlabel('alpha')
        plt.ylabel('Mean Squared Error')
        plt.title('Зависимость MSE от alpha')
        plt.legend()
        plt.show()
