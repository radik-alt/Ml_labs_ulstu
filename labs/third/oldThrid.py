import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class Third:

    def __init__(self):
        self.start_work()

    def true_fun(self, X):
        return np.cos(1.5 * np.pi * X)

    def start_work(self):
        df_training = pd.read_csv("../../data/winequality_white.csv", sep=";")
        X = df_training[
            ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]].values
        Y = df_training["quality"].values

        # X, Y = datasets.load_diabetes(return_X_y=True)
        X = Y[:, np.newaxis]

        x_train, x_test, y_train, y_test = self.custom_train_test_split(X, Y)
        self.regression_model(
            x_train,
            y_train,
            x_test,
            y_test
        )

    def learn_train_model(self, x_train, y_train, x_test, y_test):
        model = LinearRegression()

        print(x_test.shape)
        print(y_test.shape)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        R = r2_score(y_test, y_pred)

        plt.figure(figsize=(8, 10))
        plt.scatter(x_test, y_test, color="blue", label="Test Data")
        plt.plot(x_test, y_pred, color="red", label="Predict Data")
        plt.title("Line Regression")
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.show()

        print("Coefficients: \n", model.coef_)
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
        degrees = [5, 7]

        plt.figure(figsize=(14, 5))
        for i in range(len(degrees)):
            ax = plt.subplot(1, len(degrees), i + 1)
            plt.setp(ax, xticks=(), yticks=())

            print(x_train.shape)
            print(y_train.shape)

            polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline(
                [
                    ("polynomial_features", polynomial_features),
                    ("linear_regression", linear_regression),
                ]
            )

            pipeline.fit(x_train, y_train)

            scores = cross_val_score(
                pipeline, x_train, y_train, scoring="neg_mean_squared_error", cv=10
            )

            y_predict = pipeline.predict(x_test)

            plt.plot(x_test, y_predict, label="Model")
            plt.scatter(x_train, y_train, edgecolor="r", s=20, label="Samples")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(loc="best")
            plt.title(
                "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
                    degrees[i], -scores.mean(), scores.std()
                )
            )
        plt.show()

    def regression_model(self, x_train, y_train, x_test, y_test):
        alphas = np.logspace(-6, 6, 13)

        train_scores = []
        test_scores = []

        for alpha in alphas:
            print(alpha)
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
