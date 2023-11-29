import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class Four_lab:

    def __init__(self):
        self.start_work()

    def start_work(self):
        df_training = pd.read_csv("winequality_white.csv", sep=";")
        X = df_training[
            ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]].values
        Y = df_training["quality"].values

        # X, Y = datasets.load_diabetes(return_X_y=True)
        X = Y[:, np.newaxis]

        x_train, x_test, y_train, y_test = self.custom_train_test_split(X, Y)
        self.scaling(x_train, y_train, x_test, y_test)

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

    def scaling(self, x_train, y_train, x_test, y_test):
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        self.perseptron(x_train_scaled, y_train, x_test_scaled, y_test)
        self.mp_classifier(x_train_scaled, y_train, x_test_scaled, y_test)

    def perseptron(self, x_train_scaled, y_train, x_test_scaled, y_test):
        perceptron = Perceptron(alpha=0.01, penalty="l1")
        perceptron.fit(x_train_scaled, y_train)
        y_pred_perceptron = perceptron.predict(x_test_scaled)
        accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
        print(f'Accuracy of Perceptron: {accuracy_perceptron}')

        train_sizes, train_scores, test_scores = learning_curve(perceptron, x_train_scaled, y_train)

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.plot(train_sizes, train_scores_mean, label='Training score')
        plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
        plt.title('Perceptron Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy Score')
        plt.legend(loc='best')
        plt.show()

    def mp_classifier(self, x_train_scaled, y_train, x_test_scaled, y_test):
        mlp_classifier = MLPClassifier(max_iter=100000, learning_rate_init=0.01, alpha=0.51, solver='adam')
        mlp_classifier.fit(x_train_scaled, y_train)
        y_pred_mlp = mlp_classifier.predict(x_test_scaled)
        accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
        print(f'Accuracy of MLPClassifier: {accuracy_mlp}')

        train_sizes, train_scores, test_scores = learning_curve(mlp_classifier, x_train_scaled, y_train)

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.plot(train_sizes, train_scores_mean, label='Training score')
        plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
        plt.title('MLPClassifier Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy Score')
        plt.legend(loc='best')
        plt.show()
