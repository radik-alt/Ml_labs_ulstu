import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class Five:

    def __init__(self):
        self.start_work()

    def remove_outliers_iqr(self, df, features):
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)]

    def start_work(self):
        df = pd.read_excel("data/oline_retail.xlsx")
        df = df.head(10000)
        df_no_outliers = self.remove_outliers_iqr(df, ["Quantity", "UnitPrice"])
        print("Исходный DataFrame:")
        print(df.describe())
        print("\nDataFrame без выбросов:")
        print(df_no_outliers.describe())
        feature = df_no_outliers[["Quantity", "UnitPrice"]]

        self.scalling(feature.values)

    def scalling(self, x):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        self.classification(x_scaled)

    def classification(self, x_scaled):
        print(x_scaled.shape)

        algorithms = [
            ('KMeans', KMeans(n_clusters=3)),
            ('Agglomerative', AgglomerativeClustering(n_clusters=3)),
            ('DBSCAN', DBSCAN(eps=0.5, min_samples=5))
        ]

        for name, algorithm in algorithms:
            labels = algorithm.fit_predict(x_scaled)
            print(f"labels: {labels}")
            silhouette_avg = silhouette_score(x_scaled, labels)
            plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=labels, cmap='viridis')
            plt.title(f'{name} - Silhouette Score: {silhouette_avg:.2f}')
            plt.show()

            print(f'{name} - Silhouette Score: {silhouette_avg:.2f}')
