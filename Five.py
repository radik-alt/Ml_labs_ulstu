import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class Five:

    def __init__(self):
        self.start_work()

    def start_work(self):
        df_training = pd.read_excel("data/oline_retail.xlsx")
        df_except_last = df_training.iloc[:, :-1]
        self.scalling(df_except_last)

    def scalling(self, x):
        print(x)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        self.classification(x_scaled)

    def classification(self, x_scaled):
        algorithms = [
            ('KMeans', KMeans(n_clusters=3)),
            ('Agglomerative', AgglomerativeClustering(n_clusters=3)),
            ('DBSCAN', DBSCAN(eps=0.5, min_samples=5))
        ]

        for name, algorithm in algorithms:
            labels = algorithm.fit_predict(x_scaled)

            silhouette_avg = silhouette_score(x_scaled, labels)

            plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=labels, cmap='viridis')
            plt.title(f'{name} - Silhouette Score: {silhouette_avg:.2f}')
            plt.show()

            print(f'{name} - Silhouette Score: {silhouette_avg:.2f}')
