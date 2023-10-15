import numpy as np
from sklearn.linear_model import LinearRegression


class MyWork:
    def __init__(self):
        self.start()

    def start(self):
        A = np.array([[2, 1, 5, 4], [1, 2, 3]])
        B = np.array([[3, 4, 7, 8], [1, 2, 3]])  # dot product output = np.dot(A, B) print(output)

        print(np.dot(A, B))
