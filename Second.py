import re
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


class Second:

    def __int__(self):
        print()

    def save_scv_file(self):
        x1_values = np.linspace(0, 10, 400)
        x2_values = np.linspace(0, 10, 400)
        y_values = np.cos(x1_values + x2_values)

        df = pd.DataFrame({'x1': x1_values, 'x2': x2_values, 'y': y_values})
        df.to_csv('my_data.csv', index=False)

    def second_lab(self):
        df = pd.read_csv('my_data.csv')

        mean_values = df.mean()
        min_values = df.min()
        max_values = df.max()

        for column in df.columns:
            print(f"Столбец '{column}':")
            print(f"Среднее значение: {mean_values[column]}")
            print(f"Минимальное значение: {min_values[column]}")
            print(f"Максимальное значение: {max_values[column]}")
            print("\n")

        mean_x1 = df['x1'].mean()
        mean_x2 = df['x2'].mean()

        filtered_df = df[(df['x1'] < mean_x1) | (df['x2'] < mean_x2)]
        filtered_df.to_csv('filtered_data.csv', index=False)

        constant = 5
        df['y1'] = df['y'] * (df['x2'] - constant)
        df['y2'] = df['y'] * (df['x1'] - constant)

        plt.figure(figsize=(10, 5))
        plt.scatter(df['x1'], df['y1'], label='y(x1)(x2 - константа)', color='b', marker='o')
        plt.xlabel('x1')
        plt.ylabel('y(x1)(x2 - константа)')
        plt.title('График y(x1)(x2 - константа)')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(df['x2'], df['y2'])

        plt.figure(figsize=(10, 5))
        plt.scatter(df['x2'], df['y2'], label='y(x2)(x1 - константа)', color='r', marker='o')
        plt.xlabel('x2')
        plt.ylabel('y(x2)(x1 - константа)')
        plt.title('График y(x2)(x1 - константа)')
        plt.legend()
        plt.grid(True)
        plt.show()

        x1 = df['x1']
        x2 = df['x2']
        X1, X2 = np.meshgrid(x1, x2)
        Y = np.cos(X1 + X2)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X1, X2, Y, cmap='viridis')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y(x1, x2)')
        ax.set_title('3D График функции y(x1, x2) для каждого x1')
        plt.show()
