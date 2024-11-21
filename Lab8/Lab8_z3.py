import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Загрузка данных из входного файла
input_file = 'Lab8\sales.csv' 
file_reader = csv.reader(open(input_file, 'r'), delimiter=',')
X = []
names = []  # Для хранения названий продуктов
l = 1  # Индекс для пропуска первой строки с заголовками

# Чтение данных
for count, row in enumerate(file_reader):
    if count == 0:  # Пропускаем заголовок
        names = row[1:]  # Предположим, что первый столбец это название продуктов
        continue
    X.append([float(x) for x in row[1:]])  # Считывание данных, начиная с 2-го столбца

# Преобразование данных в массив numpy
X = np.array(X)

# Списки для сохранения результатов
results = []

# Проведение 10 прогонов с различными параметрами
quantile_values = np.linspace(0.1, 0.9, 10)  # Разные значения quantile
bandwidth_values = [1, 2, 5]  # Разные значения bandwidth

for q in quantile_values:
    for bw in bandwidth_values:
        print(f"Running experiment with quantile={q}, bandwidth={bw}")
        
        # Оценка ширины окна входных данных
        bandwidth = estimate_bandwidth(X, quantile=q, n_samples=len(X))

        # Вычисление кластеризации методом сдвига среднего
        meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift_model.fit(X)

        # Извлечение информации о кластерах
        labels = meanshift_model.labels_
        cluster_centers = meanshift_model.cluster_centers_
        num_clusters = len(np.unique(labels))

        # Сохранение результатов
        results.append((q, bw, num_clusters, cluster_centers))

        print(f"Number of clusters = {num_clusters}")
        print(f"Centers of clusters:")
        for cluster_center in cluster_centers:
            print('\t'.join([str(round(x, 2)) for x in cluster_center]))  # Печать центров кластеров

# Отображение графиков
for i, (q, bw, num_clusters, cluster_centers) in enumerate(results):
    print(f"\nPlotting result for quantile={q}, bandwidth={bw}")
    
    # Извлечение двух признаков в целях визуализации (например, Price, Quantity)
    cluster_centers_2d = cluster_centers[:, 1:3]
    
    plt.figure(i)
    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], s=120, edgecolors='black', facecolors='none')
    offset = 0.25
    plt.xlim(cluster_centers_2d[:, 0].min() - offset * cluster_centers_2d[:, 0].ptp(),
             cluster_centers_2d[:, 0].max() + offset * cluster_centers_2d[:, 0].ptp())
    plt.ylim(cluster_centers_2d[:, 1].min() - offset * cluster_centers_2d[:, 1].ptp(),
             cluster_centers_2d[:, 1].max() + offset * cluster_centers_2d[:, 1].ptp())
    plt.title(f'Centers of 2D Clusters (quantile={q}, bandwidth={bw})')
    plt.show()
