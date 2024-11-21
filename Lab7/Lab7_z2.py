import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Загрузка данных из входного файла
X = np.loadtxt('Lab7\data_clustering.txt', delimiter=',')

quantile_ = 1
# Оценка ширины окна для Х
bandwidth_X = estimate_bandwidth(X, quantile=quantile_, n_samples=len(X))

# Кластеризация данных методом сдвига среднего
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Извлечение центров кластеров
cluster_centers = meanshift_model.cluster_centers_
print("\n quantile = ", quantile_)
print('Центры кластеров:\n', cluster_centers)

# Оценка количества кластеров
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nКоличество кластеров во входных данных =", num_clusters)

# Отображение на графике точек и центров кластеров
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # Отображение на графике точек, принадлежащих текущему кластеру
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')

    # Отображение на графике центра кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
    markerfacecolor='black', markeredgecolor='black',
    markersize=15)

plt.title('Кластеры')
plt.show()

