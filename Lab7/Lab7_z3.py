import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Загрузка данных из входного файла
X = np.loadtxt('Lab7/data_quality.txt', delimiter=',')

# Включение входных данных в график
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Входные данные')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Определение диапазонов для 5 прогонов
ranges = [
    np.arange(2, 10),   
    np.arange(2, 11),   
    np.arange(2, 12), 
    np.arange(2, 15),  
    np.arange(2, 20),   
    np.arange(5, 15),  
    np.arange(6, 16),  
    np.arange(10, 20),  
    np.arange(12, 15),  
    np.arange(15, 25)
]

# Выполнение 5 прогонов с разными диапазонами
for idx, values in enumerate(ranges):
    print(f"\nПрогон {idx + 1}: Диапазон {values[0]} до {values[-1]+1}")
    
    # Инициализация переменных для каждого прогона
    scores = []
    
    # Итерирование по количествам кластеров
    for num_clusters in values:
        # Обучение модели кластеризации KMeans
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        kmeans.fit(X)
        
        # Получение силуэтной оценки
        score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))
        #print(f"\nКоличество кластеров = {num_clusters}")
        #print(f"Силуэтная оценка = {score}")
        
        scores.append(score)

    # Отображение силуэтных оценок на графике
    plt.figure()
    plt.bar(values, scores, width=0.7, color='black', align='center')
    plt.title(f'Силуэтная оценка для прогноза {idx + 1} (Диапазон {values[0]} до {values[-1]+1})')
    
    # Извлечение наилучшей оценки и оптимального количества кластеров
    num_clusters = np.argmax(scores) + values[0]
    print(f'\nОптимальное количество кластеров для прогноза {idx + 1} =', num_clusters)

# Отображение всех графиков
plt.show()
