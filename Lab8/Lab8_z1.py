import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

# Загрузка набора данных iris
iris = datasets.load_iris()

# Разбиение данных на обучающий и тестовый наборы (в пропорции 80/20)
indices = StratifiedKFold(n_splits=5)

# Используем первый набор
train_index, test_index = next(iter(indices.split(iris.data, iris.target)))

# Извлечение обучающих данных и меток
X_train = iris.data[train_index]
y_train = iris.target[train_index]

# Извлечение тестовых данных и меток
X_test = iris.data[test_index]
y_test = iris.target[test_index]

# Извлечение количества классов
num_classes = len(np.unique(y_train))

# Создание GММ
classifier = GaussianMixture(n_components=num_classes, covariance_type='full', init_params='kmeans', max_iter=20)

# Инициализация средних GММ
classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
    for i in range(num_classes)])

# Обучение GММ-классификатора
classifier.fit(X_train)

# Вычерчивание границ
plt.figure()
colors = 'bgr'
for i, color in enumerate(colors):
    # Извлечение собственных значений и собственных векторов
    eigenvalues, eigenvectors = np.linalg.eigh(classifier.covariances_[i][:2, :2])
    # Нормализация первого собственного вектора
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
    # Извлечение угла наклона
    angle = np.arctan2(norm_vec[1], norm_vec[0])
    angle = 180 * angle / np.pi
    # Масштабный множитель для увеличения эллипсов (выбрано произвольное значение, которое нас удовлетворяет)
    scaling_factor = 8
    eigenvalues *= scaling_factor
    # Вычерчивание эллипсов
    ellipse = patches.Ellipse(classifier.means_[i, :2], eigenvalues[0], eigenvalues[1], 180 + angle, color=color)
    axis_handle = plt.subplot(1, 1, 1)
    ellipse.set_clip_box(axis_handle.bbox)
    ellipse.set_alpha(0.6)
    axis_handle.add_artist(ellipse)

# Откладывание входных и тестовых данных на графике
colors = 'bgr'
for i, color in enumerate(colors):
    cur_data = iris.data[iris.target == i]
    plt.scatter(cur_data[:,0], cur_data[:,1], marker='o',
    facecolors='none', edgecolors='black', s=40,
    label=iris.target_names[i])
    test_data = X_test[y_test == i]
    plt.scatter(test_data[:,0], test_data[:,1], marker='s',
    facecolors='black', edgecolors='black', s=40,
    label=iris.target_names[i])

# Вычисление прогнозных результатов для обучающих и тестовых данных
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('Правильность (accuracy) для обучающих данных =', accuracy_training)

y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print('Правильность (accuracy) для тестовых данных =', accuracy_testing)
plt.title('GMM-классификатор')
plt.xticks(())
plt.yticks(())
plt.show()




