import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Загрузка данных с ценами на недвижимость
housing_data = datasets.load_boston()

# Перемешивание данных
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

testSize = 0.1
randomState = 2
maxDepth = 7



# Разбиение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

# Модель на основе регрессора AdaBoost
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=maxDepth), n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Вычисление показателей эффективности регрессора AdaBoost
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred )
print("\nРЕГРЕССОР ADABOOST")
print("test_size = ",testSize," random_state = ", randomState, " max_depth = ",maxDepth)
print("Среднеквадратическая ошибка =", round(mse, 2))
print("Объяснённая дисперсия =", round(evs, 2))

# Извлечение важности признаков
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Нормализация значений важности признаков
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортировка и перестановка значений
index_sorted = np.flipud(np.argsort(feature_importances))

# Расстановка меток вдоль оси Х
pos = np.arange(index_sorted.shape[0]) + 0.5

# Построение столбчатой диаграммы
plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Относительная важность')
plt.title('Относительная важность, определённая посредством регрессора AdaBoost')
plt.show()
