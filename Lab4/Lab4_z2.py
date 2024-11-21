import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Входной файл, содержащий данные
input_file = 'Lab4\data_multivar_regr.txt'

# Загрузка данных
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разбивка данных на обучающий и тестовый наборы
koef = 0.9
num_training = int(koef * len(X))
num_test = len(X) - num_training

# Тренировочные данные
X_train, y_train = X[:num_training], y[:num_training]
# Тестовые данные
X_test, y_test = X[num_training:], y[num_training:]

# Создание модели линейного регрессора
linear_regressor = linear_model.LinearRegression()
# Обучение модели с использованием обучающих наборов
linear_regressor.fit(X_train, y_train)

# Прогнозирование результата
y_test_pred = linear_regressor.predict(X_test)

# Измерение метрических характеристик
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Полиномиальная регрессия
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[25.2, 23.7, 17.6]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\ndatapoint:", datapoint)
print("training size:", koef*100,"%")
print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))
