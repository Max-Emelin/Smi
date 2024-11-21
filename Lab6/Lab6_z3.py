import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# Загрузка входных данных
input_file = 'Lab6\Adata_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разбиение данных на три класса на основании меток
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Разбиение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Определение сетки значений параметров
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]}, {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]} ]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n##### Поиск оптимальных значений параметров для", metric)
    print("\nВ таблице ниже в каждом столбце первым указано значение")
    print("n_estimators, вторым - max_depth, третьим - mean_test_score.")
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0),parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

print("\nGrid scores (оценка) для значений параметров:")
par1 = classifier.cv_results_['param_n_estimators']
par2 = classifier.cv_results_['param_max_depth']
par3 = classifier.cv_results_['mean_test_score']
par = [par1, par2, par3]
for par in par:
    print (par)

print("\nЛучшие значения параметров:", classifier.best_params_)
y_pred = classifier.predict(X_test)
print("\nОтчет о качестве классификатора:\n")
print(classification_report(y_test, y_pred))