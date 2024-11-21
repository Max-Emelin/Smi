import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier

# Загрузка входных данных
input_file = 'Lab5\data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разделение входных данных на два класса на основании меток
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Визуализация входных данных
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
 edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
 edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

# Разбиение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.25, random_state=10)

# Классификатор на основе дерева решений
params = {'random_state': 1, 'max_depth': 20}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)


visualize_classifier(classifier, X_train, y_train)
visualize_classifier(classifier, X_test, y_test)
y_test_pred = classifier.predict(X_test)


# Оценка работы классификатора
class_names = ['Class-0', 'Class-1']
print("\ntest_size = ", 0.25, "random_state = ", 10, "params = ", params)
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")
plt.show()