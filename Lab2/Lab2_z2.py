import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utilities import visualize_classifier

def test() :
    # Определение образца входных данных
    X = np.array([
        [3.1, 7.2], [4, 6.7], [2.9, 8], 
        [5.1, 4.5], [6, 5], [5.6, 5], 
        [3.3, 0.4], [3.9, 0.9], [2.8, 1],
        [0.5, 3.4], [1, 4], [0.6, 4.9]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    # Создание логистического классификатора
    #classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # Тренировка классификатора
    classifier.fit(X, y)

    # Визуализация работы классификатора
    visualize_classifier(classifier, X, y)
def first() :
    # Новый набор данных с 2 признаками и 4 классами, которые легче разделить
    X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], 
                [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], 
                [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])  # 2 признака
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # 4 класса

    # Преобразуем данные, чтобы классы были более разделимыми
    X[:6, 0] += 1  # Смещение первого признака для классов 0 и 1
    X[6:, 0] -= 1  # Смещение первого признака для классов 2 и 3
    
    # Создание логистического классификатора
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # Тренировка классификатора
    classifier.fit(X, y)

    # Визуализация работы классификатора
    visualize_classifier(classifier, X, y)
def second() :
    # Новый набор данных с 3 признаками и 2 классами
    X = np.array([
    [5.5, 3.1], [2.3, 4.7], [8.9, 1.5], [6.3, 7.8], [2.8, 5.6], 
    [3.5, 9.2], [1.2, 3.7], [4.6, 2.4], [7.2, 0.9], [9.1, 6.5], 
    [4.4, 7.3], [5.9, 2.1]
    ])

    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])
    # Создание логистического классификатора
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # Тренировка классификатора
    classifier.fit(X, y)

    # Визуализация работы классификатора
    visualize_classifier(classifier, X, y)
def third() :
    # Новый набор данных с 2 признаками и 4 классами, которые легче разделить
    X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], 
                [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], 
                [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])  # 2 признака
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # 4 класса

    # Преобразуем данные, чтобы классы были более разделимыми
    X[:6, 0] += 1  # Смещение первого признака для классов 0 и 1
    X[6:, 0] -= 1  # Смещение первого признака для классов 2 и 3
    
    # Создание логистического классификатора
    classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

    # Тренировка классификатора
    classifier.fit(X, y)

    # Визуализация работы классификатора
    visualize_classifier(classifier, X, y)

#test()
#first()
#second()
third()

