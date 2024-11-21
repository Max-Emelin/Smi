import numpy as np
from sklearn import preprocessing

def test() :
    # Предоставление меток входных данных
    input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

    # Создание кодировщика и установление соответствия между метками и числами
    encoder = preprocessing.LabelEncoder()
    encoder.fit(input_labels)

    # Вывод отображения
    print("\nLabel mapping:")
    for i, item in enumerate(encoder.classes_):
     print(item, '-->', i)

    # Преобразование меток с помощью кодировщика
    test_labels = ['green', 'red', 'black']
    encoded_values = encoder.transform(test_labels)
    print("\nLabels =", test_labels)
    print("Encoded values =", list(encoded_values))

    # Декодирование набора чисел с помощью декодера
    encoded_values = [3, 0, 4, 1]
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\nEncoded values =", encoded_values)
    print("Decoded labels =", list(decoded_list))

def first() :
   input_labels = ['cat', 'dog', 'rabbit', 'dog', 'cat', 'rabbit']

   # Создание кодировщика и установление соответствия между метками и числами
   encoder = preprocessing.LabelEncoder()
   encoder.fit(input_labels)

   # Вывод отображения
   print("\nLabel mapping:")
   for i, item in enumerate(encoder.classes_):
     print(item, '-->', i)

    # Преобразование меток с помощью кодировщика
   test_labels = ['rabbit', 'cat', 'dog']
   encoded_values = encoder.transform(test_labels)
   print("\nLabels =", test_labels)
   print("Encoded values =", list(encoded_values))

    # Декодирование набора чисел с помощью декодера
   encoded_values = [1, 0, 2]
   decoded_list = encoder.inverse_transform(encoded_values)
   print("\nEncoded values =", encoded_values)
   print("Decoded labels =", list(decoded_list))

def second() :
    input_labels = ['apple', 'orange', 'banana', 'banana', 'apple']

    # Создание кодировщика и установление соответствия между метками и числами
    encoder = preprocessing.LabelEncoder()
    encoder.fit(input_labels)

    # Вывод отображения
    print("\nLabel mapping:")
    for i, item in enumerate(encoder.classes_):
        print(item, '-->', i)

    # Преобразование меток с помощью кодировщика
    test_labels = ['banana', 'apple', 'orange']
    encoded_values = encoder.transform(test_labels)
    print("\nLabels =", test_labels)
    print("Encoded values =", list(encoded_values))

    # Декодирование набора чисел с помощью декодера
    encoded_values = [2, 0, 1]
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\nEncoded values =", encoded_values)
    print("Decoded labels =", list(decoded_list))

def third() :
    input_labels = ['small', 'medium', 'large', 'small', 'large', 'medium']

    # Создание кодировщика и установление соответствия между метками и числами
    encoder = preprocessing.LabelEncoder()
    encoder.fit(input_labels)

    # Вывод отображения
    print("\nLabel mapping:")
    for i, item in enumerate(encoder.classes_):
        print(item, '-->', i)

    # Преобразование меток с помощью кодировщика
    test_labels = ['medium', 'large', 'small']
    encoded_values = encoder.transform(test_labels)
    print("\nLabels =", test_labels)
    print("Encoded values =", list(encoded_values))

    # Декодирование набора чисел с помощью декодера
    encoded_values = [0, 1, 2]
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\nEncoded values =", encoded_values)
    print("Decoded labels =", list(decoded_list))

#test()
#first()
#second()
third()
