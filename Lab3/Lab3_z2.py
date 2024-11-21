import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Входной файл, содержащий данные
input_file = 'Lab3\income_data.txt'

# Чтение данных
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Преобразование в массив numpy
X = np.array(X)

# Преобразование строковых данных в числовые
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Создание SVМ-классификатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Обучение классификатора
classifier.fit(X, y)

# Перекрёстная проверка
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Вычисление F-меры для SVМ-классификатора
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Предсказание результата для тестовой точки данных
#input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']
#input_data = ['48', 'Local-gov', '67229', 'Masters', '14', 'Separated', 'Adm-clerical', 'Unmarried', 'Black', 'Female', '0', '0', '40', 'United-States']
#input_data = ['45', 'Local-gov', '185588', 'Bachelors', '13', 'Divorced', 'Prof-specialty', 'Unmarried', 'White', 'Female', '0', '0', '40', 'United-States']
input_data = ['50', 'Private', '175029', 'Assoc-voc', '11', 'Married-civ-spouse', 'Prof-specialty', 'Husband', 'White', 'Male', '0', '0', '55', 'United-States']



# Кодирование тестовой точки данных
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count += 1
input_data_encoded = np.array([input_data_encoded])


# Волнение классификатора для кодированной точки данных и вывод результата 
predicted_class = classifier.predict(input_data_encoded)
print("test_size: ", 0.9)
print(label_encoder[-1].inverse_transform(predicted_class)[0])
