import numpy as np
from sklearn import preprocessing

input_data1 = np.array([ 
    [5.1,   -2.9,    3.3,   15,     78.3],
    [-1.2,  7.8,    -6.1,   78,     -55.1],
    [3.9,   0.4,    2.1,    564.9,  0]
])

input_data2 = np.array([ 
    [3.7, -1.5,  6.2,  8.1,  12.9],
    [-7.3,  5.8, -3.9,  14.2, -4.5],
    [9.0,  2.1,  0.6,  5.6,  -1.2]
])

input_data3 = np.array([ 
    [145.2, -300.7,  89.3,  512.9, -132.4],
    [-98.4,  230.8,  -56.2,  -12.7,  402.5],
    [322.1, -43.9,   114.6,  -89.1,  260.0]
])

# Бинаризация данных
#data_binarized = preprocessing.Binarizer (threshold= 15).transform (input_data)
#print ("\nБинаризированные данные:\n", data_binarized)

# Вывод среднего значения и стандартного отклонения
#print ("\nДО:")
#print ("Среднее =", input_data3.mean (axis=0))
#print ("СКО =", input_data3.std (axis=0))

# Исключение среднего
#data_scaled = preprocessing.scale (input_data3)
#print ("\nПОСЛЕ:")
#print ("Среднее =", data_scaled.mean(axis=0))
#print ("СКО =", data_scaled.std(axis=0))

# Масштабирование MinMax
#data_scaler_minmax = preprocessing.MinMaxScaler (feature_range=(8, 60))
#data_scaled_minmax = data_scaler_minmax.fit_transform (input_data1)
#print ("\nМасштабированные данные(8, 60) :\n", data_scaled_minmax)


# Нормализация данных
data_normalized_l1 = preprocessing.normalize(input_data3, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data3, norm='l2')
print ("\nLl-нормализация набор 3:\n", data_normalized_l1)
print ("\nL2-нормализация набор 3:\n", data_normalized_l2)