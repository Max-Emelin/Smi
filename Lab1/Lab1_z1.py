from sklearn import datasets

load_digits = datasets.load_digits()
load_linnerud = datasets.load_linnerud()
load_wine  = datasets.load_wine()

#print ("load_digits:\n",load_digits.data)
#print ("load_linnerud:\n",load_linnerud.data)
#print ("load_wine:\n",load_wine.data)

#print("load_digits.target:\n",load_digits.target)
#print("load_linnerud.target:\n",load_linnerud.target)
#print("load_wine.target:\n",load_wine.target)

#print("load_digits.data:\n",load_digits.data)
#print("load_linnerud.data:\n",load_linnerud.data)
#print("load_wine.data:\n",load_wine.data)

#print("load_digits.feature_names:\n",load_digits.feature_names)
#print("load_linnerud.feature_names:\n",load_linnerud.feature_names)
#print("load_wine.feature_names:\n",load_wine.feature_names)


#print("load_digits.DESCR:\n",load_digits.DESCR)
#print("load_linnerud.DESCR:\n",load_linnerud.DESCR)
#print("load_wine.DESCR:\n",load_wine.DESCR)

digits = datasets.load_digits()
print("digits.images[1]:\n", digits.images[1])