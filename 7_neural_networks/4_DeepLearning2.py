"""
Deep Learning
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

dataset = pd.read_csv(path + "Credit2.csv", sep=";")
dataset

#separação dos variáveis, ignoro primeira pois não tem valor semântico
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:, 10].values
#temos um arry e não mais um data frame
X

#label encoder coluna checking_status
#atribui valores de zero a 3
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
X
#one hot encoder coluna credit_history
#deve adicionar 5 colunas
onehotencoder = make_column_transformer((OneHotEncoder(categories='auto', sparse=False), [1]), remainder="passthrough")
X = onehotencoder.fit_transform(X)
X

#Excluimos a variável para evitar a dummy variable trap X = X:,1: X
#Laber encoder com a classe
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
y

#separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(len(X_train),len(X_test),len(y_train),len(y_test))

#Feature Scalling, Padronização z-score
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred

#matriz de confusão
cm = confusion_matrix(y_test, y_pred)
cm




