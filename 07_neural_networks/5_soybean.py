"""
Faça Você Mesmo ML e RNA
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

base = pd.read_csv(path + 'soybean.csv')
base.head()

base.shape

# Criação da variável X que presenta os atributos previsores
X = base.iloc[:, 0:35].values

# Criação da variável y que contém as respostas
y = base.iloc[:, 35].values

labelencoder = LabelEncoder()

for x in range(35):
    X[:, x] = labelencoder.fit_transform(X[:, x])

# Divisão da base em treino e teste (70% para treinamento e 30% para teste)
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, 
                                                                  test_size=0.3,
                                                                  random_state=0)

 # Criação do classificador Random Forest
from sklearn.ensemble import RandomForestClassifier
floresta = RandomForestClassifier(n_estimators = 100)
floresta.fit(X_treinamento, y_treinamento)

# Previsões
previsoes = floresta.predict(X_teste)
previsoes

from sklearn.metrics import confusion_matrix, accuracy_score
matriz = confusion_matrix(y_teste, previsoes)
matriz

taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_acerto

taxa_erro = 1 - taxa_acerto
taxa_erro
