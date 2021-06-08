"""
Redes neurais artificiais com keras
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

# Carregamento da base de dados e criação dos previsores (variáveis independentes - X) e da classe (variável dependente - y)
base = datasets.load_iris()
previsores = base.data
classe = base.target
classe

# Transformação da classe para o formato "dummy", pois temos uma rede neural com 3 neurônios na camada de saída
classe_dummy = np_utils.to_categorical(classe)
classe_dummy

# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe_dummy,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

# Criação da estrutura da rede neural com a classe Sequential (sequência de camadas)
modelo = Sequential()
#primeira camada oculta, 5 neuronios, 4 neuronios de entrada
modelo.add(Dense(units = 5, input_dim = 4))
#segunda camada oculta
modelo.add(Dense(units = 4))
# Função softmax porque temos um problema de classificação com mais de duas classes 
#(é gerada uma probabilidade em cada neurônio)
modelo.add(Dense(units = 3, activation = 'softmax'))

# Visualização da estrutura da rede neural
modelo.summary()

# Configuração dos parâmetros da rede neural (adam = algoritmo para atualizar os pesos e loss = cálculo do erro)
modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
# Treinamento, dividindo a base de treinamento em uma porção para validação (validation_data)
modelo.fit(X_treinamento, y_treinamento, epochs = 1000,
           validation_data = (X_teste, y_teste))

# Previsões e mudar a variável para True ou False de acordo com o threshold 0.5
previsoes = modelo.predict(X_teste)
previsoes = (previsoes > 0.5)
previsoes

# Como é um problema com três saídas, precisamos buscar a posição que possui o maior valor (são retornados 3 valores)
y_teste_matrix = [np.argmax(t) for t in y_teste]
y_previsao_matrix = [np.argmax(t) for t in previsoes]

# Geração da matriz de confusão
confusao = confusion_matrix(y_teste_matrix, y_previsao_matrix)
confusao



