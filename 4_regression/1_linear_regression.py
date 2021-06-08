"""
Regressão linear simples
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot 

import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

base = pd.read_csv(path + 'cars.csv')
base.shape

base.head()

base = base.drop(['Unnamed: 0'], axis = 1)
base.head()

# Definição das variáveis X e Y, X distância é a variável independente e Y velocidade é a variável dependente
X = base.iloc[:, 1].values
y = base.iloc[:, 0].values
X

# Cálculo da correlação entre X e Y
correlacao = np.corrcoef(X, y)
correlacao

#formato de matriz com uma coluna a mais
X = X.reshape(-1, 1) 
# Criação do modelo e treinamento (fit indica que o treinamento deve ser executado)
modelo = LinearRegression()
modelo.fit(X, y)

# Visualização dos coeficientes
modelo.intercept_

#inclinacao
modelo.coef_

# Geração do gráfico com os pontos reais e as previsões
plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color = 'red')

# Previsão da "distância 22 pés" usando a fórmula manual
# interceptação * inclinação * valor de dist
#Qual velocidade se levou 22 pés pra parar?
modelo.intercept_ + modelo.coef_ * 22

# Previsão utilizando função do sklearn
modelo.predict([[22]])

# Gráfico para visualizar os residuais
visualizador = ResidualsPlot(modelo)
visualizador.fit(X, y)
visualizador.poof()



"""
Regressão linear múltipla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

base = pd.read_csv(path + 'mt_cars.csv')
base.shape

#exclui coluna
base = base.drop(['Unnamed: 0'], axis = 1)

# Criação de X e Y: variável independente e variável dependente
# Cálculo da correlação entre X e Y
X = base.iloc[:, 2].values #coluna disp
y = base.iloc[:, 0].values #coluna mpg
correlacao = np.corrcoef(X, y)
correlacao

# Mudança do formato de X para o formato de matriz (necessário para versões mais recentes do sklearn)
X = X.reshape(-1, 1)

# Criação do modelo, treinamento, visualização dos coeficientes e do score do modelo
modelo = LinearRegression()
modelo.fit(X, y)

#Interceptação
modelo.intercept_

#inclinação
modelo.coef_

#score R^2
modelo.score(X, y)

# Geração das previsões
previsoes = modelo.predict(X)
previsoes

# Criação do modelo, utilizando a biblioteca statsmodel 
#podemos ver r ajustadodo r2
modelo_ajustado = sm.ols(formula = 'mpg ~ disp', data = base)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

# Visualização dos resultados
plt.scatter(X, y)
plt.plot(X, previsoes, color = 'red')

# Previsão para somente um valor
modelo.predict([[200]])

# Criação de novas variáveis X1 e Y1 e novo modelo para comparação com o anterior
# 3 variáveis dependentes para prever mpg: cyl	disp	hp
X1 = base.iloc[:, 1:4].values
X1

y1 = base.iloc[:, 0].values
modelo2 = LinearRegression()
modelo2.fit(X1, y1)
#R^2
modelo2.score(X1, y1)

# Criação do modelo ajustado com mais atributos (regressão linear múltipla)
#usando stats models
modelo_ajustado2 = sm.ols(formula = 'mpg ~ cyl + disp + hp', data = base)
modelo_treinado2 = modelo_ajustado2.fit()
modelo_treinado2.summary()

# Previsão de um novo registro
novo = np.array([4, 200, 100])
novo = novo.reshape(1, -1)
modelo2.predict(novo)

modelo_treinado2.predict(novo)


"""
Regressão Linear, training
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

base = pd.read_csv(path + 'slr12.csv', sep=';')
base.shape
base.head()

# Definição das variáveis X e Y, X FrqAnual é a variável independente e Y CusInic é a variável dependente
X = base.iloc[:, 0].values
y = base.iloc[:, 1].values
X

# Cálculo da correlação entre X e Y
correlacao = np.corrcoef(X, y)
correlacao

#formato de matriz com uma coluna a mais
X = X.reshape(-1, 1) 
# Criação do modelo e treinamento (fit indica que o treinamento deve ser executado)
modelo = LinearRegression()
modelo.fit(X, y)

# Geração do gráfico com os pontos reais e as previsões
plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color = 'red')

#valor anual da franquina
valr =  1300
modelo.predict([[valr]])








