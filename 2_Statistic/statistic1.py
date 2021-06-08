"""
Amostragem Simples
"""
import pandas as pd
import numpy as np
import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

base = pd.read_csv(path + 'iris.csv')
print(base)
print(base.shape)

np.random.seed(2345)

# 150 amostras, de 0 a 1, com reposição, probabilidades equivalentes
amostra = np.random.choice(a = [0, 1], size = 150, replace = True, p = [0.7, 0.3])
print(amostra)

base_final = base.loc[amostra == 0]
print(base_final.shape)

base_final2 = base.loc[amostra == 1]
print(base_final2.shape)


"""
Amostragem estratificada
"""
import pandas as pd
from sklearn.model_selection import train_test_split

iris = pd.read_csv(path + 'iris.csv')
print(iris['class'].value_counts())

X, _, y, _ = train_test_split(iris.iloc[:, 0:4], iris.iloc[:, 4], test_size = 0.5, stratify = iris.iloc[:,4])
print(y.value_counts())

infert = pd.read_csv(path + 'infert.csv')
print(infert)
print(infert['education'].value_counts())

# Criando uma amostra com somente 40% dos registros (por isso é definido 0.6, pois é gerado o inverso)
X1, _, y1, _ = train_test_split(infert.iloc[:, 2:9], infert.iloc[:, 1], test_size = 0.6, stratify = infert.iloc[:, 1])
y1.value_counts()

"""
Amostragem sistemática
"""
import numpy as np
import pandas as pd
from math import ceil

# Criação das variáveis para representar a população, a amostra e o valor de k
populacao = 150
amostra = 15
k = ceil(populacao / amostra)
print(k)

# Definição do valor randômico para inicializar a amostra, iniciando em 1 até k + 1
r = np.random.randint(low = 1, high = k + 1, size = 1)
print(r)

# Criamos um for para somar os próximos valores, baseado no primeiro valor r que foi definido acima
acumulador = r[0]
sorteados = []
for i in range(amostra):
    #print(acumulador)
    sorteados.append(acumulador)
    acumulador += k
print(sorteados)

len(sorteados)

# Carregamos a base de dados e criamos a base_final somente com os valores sorteados
base = pd.read_csv(path + 'iris.csv')
base_final = base.loc[sorteados]
print(base_final)


"""
Medidas de centralidade e variabilidade
"""
import numpy as np
from scipy import stats

# Criação da variável com os dados dos jogadores, visualização da mediana e média
jogadores = [40000, 18000, 12000, 250000, 30000, 140000, 300000, 40000, 800000]
print(np.mean(jogadores))
print(np.median(jogadores))
# Criação da variável para geração dos quartis (0%, 25%, 50%, 75% e 100%) 
quartis = np.quantile(jogadores, [0, 0.25, 0.5, 0.75, 1])
print(quartis)

#visualização do desvio padrão
np.std(jogadores, ddof = 1)

# Visualização de estatísticas mais detalhadas usando a biblioteca scipy
print(stats.describe(jogadores))

"""
Distribuição normal
"""
from scipy.stats import norm

# Conjunto de objetos em uma cesta, a média é 8 e o desvio padrão é 2
# Qual a probabilidade de tirar um objeto que peso é menor que 6 quilos?
print(norm.cdf(6, 8, 2))

# Qual a probabilidade de tirar um objeto que o peso á maior que 6 quilos?
#norm.sf(6, 8, 2)
print(1 - norm.cdf(6, 8, 2))

# Qual a probabilidade de tirar um objeto que o peso é menor que 6 ou maior que 10 quilos?
print(norm.cdf(6, 8, 2) + norm.sf(10, 8, 2))

# Qual a probabilidade de tirar um objeto que o peso é menor que 10 e maior que 8 quilos?
print(norm.cdf(10, 8, 2) - norm.cdf(8, 8, 2))

"""
Teste distribuição normal
"""
from scipy import stats
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt

# Criação de uma variável com dados em uma distribuição normal com a função rvs (100 elementos)
dados = norm.rvs(size = 1000)
print(dados)

#histograma
plt.hist(dados, bins = 20)
plt.title('Dados')

# Geração de gráfico para verificar se a distribuição é normal
fig, ax = plt.subplots()
stats.probplot(dados, fit=True,   plot=ax)
plt.show()

# Execução do teste de Shapiro
#segundo argumento é o valor de p, não há como rejeitar a hipótese nula
stats.shapiro(dados)

"""
Dados não normais
"""
dados2 = skewnorm.rvs(4, size=1000)
#histograma
plt.hist(dados2, bins = 20)
plt.title('Dados')
# Geração de gráfico para verificar se a distribuição é normal
fig, ax = plt.subplots()
stats.probplot(dados2, fit=True,   plot=ax)
plt.show()
stats.shapiro(dados2)




stop = 1
