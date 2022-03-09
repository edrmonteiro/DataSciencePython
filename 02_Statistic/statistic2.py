"""
Distribuição T Student
"""
import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

from scipy.stats import t
# Média de salário dos cientistas de dados = R$ 75,00 por hora Amostra com 9 funcionários e desvio padrão = 10
# Qual a probabilidade de selecionar um cientista de dados e o salário ser menor que R$ 80 por hora
t.cdf(1.5, 8)
# Qual a probabilidade do salário ser maior do que 80?
t.sf(1.5, 8)
# Somatório da execução dos dois códigos acima (lado esquerdo + lado direito da distribuição)
t.cdf(1.5, 8) + t.sf(1.5, 8)

"""
Distribuição binomial
"""
from scipy.stats import binom
# Jogar uma moeda 5 vezes, qual a probabilidade de dar cara 3 vezes?
# eventos , experimentos, probabilidades
prob = binom.pmf(3, 5, 0.5)
prob
# Passar por 4 sinais de 4 tempos, qual a probabilidade de pegar sinal verde
# nenhuma, 1, 2, 3 ou 4 vezes seguidas?
binom.pmf(0, 4, 0.25) + binom.pmf(1, 4, 0.25) + binom.pmf(2, 4, 0.25) + binom.pmf(3, 4, 0.25) + binom.pmf(4, 4, 0.25)
# Passar por 4 sinais de 4 tempos, qual a probabilidade de pegar sinal verde
# nenhuma, 1, 2, 3 ou 4 vezes seguidas?
binom.pmf(0, 4, 0.25) + binom.pmf(1, 4, 0.25) + binom.pmf(2, 4, 0.25) + binom.pmf(3, 4, 0.25) + binom.pmf(4, 4, 0.25)
# E se forem sinais de dois tempos?
binom.pmf(4, 4, 0.5)
# Probabilidade acumulativa
binom.cdf(4, 4, 0.25)
# Concurso com 12 questões, qual a probabilidade de acertar 7 questões considerando
# que cada questão tem 4 alternativas?
binom.pmf(7, 12, 0.25)
# Probabilidade de acertar as 12 questões
binom.pmf(12, 12, 0.25) 


"""
Distribuição de Poisson
"""
from scipy.stats import poisson
# Qual a probabilidade de ocorrerem 3 acidentes no dia?
poisson.pmf(3, 2)
# Qual a probabilidade de ocorrerem 3 ou menos acidentes no dia?
poisson.cdf(3, 2)
# Qual a probabilidade de ocorrerem mais de 3 acidentes no dia?
poisson.sf(3, 2)
poisson.cdf(3, 2) + poisson.sf(3, 2)


"""
Qui Quadrado
"""
import numpy as np
from scipy.stats import chi2_contingency
# Criação da matriz com os dados e execução do teste
novela = np.array([[19, 6], [43, 32]])
novela
#segundo valor é o pvalue
#Valor de p é maior que 0,05 não há evidências de diferença significativa (hipótese nula): não há diferença significativa
chi2_contingency(novela)
novela2 = np.array([[22, 3], [43, 32]])
novela2
#agora valor de p menor que 0,05, podemos rejeitar a hipótese nula em favor da hipótese alternativa: há diferença significativa
chi2_contingency(novela2)


"""
Anova
"""
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

# Carregamento da base de dados
tratamento = pd.read_csv(path + 'anova.csv', sep = ';')
tratamento.head()
# Boxplot agrupando os dados pelo remédio
tratamento.boxplot(by = 'Remedio', grid = False)

# Criação do modelo de regressão linear e execução do teste
modelo1 = ols('Horas ~ Remedio', data = tratamento).fit()
resultados1 = sm.stats.anova_lm(modelo1)
# Observar valor de p maior que 0,05 (Pr(>F)) Hipótese nula de que não há diferença significativa
resultados1

# Criação do segundo modelo utilizando mais atributos e execução do teste
modelo2 = ols('Horas ~ Remedio * Sexo', data = tratamento).fit()
resultados2 = sm.stats.anova_lm(modelo2)
#Nenhum valor de P mostra diferença significativa
resultados2


#Se houver diferença o teste de Tukey é executado
# Execução do teste de Tukey e visualização dos gráficos com os resultados
mc = MultiComparison(tratamento['Horas'], tratamento['Remedio'])
resultado_teste = mc.tukeyhsd()
print(resultado_teste)
resultado_teste.plot_simultaneous()