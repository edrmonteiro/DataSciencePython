"""
#Formação Cientista de Dados - Fernando Amaral e Jones Granatyr

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

base = pd.read_csv(path + 'trees.csv')
base.shape
base.head()
# Criação do histograma, considerando somente o segundo atributo da base de dados e com duas divisões (bins)
# A variável 'h' armazena as faixas de valores de Height
h = np.histogram(base.iloc[:,1], bins = 6)
h
# Visualização do histograma com 6 divisões (bins)
plt.hist(base.iloc[:,1], bins = 6)
plt.title('Árvores')
plt.ylabel('Frequência')
plt.xlabel('Altura')



"""
Gráficos de densidade
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base = pd.read_csv(path + 'trees.csv')
base.head()

# Histograma com 10 divisões (bins) e somente para o primeiro atributo da base de dados
plt.hist(base.iloc[:,1], bins = 6)
# Histograma com a linha de distribuição de frequência, com 6 divisões (bins)
#kde = linha de densidade
sns.distplot(base.iloc[:,1], hist = True, kde = False,
             bins = 6, color = 'blue',
             hist_kws={'edgecolor': 'black'})
#densidade
sns.distplot(base.iloc[:,1], hist = False, kde = True,
             bins = 6, color = 'blue',
             hist_kws={'edgecolor': 'black'})

#densidade e histograma
sns.distplot(base.iloc[:,1], hist = True, kde = True,
             bins = 6, color = 'blue',
             hist_kws={'edgecolor': 'black'})          



"""
Gráfico de dispersão
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento da base de dados
base = pd.read_csv(path + 'trees.csv')
base.head()
# Gráfico de dispersão considerando o volume e a dispersão
plt.scatter(base.Girth, base.Volume, color = 'blue', facecolors = 'none', marker = '*')
plt.title('Árvores')
plt.xlabel('Volume')
plt.ylabel('Circunferência')
# Gráfico de linha considerando o volume e o atributo "girth"
plt.plot(base.Girth, base.Volume)
plt.title('Árvores')
plt.xlabel('Volume')
plt.ylabel('Circunferência')

# Gráfico de dispersão com 'afastamento' dos dados (jitter)
#fit_reg linha de tendência
sns.regplot(base.Girth, base.Volume, data = base, x_jitter = 0.3, fit_reg = False)


"""
Gráfico de dispersão com legenda
"""
import pandas as pd
import matplotlib.pyplot as plt


base = pd.read_csv(path + 'co2.csv')
base.head()

#criando duas variáveis para cada atributo (x = conc e y = uptake)
x = base.conc
y = base.uptake

# Retorna os valores únicos do atributo "treatment"
unicos = list(set(base.Treatment))
unicos

# Percorre cada tipo de tratamento (chilled e nonchilled) e cria um gráfico de dispersão
for i in range(len(unicos)):
    indice = base.Treatment == unicos[i]
    plt.scatter(x[indice], y[indice], label = unicos[i])
plt.legend(loc = 'lower right')


"""
Divisão da tela (subgráficos)
"""
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv(path + 'trees.csv')
base.head()

# girth com volume
plt.scatter(base.Girth, base.Volume)

# girth com heigth
plt.scatter(base.Girth, base.Height)

# height com volume
plt.scatter(base.Height, base.Volume, marker = '*')

# histograma volume
plt.hist(base.Volume)

#imprimindo juntos
# Criação de figura, no qual os gráficos serão posicionados
plt.figure(1)
plt.subplot(2,2,1)
plt.scatter(base.Girth, base.Volume)
plt.subplot(2,2,2)
plt.scatter(base.Girth, base.Height)
plt.subplot(2,2,3)
plt.scatter(base.Height, base.Volume, marker = '*')
plt.subplot(2,2,4)
plt.hist(base.Volume)



"""
Boxplot
"""
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv(path + 'trees.csv')
base.head()

# Geração do boxplot
# patch_artist = True preenche, showfliers outliers
plt.boxplot(base.Volume, vert = False, showfliers = False, notch = True,patch_artist = True)
plt.title('Árvores')
plt.xlabel('Volume')

#dados por linha
plt.boxplot(base)
plt.title('Árvores')
plt.xlabel('Dados')

# Geração de 3 boxplots, cada um mostrado informações diferentes
plt.boxplot(base.Volume, vert = False)
plt.boxplot(base.Girth, vert = False)
plt.boxplot(base.Height, vert = False)
plt.title('Árvores')
plt.xlabel('Dados')



"""
Gráfico de barras e setores
"""
import pandas as pd

base = pd.read_csv(path + 'insect.csv')
base.shape

#dados
base.head()

# Agrupamento dos dados baseado no atributo 'spray', contando e somando os registros
agrupado = base.groupby(['spray'])['count'].sum()
agrupado

# Gráfico de barras
agrupado.plot.bar(color = 'gray')

#cores
agrupado.plot.bar(color = ['blue','yellow','red','green','pink','orange'])

#cores
agrupado.plot.bar(color = ['blue','yellow','red','green','pink','orange'])

# Gráfico de pizza
agrupado.plot.pie()

#com legenda
agrupado.plot.pie(legend = True)



"""
Boxplot com Seaborn
"""
import pandas as pd
import seaborn as srn

base = pd.read_csv(path + 'trees.csv')
base.head()
# Visualização de um boxplot
srn.boxplot(base.Volume).set_title('Árvores')
# Visualização de vários boxplots na mesma imagem
srn.boxplot(data = base)



"""
Histograma com densidade e seabord
"""
import pandas as pd
import seaborn as srn
import matplotlib.pyplot as plt

base = pd.read_csv(path + 'trees.csv')
base.head()
# Histograma com 10 divisões (bins) e com gráfico de densidade
srn.distplot(base.Volume, bins = 10, axlabel = 'Volume').set_title('Árvores')

# Carregamento de outra base de dados
base2 = pd.read_csv(path + 'chicken.csv')
base2.head()

# Criação de novo dataframe agrupando o atributo 'feed'
agrupado = base2.groupby(['feed'])['weight'].sum()
agrupado

# Novo dataframe somente para testar os filtros do pandas
teste = base2.loc[base2['feed'] == 'horsebean']
teste

# Novo dataframe somente para testar os filtros do pandas
teste = base2.loc[base2['feed'] == 'horsebean']
teste

# Histograma considerando somente o valor 'horsebean'
srn.distplot(base2.loc[base2['feed'] == 'horsebean'].weight, hist = False).set_title('horsebean')

# Histograma considerando somente o valor 'casein'
srn.distplot(base2.loc[base2['feed'] == 'casein'].weight).set_title('casein')

# Histograma considerando somente o valor 'linseed'
srn.distplot(base2.loc[base2['feed'] == 'linseed'].weight).set_title('linseed')

# Histograma considerando somente o valor 'meatmeal'
srn.distplot(base2.loc[base2['feed'] == 'meatmeal'].weight).set_title('meatmeal')

# Histograma considerando somente o valor 'soybean'
srn.distplot(base2.loc[base2['feed'] == 'soybean'].weight).set_title('soybean')

# Histograma considerando somente o valor 'sunflower'
srn.distplot(base2.loc[base2['feed'] == 'sunflower'].weight).set_title('sunflower')

#impressão em gráfico 2 x 3
plt.figure()
plt.subplot(3,2,1)
srn.distplot(base2.loc[base2['feed'] == 'horsebean'].weight, hist = False).set_title('horsebean')
plt.subplot(3,2,2)
srn.distplot(base2.loc[base2['feed'] == 'casein'].weight).set_title('casein')
plt.subplot(3,2,3)
srn.distplot(base2.loc[base2['feed'] == 'linseed'].weight).set_title('linseed')
plt.subplot(3,2,4)
srn.distplot(base2.loc[base2['feed'] == 'meatmeal'].weight).set_title('meatmeal')
plt.subplot(3,2,5)
srn.distplot(base2.loc[base2['feed'] == 'soybean'].weight).set_title('soybean')
plt.subplot(3,2,6)
srn.distplot(base2.loc[base2['feed'] == 'sunflower'].weight).set_title('sunflower')
#ajusta o layout para não haver sobreposição
plt.tight_layout()



"""
Gráfico de dispersão com seaborn
"""
import pandas as pd
import seaborn as srn
import matplotlib.pyplot as plt

base = pd.read_csv(path + 'co2.csv')
base.head()
# Gráfico de dispersão utilizando os atributos conc e uptake, agrupamento pelo type
srn.scatterplot(base.conc, base.uptake, hue = base.Type)

# Seleção de registros específicos da base de dados (Quebec e Mississipi)
q = base.loc[base['Type'] == 'Quebec']
m = base.loc[base['Type'] == 'Mississippi']

# Subgráfico (1 linha e duas colunas) mostrando gráficos sobre cada região
plt.figure()
plt.subplot(1,2,1)
srn.scatterplot(q.conc, q.uptake).set_title('Quebec')
plt.subplot(1,2,2)
srn.scatterplot(m.conc, m.uptake).set_title('Mississippi')
plt.tight_layout()


# Refrigerado e não refrigerado
ch = base.loc[base['Treatment'] == 'chilled']
nc = base.loc[base['Treatment'] == 'nonchilled']

# Gráfico somente com 'chilled' e 'nonchilled'
plt.figure()
plt.subplot(1,2,1)
srn.scatterplot(ch.conc, ch.uptake).set_title('Chilled')
plt.subplot(1,2,2)
srn.scatterplot(nc.conc, nc.uptake).set_title('Non chilled')
plt.tight_layout()

# Carregamento de outro arquivo, cancer de esofago
base2 = pd.read_csv(path + 'esoph.csv')
base2

# Gráfico entre os atributos 'alcgp' e 'ncontrols'
srn.catplot(x = 'alcgp', y = 'ncontrols', data = base2, jitter = False)

# Gráfico entre os atributos 'alcgp' e 'ncontrols', com agrupamento
srn.catplot(x = 'alcgp', y = 'ncontrols', data = base2, col = 'tobgp')



"""
Faça você mesmo - Visualização
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as srn

base = pd.read_csv(path + 'dados.csv', sep=';')
base.shape

base.head()

srn.distplot(base.PIB, bins = 10, axlabel = 'PIB').set_title('PIB')

srn.distplot(base.VALOREMPENHO, bins = 10, axlabel = 'Valor do Empenho').set_title('Empenho')

agrupado = base.sort_values('PIB').head(10)
agrupado = agrupado.iloc[:,1:3]
agrupado
agrupado.plot.bar(x='MUNICIPIO',y='PIB', color = 'gray')

agrupado = base.sort_values('VALOREMPENHO').head(10)
agrupado = agrupado.iloc[:,[1,3]]
agrupado
agrupado.plot.bar(x='MUNICIPIO',y='VALOREMPENHO', color = 'gray')













