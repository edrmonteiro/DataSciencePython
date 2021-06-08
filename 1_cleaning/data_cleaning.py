import pandas as pd
import seaborn as srn
import statistics  as sts

dataset = pd.read_csv("Churn.csv", sep=";")
#visulizar
dataset.head()

#tamanho
dataset.shape

#primeiro problema é dar nomes as colunas
dataset.columns = ["Id","Score","Estado","Genero","Idade","Patrimonio","Saldo","Produtos","TemCartCredito",
                    "Ativo","Salario","Saiu"]

#visulizar
dataset.head()

#explorar dados categoricos
#estado
agrupado = dataset.groupby(['Estado']).size()
agrupado

agrupado.plot.bar(color = 'gray')

#genero
agrupado = dataset.groupby(['Genero']).size()
agrupado

agrupado.plot.bar(color = 'gray')

#explorar colunas numéricas
#score
dataset['Score'].describe()

srn.boxplot(dataset['Score']).set_title('Score')
srn.distplot(dataset['Score']).set_title('Score')

#idade
dataset['Idade'].describe()

srn.boxplot(dataset['Idade']).set_title('Idade')
srn.distplot(dataset['Idade']).set_title('Idade')

#saldo
dataset['Saldo'].describe()
srn.boxplot(dataset['Saldo']).set_title('Saldo')
srn.distplot(dataset['Saldo']).set_title('Saldo')

#salário
dataset['Salario'].describe()

srn.boxplot(dataset['Salario']).set_title('Salario')
srn.distplot(dataset['Salario']).set_title('Salario')


#contamos valores NAN
#genero e salário
dataset.isnull().sum()

#salarios
#remover nas e substiutir pela mediana
dataset['Salario'].describe()

mediana = sts.median(dataset['Salario'])
mediana

#substituir NAN por mediana
dataset['Salario'].fillna(mediana, inplace=True)

#Verificamos se NAN não existem mais
dataset['Salario'].isnull().sum()

#genero, falta de padronização e NAs
agrupado = dataset.groupby(['Genero']).size()
agrupado

#total de Nas
dataset['Genero'].isnull().sum()

#preenche NAs com Masculino (moda)
dataset['Genero'].fillna('Masculino', inplace=True)

#verificamos novamente NANs
dataset['Genero'].isnull().sum()

#padroniza de acordo com o dominio
dataset.loc[dataset['Genero'] ==  'M', 'Genero'] = "Masculino"
dataset.loc[dataset['Genero'].isin( ['Fem','F']), 'Genero'] = "Feminino"
#visualiza o resultado
agrupado = dataset.groupby(['Genero']).size()
agrupado

#idades fora do dominio
dataset['Idade'].describe()

#visualizar 
dataset.loc[(dataset['Idade'] <  0 )  | ( dataset['Idade'] >  120) ]

#calular a mediana
mediana = sts.median(dataset['Idade'])
mediana

#substituir
dataset.loc[(dataset['Idade'] <  0 )  | ( dataset['Idade'] >  120), 'Idade'] = mediana

#verificamos se ainda existem idades fora do domínio
dataset.loc[(dataset['Idade'] <  0 )  | ( dataset['Idade'] >  120) ]

#dados duplicados, buscamos pelo ID
dataset[dataset.duplicated(['Id'],keep=False)]

#excluimos pelo ID
dataset.drop_duplicates(subset="Id", keep='first',inplace=True)
#buscamos duplicados 
dataset[dataset.duplicated(['Id'],keep=False)]

#estado foram do domínio
agrupado = dataset.groupby(['Estado']).size()
agrupado

#atribuomos RS (moda)
dataset.loc[dataset['Estado'].isin( ['RP','SP','TD']), 'Estado'] = "RS"
agrupado = dataset.groupby(['Estado']).size()

#verificamos o resultado
agrupado

#outliers em salário, vamos considerar 2 desvios padrão
desv = sts.stdev(dataset['Salario'])
desv

#definir padrão como maior que 2 desvios padrão
#checamos se algum atende critério
dataset.loc[dataset['Salario'] >=  2 * desv ] 

#vamos atualiar salarios para mediana, calculamos
mediana = sts.median(dataset['Salario'])
mediana

#atribumos
dataset.loc[dataset['Salario'] >=  2 * desv, 'Salario'] = mediana
#checamos se algum atende critério
dataset.loc[dataset['Salario'] >=  2 * desv ] 

dataset.head()

dataset.shape


dataset = pd.read_csv("tempo.csv", sep=";")
#visulizar
dataset.head()
#explorar dados categoricos
#aparencia
agrupado = dataset.groupby(['Aparencia']).size()
agrupado

agrupado.plot.bar(color = 'gray')
#aparencia
agrupado = dataset.groupby(['Vento']).size()
agrupado

agrupado.plot.bar(color = 'gray')
#jogar
agrupado = dataset.groupby(['Jogar']).size()
agrupado
agrupado.plot.bar(color = 'gray')
#explorar colunas numéricas
#temperatura
dataset['Temperatura'].describe()
srn.boxplot(dataset['Temperatura']).set_title('Temperatura')
srn.distplot(dataset['Temperatura']).set_title('Temperatura')

#Umidade
dataset['Umidade'].describe()
srn.boxplot(dataset['Umidade']).set_title('Umidade')
srn.distplot(dataset['Umidade']).set_title('Umidade')

#contamos valores NAN
dataset.isnull().sum()

#aparencia valor invalido
agrupado = dataset.groupby(['Aparencia']).size()
agrupado

dataset.loc[dataset['Aparencia'] ==  'menos', 'Aparencia'] = "Sol"
#visualiza o resultado
agrupado = dataset.groupby(['Aparencia']).size()
agrupado

#temperatura fora do dominio
dataset['Temperatura'].describe()

#visualizar 
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130) ]

#calular a mediana
mediana = sts.median(dataset['Temperatura'])
mediana

#substituir
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130), 'Temperatura'] = mediana

#verificamos se ainda existem #verificamos se ainda existem idades fora do domínio
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130) ]

#umidade, dominio e NAs
agrupado = dataset.groupby(['Umidade']).size()
agrupado

#total de Nas
dataset['Umidade'].isnull().sum()

#calular a mediana
mediana = sts.median(dataset['Umidade'])
mediana

#preenche NAs
dataset['Umidade'].fillna(mediana, inplace=True)

dataset['Umidade'].isnull().sum()

#visuliza de acordo com o dominio de acordo com o dominio
dataset.loc[(dataset['Umidade'] <  0 )  | ( dataset['Umidade'] >  100) ]

#atualiza comm mediana
dataset.loc[(dataset['Umidade'] <  0 )  | ( dataset['Umidade'] >  100), 'Umidade'] = mediana

#visuliza novamente
dataset.loc[(dataset['Umidade'] <  0 )  | ( dataset['Umidade'] >  100) ]

#Vento
agrupado = dataset.groupby(['Vento']).size()
agrupado

#total de Nas
dataset['Vento'].isnull().sum()

#preenche NAs
dataset['Vento'].fillna('FALSO', inplace=True)

#total de Nas
dataset['Vento'].isnull().sum()






















