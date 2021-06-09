"""
Igraph
"""
from igraph import Graph
from igraph import plot
import os
path = os.path.abspath(os.getcwd()) + r"/0_dataset/"

# Definição do grafo com as arestas
grafo1 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = True)
# Definição do rótulo de cada vértice
grafo1.vs['label'] = range(grafo1.vcount())
print(grafo1)

#gráfico
plot(grafo1, bbox = (0,0,300,300))

# Criação do segundo grafo
grafo2 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(0,3),(3,2),(2,1),(1,0)], directed = True)
grafo2.vs['label'] = range(grafo2.vcount())
plot(grafo2, bbox = (0,0,400,400))

# Grafo com Laço
grafo3 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(1,1)], directed = True)
grafo3.vs['label'] = range(grafo3.vcount())
plot(grafo3, bbox = (0,0,300,300))

# Criação do quarto grafo
grafo4 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(1,1)], directed = True)
# adicionamos vertice isolado
grafo4.add_vertex(5)
grafo4.vs['label'] = range(grafo4.vcount())
plot(grafo4, bbox = (0,0,300,300))

"""
ex2
"""
from igraph import Graph
from igraph import plot
# Grafo direcionado
grafo1 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = True)
grafo1.vs['label'] = range(grafo1.vcount())
print(grafo1)

# Criação Não direcionado
grafo2 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = False)
grafo2.vs['label'] = range(grafo2.vcount())
print(grafo2)

#gráfico
plot(grafo2, bbox=(0,0,300,300))

# Adicionando vértices e arestas por meio das funções add_vertices e add_vertex
grafo3 = Graph(directed = False)
grafo3.add_vertices(10)
#grafo3.add_vertex(16)
grafo3.add_edges([(0,1),(2,2),(2,3),(3,0)])
grafo3.vs['label'] = range(grafo3.vcount())
print(grafo3)
plot(grafo3, bbox=(0,0,300,300))

# Criação do quarto grafo, configurando rótulos personalizados para os vértices
grafo4 = Graph(directed = False)
grafo4.add_vertices(5)
grafo4.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(2,1)])
grafo4.add_vertex(5)
grafo4.add_vertex(6)
grafo4.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

plot(grafo4, bbox=(0,0,300,300))


"""
recriamos o grafo 4
"""
from igraph import Graph
from igraph import plot
grafo4 = Graph(directed = False)
grafo4.add_vertices(5)
grafo4.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(2,1)])
grafo4.add_vertex(5)
grafo4.add_vertex(6)
grafo4.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grafo4.vs['name'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Impressão da matriz de adjacência
print(grafo4.get_adjacency())

# linha
grafo4.get_adjacency()[0,]

#linha e coluna
grafo4.get_adjacency()[0,1]

# Estrutura de repetição para percorrer cada vértice, visualizando o nome e o rótulo
for v in grafo4.vs:
   print(v) 

plot(grafo4, bbox=(0,0,300,300))

# Percorrer os vértices para visualizar os pesos
for v in grafo4.vs:
    print(v)

grafo4.vs[0]

# Definição do tipo de amizado e do peso das relações
grafo4.es['TipoAmizade'] = ['Amigo', 'Inimigo', 'Inimigo', 'Amigo']
grafo4.es['weight'] = [1,2,1,3]
print(grafo4)

# Percorrer os vértices, tipo de amizade
for e in grafo4.es:
    print(e)

#propriedades e valores de uma posição
grafo4.es[0]

# tipos de amizade
grafo4.es['TipoAmizade']

print(grafo4)

# Mudança dos tipos das relações em grafos já existentes
grafo4.vs['type'] = 'Humanos'
grafo4.vs['name'] = 'Amizades'
print(grafo4)
plot(grafo4, bbox=(0,0,300,300))

"""
impressão
"""
from igraph import Graph
from igraph import plot

# Criação de grafo com pesos entre as relações
grafo5 = Graph(edges = [(0,1),(2,3),(0,2),(0,3)], directed = True)
grafo5.vs['label'] = ['Fernando', 'Pedro', 'Jose', 'Antonio']
grafo5.vs['peso'] = [40,30,30,25]
grafo5.es['TipoAmizade'] = ['Amigo', 'Inimigo', 'Inimigo', 'Amigo']
grafo5.es['weight'] = [1,2,1,3]

# Visualizar informações sobre os vértices
for v in grafo5.vs:
    print(v)

# Visualizar informações sobre as arestas
for e in grafo5.es:
    print(e)

# Definição de cores para os vértices
grafo5.vs['cor'] = ['blue', 'red', 'yellow', 'green']
plot(grafo5, bbox=(0,0,300,300),vertex_color = grafo5.vs['cor'])

# pesos para as arrestas
plot(grafo5, bbox=(300,300),  edge_width = grafo5.es['weight'],
     vertex_color = grafo5.vs['cor'])

# Pesos para os vértices
plot(grafo5, bbox=(300,300), vertex_size = grafo5.vs['peso'],
     edge_width = grafo5.es['weight'],
     vertex_color = grafo5.vs['cor'])

# Curvatura
plot(grafo5, bbox=(300,300), vertex_size = grafo5.vs['peso'],
     edge_width = grafo5.es['weight'],
     vertex_color = grafo5.vs['cor'],
     edge_curved = 0.4)

# Formato
plot(grafo5, bbox=(0,0,300,300), vertex_size = grafo5.vs['peso'],
     edge_width = grafo5.es['weight'],
     vertex_color = grafo5.vs['cor'],
     edge_curved = 0.4, vertex_shape = 'triangle')


"""
metricas
"""
from igraph import Graph
from igraph import plot
import igraph

# Carregamento de grafo no formato graphml
grafo = igraph.load(path + 'Grafo.graphml')
print(grafo)

plot(grafo, bbox = (0,0,600,600))

# Visualização do grau de entrada, saída e entrada + saída do grafo
grafo.degree(mode = 'all')
#grafo.degree(mode = 'in')
#grafo.degree(mode = 'out')

# Obtendo e imprimindo somente os graus de entrada
grau = grafo.degree(mode = 'in')
print(grau)

#gerando o grafo com vertice proporcional ao grau
plot(grafo, vertex_size = grau ,bbox = (0,0,600,600))

# Retorno do diâmetro do grafo (maior distância entre os vértices)
grafo.diameter(directed = True)

# Retorno dos vértices que possuem a maior distância entre os pontos do grafo
grafo.get_diameter()

# Retorno dos vizinhos de cada vértice
grafo.neighborhood()

# Verificar se o grafo é isomórfico
grafo2 = grafo
grafo.isomorphic(grafo2)


"""
caminhos distancias
"""
from igraph import Graph
from igraph import plot

# Criação de grafo direcionado com pesos entre as arestas
grafo = Graph(edges = [(0,2),(0,1),(1,4),(1,5),(2,3),(6,7),(3,7),(4,7),(5,6)],
                       directed = True)
grafo.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
grafo.es['weight'] = [2,1,2,1,2,1,3,1]

# Visualização do grafo com os pesos
plot(grafo, bbox = (0,0,300,300), edge_label = grafo.es['weight'])

# Caminho entre os vértices A - H
caminho_vertice = grafo.get_shortest_paths(0,7, output = 'vpath')
for n in caminho_vertice[0]:
    print(grafo.vs[n]['label'])

# Obtenção dos caminhos mais curtos entre 0 e 7
caminho_aresta = grafo.get_shortest_paths(0,7, output = 'epath')
caminho_aresta

# Mostrar o caminho 
caminho_aresta_id = []
for n in caminho_aresta[0]:
    caminho_aresta_id.append(n)
caminho_aresta_id

# Somatório dos pesos (ou distâncias) entre os vértices do caminho
distancia = 0
for e in grafo.es:
    #print(e.index)
    if e.index in caminho_aresta_id:
        distancia += grafo.es[e.index]['weight']
distancia 

"""
Caminhos distancias impressão
"""
from igraph import Graph
from igraph import plot

# Criação de um gráfico direcionado com pesos entre as arestas
grafo = Graph(edges = [(0,2),(0,1),(1,4),(1,5),(2,3),(6,7),(3,7),(4,7),(5,6)],
                       directed = True)
grafo.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
grafo.es['weight'] = [2,1,2,1,2,1,3,1]

# Visualização dos resultados
plot(grafo, bbox = (0,0,300,300), edge_label = grafo.es['weight'])

# Menor caminho entre A - H (retorna os vértices)
caminho_vertice = grafo.get_shortest_paths(0,7, output = 'vpath')
caminho_vertice

# Retorna as arestas que ligam os dois vértices
caminho_aresta = grafo.get_shortest_paths(0,7, output = 'epath')
caminho_aresta

# Mostra o ID dos vértices que fazem parte do caminho
caminho_aresta_id = []
for n in caminho_aresta[0]:
    caminho_aresta_id.append(n)
caminho_aresta_id  

# Mostra o nome dos vértices que fazem parte do caminho
caminho_nome_vertices = []
for n in caminho_vertice[0]:
    print(grafo.vs[n]['label'])
    caminho_nome_vertices.append(grafo.vs[n]['label'])
caminho_nome_vertices

# Colorir os vértices do caminho
for v in grafo.vs:
    #print(v)
    if v['label'] in caminho_nome_vertices:
        v['color'] = 'green'
    else:
        v['color'] = 'gray'

# Colorir as arestas do caminho
for e in grafo.es:
    #print(e)
    if e.index in caminho_aresta_id:
        e['color'] = 'green'
    else:
        e['color'] = 'gray'

plot(grafo, bbox=(0,0,300,300))


"""
Comunidades
ex1
"""
from igraph import Graph
from igraph import plot
import igraph
import numpy as np

# Carregamento de grafo no formato graphml
grafo = igraph.load(path + 'Grafo.graphml')
print(grafo)

# Visualização do grafo
plot(grafo, bbox = (0,0,600,600))

# Visualização das comunidades
comunidades = grafo.clusters()
print(comunidades)

# Visualização em qual comunidade qual registro foi associado
comunidades.membership

# Visualização do grafo
cores = comunidades.membership
# Array de cores para defirmos cores diferentes para cada grupo
cores = np.array(cores)
cores = cores * 20
cores = cores.tolist()
plot(grafo, vertex_color = cores)


"""
ex2
"""
# Criação de grafo direcionado com pesos nas arestas
grafo2 = Graph(edges = [(0,2),(0,1),(1,4),(1,5),(2,3),(6,7),(3,7),(4,7),(5,6)],
                       directed = True)
grafo2.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
grafo2.es['weight'] = [2,1,2,1,2,1,3,1]

# Visualização do grafo
plot(grafo2, bbox = (0,0,300,300))

# Visualização de comunidades e em qual comunidade cada registro foi associado
comunidades2 = grafo2.clusters()
print(comunidades2)
comunidades2.membership

# Função mais otimizada para visualização das comunidades
c = grafo2.community_edge_betweenness()
print(c)
# Obtenção do número de clusters
c.optimal_count
# Visualização da nova comunidade
comunidades3 = c.as_clustering()
print(comunidades3)
comunidades3.membership

# Geração do grafo das comunidades colocando cores entre os grupos identificados
plot(grafo2, vertex_color = comunidades3.membership)
cores = comunidades3.membership
# Array de cores para defirmos cores diferentes para cada grupo
cores = np.array(cores)
cores = cores * 100
cores = cores.tolist()

plot(grafo2, bbox = (0,0,300,300), vertex_color = cores)

# Visualização dos cliques
cli = grafo.as_undirected().cliques(min = 4)
print(cli)
len(cli)

