import matplotlib.pyplot as plt
import math
import networkx as nx
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

Yeast_class = {'MIT':0, 'NUC':1, 'CYT':2, 'ME1':3, 'EXC':4, 'ME2':5, 'ME3':6, 'VAC':7, 'POX':8, 'ERL':9}
Iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
def load_data(id):
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    y = dataset.data.targets
    data = X.values.tolist()
    for index in range(len(data)):
        if type(y.values.tolist()[index][0]) == str:
            if id == 53:
                data[index].append(Iris_class[y.values.tolist()[index][0]])
            if id == 110:
                data[index].append(Yeast_class[y.values.tolist()[index][0]])
        else:
            data[index].append(y.values.tolist()[index][0])

    return data



def compute_Euclidean_Distance(x1, x2):
    # return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(x1, x2)]))
    distance = 0
    for index in range(len(x1)):
        distance += (x1[index] - x2[index]) * (x1[index] - x2[index])
    distance = math.sqrt(distance)
    return distance

def find_k_neighborhood(index, data, k):
    distance_list = []
    x = data[index][:-1]
    for i in range(len(data)):
        if i == index:
            continue
        line = data[i]
        _x = line[:-1]
        distance = compute_Euclidean_Distance(x, _x)
        distance_list.append((distance, i))
    distance_list = sorted(distance_list, key=lambda x:x[0])

    return distance_list[:k]

def find_k_neighborhood_withlabel(k_set, data, y):
    k_set_withlabel = []
    for item in k_set:
        if data[item[1]][-1] == y:
            k_set_withlabel.append(item)

    return k_set_withlabel

def find_component_edges(v, E):
    edges = []
    for edge in E:
        if edge[0] in v and edge[1] in v:
            edges.append(edge)
    return edges

def compute_purity(G, k):
    d = 0
    for node in G.nodes():
        for edge in G.edges():
            if node in edge:
                d += 1
    D = d / len(G.nodes())
    purity = D / (2 * k)
    return purity


def construct_k_associated_graph(data, k):
    # C = {}
    K_associated_graph = []
    V = []  #点集
    E = []  #边集
    G = nx.DiGraph()
    for index in range(len(data)):
        line = data[index]
        V.append(index)
        x = line[:-1]
        y = line[-1]
        k_set = find_k_neighborhood(index, data, k)  # 标签无关的k近邻点集合，其中元素表示为(distance, index)
        # print(k_set)
        k_set_withlabel = find_k_neighborhood_withlabel(k_set, data, y)
        # print(k_set_withlabel)
        for neighborhood in k_set_withlabel:
            E.append((index, neighborhood[1]))

    G.add_nodes_from(V)
    G.add_edges_from(E)
    C = nx.connected_components(G.to_undirected())
    for item in C:
        component_V = list(item)
        component_E = find_component_edges(list(item), E)
        component_Graph = nx.DiGraph()
        component_Graph.add_nodes_from(component_V)
        component_Graph.add_edges_from(component_E)
        purity = compute_purity(component_Graph, k)
        K_associated_graph.append((component_Graph, purity))

    return K_associated_graph


