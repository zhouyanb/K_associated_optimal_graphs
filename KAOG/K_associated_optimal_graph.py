from KAOG import K_associated_graph
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

def draw_K_associated_optimal_graph(K_associated_optimal_graph, data):
    V = []
    E = []
    pos = {}
    color_list = ['black', 'red', 'blue']
    node_color = []
    for component in K_associated_optimal_graph:
        V.extend(component[0].nodes())
        E.extend(component[0].edges())
    V = list(set(V))
    for edge in E:
        if edge[0] == edge[1]:
            print(edge)
    for index in V:
        pos[index] = (data[index][0], data[index][1])
        node_color.append(color_list[data[index][-1]])
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=node_color, node_size=50)
    plt.show()


def draw_graph(KAOG, data):
    nodes = []
    edges = []
    node_color = []
    color_list = ['black', 'red', 'blue']
    for item in KAOG:
        nodes.extend(list(item[0].nodes()))
        edges.extend(list(item[0].edges()))
    nodes = list(set(nodes))
    for index in nodes:
        # print(color_list[int(data[index][-1])])
        node_color.append(color_list[int(data[index][-1])])
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for edge in edges:
        start = edge[0]
        end = edge[1]
        distance = K_associated_graph.compute_Euclidean_Distance(data[start][:-1], data[end][:-1])
        G.add_edge(start, end,  weight=distance)
    # G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    # nx.draw_networkx_edges(G, pos, width=[float(d['weight']*2) for (u, v, d) in G.edges(data=True)])
    # nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=50)
    nx.draw(G, pos, node_color=node_color, node_size=50)
    # distance = nx.get_edge_attributes(G, 'length')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=distance)
    plt.show()


def compute_avgDegree(K_associated_graph):
    N = 0
    D = 0
    for item in K_associated_graph:
        component = item[0]
        for node in component.nodes():
            for edge in component.edges():
                if node in edge:
                    D += 1
        N += len(component.nodes())
        # D += len(component.edges())
    avgDegree = D / N
    return avgDegree

# def compute_avgPurity(K_associated_graph):
#     purity = 0
#     for item in K_associated_graph:
#         purity += item[1]
#     return purity / len(K_associated_graph)

def is_contain(component, sub_component):
    nodes = component.nodes()
    sub_nodes = sub_component.nodes()
    edges = component.edges()
    sub_edges = sub_component.edges()
    for node in sub_nodes:
        if node not in nodes:
            return 0
    for edge in sub_edges:
        if edge not in edges:
            return 0
    return 1

def construct_k_associated_optimal_graph(data):
    k = 1
    K_associated_optimal_graph = []
    ka_Graph = K_associated_graph.construct_k_associated_graph(data, k)
    for item in ka_Graph:
        K_associated_optimal_graph.append((item[0], item[1], k))
    lastAvgDegree = compute_avgDegree(ka_Graph)
    while True:
        k += 1
        ka_Graph = K_associated_graph.construct_k_associated_graph(data, k)
        for component in ka_Graph:
            flag = 1
            remove_list = []
            for index in range(len(K_associated_optimal_graph)):
                opt_component = K_associated_optimal_graph[index]
                # if nx.isomorphism.DiGraphMatcher(component[0], opt_component[0]).subgraph_is_isomorphic():
                if is_contain(component[0], opt_component[0]):
                    if component[1] >= opt_component[1]:
                        remove_list.append(index)
                    else:
                        flag = 0
            if flag == 1:
                remove_list.reverse()
                for remove_index in remove_list:
                    K_associated_optimal_graph.pop(remove_index)
                K_associated_optimal_graph.append((component[0], component[1], k))

        AvgDegree = compute_avgDegree(ka_Graph)
        if (AvgDegree - lastAvgDegree) < (AvgDegree / k):
            break
        else:
            lastAvgDegree = AvgDegree

    return K_associated_optimal_graph