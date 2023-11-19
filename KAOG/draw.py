from KAOG import K_associated_graph
from KAOG import K_associated_optimal_graph
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split


Segment_class = {'BRICKFACE':0, 'SKY':1, 'FOLIAGE':2, 'CEMENT':3, 'WINDOW':4, 'PATH':5, 'GRASS':6}
def load_data_uci(id):
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    y = dataset.data.targets
    data = X.values.tolist()
    for index in range(len(data)):
        if type(y.values.tolist()[index][0]) == str:
            if id == 50:
                data[index].append(Segment_class[y.values.tolist()[index][0]])
        else:
            data[index].append(y.values.tolist()[index][0])

    return data

Yeast_class = {'MIT':0, 'NUC':1, 'CYT':2, 'ME1':3, 'EXC':4, 'ME2':5, 'ME3':6, 'VAC':7, 'POX':8, 'ERL':9}
Iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
Ecoli_class = {'cp':0, 'im':1, 'pp':2, 'imU':3, 'om':4, 'omL':5, 'imL':6, 'imS':7}
Balance_class = {'L':0, "B":1, "R":2}
def load_data(id):
    if id == "segment":
        data = load_data_uci(50)
        return data
    data_path = "Data/" + id + ".data"
    df = pd.read_csv(data_path)
    dataset = df.values.tolist()
    data = []
    for index in range(len(dataset)):
        if id == 'yeast':
            line = dataset[index][0].split()
            line = line[1:]
            line[-1] = Yeast_class[line[-1]]
            line = [float(x) for x in line]
            data.append(line)
        if id == 'zoo':
            line = dataset[index][1:]
            data.append(line)
        if id == 'wine':
            line = dataset[index][1:]
            line.append(dataset[index][0])
            data.append(line)
        if id == 'iris':
            line = dataset[index]
            line[-1] = Iris_class[dataset[index][-1]]
            data.append(line)
        if id == 'glass':
            line = dataset[index][1:]
            data.append(line)
        if id == 'ecoli':
            line = dataset[index][0].split()
            line = line[1:]
            line[-1] = Ecoli_class[line[-1]]
            line = [float(x) for x in line]
            data.append(line)
        if id == 'balance':
            line = dataset[index][1:]
            line.append(Balance_class[dataset[index][0]])
            data.append(line)
        if id == 'libras' or id == 'hayes-roth':
            line = dataset[index]
            data.append(line)

    return data


def draw(data):
    color_list = ['black', 'red', 'blue']
    for line in data:
        plt.scatter(line[0], line[1], color=color_list[line[-1]])
    plt.show()

def draw_iris_graph(KAOG, data):
    nodes = []
    edges = []
    node_color = []
    node_shape = []
    color_list = ['black', 'red', 'blue']
    shape_list = ['o', '^', 'v']
    for item in KAOG:
        nodes.extend(list(item[0].nodes()))
        edges.extend(list(item[0].edges()))
    nodes = list(set(nodes))
    nodes_class = {0: [], 1: [], 2: []}
    pos = {}
    G = nx.Graph()
    for index in nodes:
        pos[index] = (data[index][0], data[index][1])
        nodes_class[int(data[index][-1])].append(index)
        # node_shape.append(shape_list[int(data[index][-1])])


    # G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_class[0], node_shape=shape_list[0], node_color=color_list[0], node_size=25)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_class[1], node_shape=shape_list[1], node_color=color_list[1], node_size=25)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_class[2], node_shape=shape_list[2], node_color=color_list[2], node_size=25)
    nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_edges(G, pos)
    # nx.draw(G, pos, node_color=node_color,  node_size=25, node_shape="^")

    plt.show()


dataset = ["yeast", "teaching", "zoo", "image", "wine", "iris", "glass", "ecoli", "balance", "vowel", "libras", "hayes-roth", "segment", "vehicle", "wineq"]
data = load_data(dataset[5])
del_data = []
dataset = []
_dataset = []
for line in data:
    _dataset.append([line[0], line[3], line[4]])

for index in range(len(_dataset)):
    line = _dataset[index]
    if line in dataset:
        del_data.append(index)
        continue
    else:
        dataset.append(line)
del_data.reverse()
for del_index in del_data:
    data.pop(del_index)

print(len(data))
# print(data)
# draw(dataset)
components = []
KAG = K_associated_graph.construct_k_associated_graph(dataset, 1)
KAG2 = K_associated_graph.construct_k_associated_graph(dataset, 2)
KAG3 = K_associated_graph.construct_k_associated_graph(dataset, 3)
KAG4 = K_associated_graph.construct_k_associated_graph(dataset, 4)
KAOG = K_associated_optimal_graph.construct_k_associated_optimal_graph(dataset)
# index = 4
# components.append(KAG3[index])
print(KAG)
print(KAG2)
print(KAG3)
print(KAG4)
max_k = max([y[2] for y in KAOG])
print("K is {}".format(max_k))
# draw_iris_graph(KAG, dataset)
# draw_iris_graph(KAG2, dataset)
# draw_iris_graph(KAG3, dataset)
# draw_iris_graph(KAG4, dataset)
draw_iris_graph(KAOG, dataset)
# print(KAG3[index][1])
# draw_iris_graph(components, dataset)
# print(KAG3[index][0].nodes())
# print(KAG3[index][0].edges())