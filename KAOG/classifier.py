from KAOG import K_associated_graph
from KAOG import K_associated_optimal_graph
import pandas as pd
from ucimlrepo import fetch_ucirepo
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

def find_k_connection(x, data, k):
    distance_list = []
    for index in range(len(data)):
        line = data[index]
        _x = line[:-1]
        distance = K_associated_graph.compute_Euclidean_Distance(x, _x)
        distance_list.append((distance, index))
    distance_list = sorted(distance_list, key=lambda x: x[0])
    return distance_list[:k]

def find_connected_component(k_connection, KAOG):
    connected_components = []
    for connection in k_connection:
        for component in KAOG:
            if connection[1] in list(component[0].nodes()):

                connected_components.append(component)

    return connected_components

def find_component_label(component, data):
    nodes = list(component.nodes())
    index = nodes[0]
    return data[index][-1]

def compute_connected_num(component, component_list):
    sum = 0
    for item in component_list:
        if item == component:
            sum += 1
    return sum

def classifier(x, KAOG, data):
    max_k = max([y[2] for y in KAOG])
    k_connection = find_k_connection(x, data, max_k)
    connected_components = find_connected_component(k_connection, KAOG)
    connected_components_set = list(set(connected_components))
    # print(connected_components_set)
    if len(connected_components_set) == 1:
        return find_component_label(connected_components[0][0], data)
    P_list = []
    P_condition_list = []
    P_set = 0
    purity_sum = sum([item[1] for item in connected_components_set])
    if purity_sum == 0:
        return find_component_label(connected_components_set[0][0], data)
    for p_index in range(len(connected_components_set)):
        P = connected_components_set[p_index][1] / purity_sum
        P_condition = compute_connected_num(connected_components_set[p_index], connected_components) / connected_components_set[p_index][2]
        P_set += P_condition * P
        P_list.append(P)
        P_condition_list.append(P_condition)

    p_list = []

    for index in range(len(connected_components_set)):
        p = (P_condition_list[index] * P_list[index]) / P_set

        p_list.append(p)

    max_P = max(p_list)
    component_index = p_list.index(max_P)
    return find_component_label(connected_components_set[component_index][0], data)

dataset = ["yeast", "teaching", "zoo", "image", "wine", "iris", "glass", "ecoli", "balance", "vowel", "libras", "hayes-roth", "segment", "vehicle", "wineq"]
dataset_id = 5
data = load_data(dataset[dataset_id])
train_data, test_data = train_test_split(data, test_size=0.3)

accuracy = 0
KAOG = K_associated_optimal_graph.construct_k_associated_optimal_graph(train_data)
for index in range(len(test_data)):
    # print(index)
    pred_label = classifier(test_data[index][:-1], KAOG, train_data)
    if pred_label == test_data[index][-1]:
        accuracy += 1

max_k = max([y[2] for y in KAOG])
print("K is {}".format(max_k))
print("DataSet : {}".format(dataset[dataset_id]))
print("Accuracy : {}".format(accuracy / len(test_data)))

