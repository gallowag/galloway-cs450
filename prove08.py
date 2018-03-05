from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats
import pandas as express
import numpy as np
import plotly as py
import plotly.graph_objs as go
import random
from math import e
from sklearn.metrics import accuracy_score

# Loads the Iris dataset
def load_iris():
    iris = datasets.load_iris()
    normalized = stats.zscore(iris.data)
    return train_test_split(normalized, iris.target, test_size=0.30)

# Loads the Pima Indian Diabetes dataset
def load_pima_indian_diabetes():
    # Read data
    df = express.io.parsers.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data", header=None)
    
    # Remove rows with missing data
    df[[1,2,3,4,5]] = df[[1,2,3,4,5]].replace(0, np.NaN)
    df.dropna(inplace=True)
    
    # Set up data and targets
    people = df.values
    
    target = people[:,8].astype(int)
    data = people[:,[0,1,2,3,4,5,6,7]].astype(float)
    
    # Normalize
    normalized = stats.zscore(data)
    
    return train_test_split(normalized, target, test_size=0.30)

# Calculate sigmoid function
def calculate_sigmoid(h):
    return 1 / (1 + e**(-h))

# Node class definition
class Node:
    def __init__(self, num_inputs):
        weights = []
        for i in range(0, num_inputs + 1):
            weights.append(random.uniform(0, 2) - 1)

        self.bias = -1
        self.error = -1000
        self.weights = weights
        self.l_r = .1
        
    def get_error(self):
        return self.error
        
    def get_weight(self, index):
        return self.weights[index]
    
    # Updates all weights of current node
    def update_weights(self):
        for i,weight in enumerate(self.weights):
            new_weight = weight - (self.l_r * self.error * self.a)
            self.weights[i] = new_weight
    
    # Calculates the "a" value of current node
    def calculate_a(self, inputs):
        h = self.bias * self.get_weight(0)
        
        # Calculate h for each input
        for i,input in enumerate(inputs):
            h += input * self.get_weight(i + 1)
            
        self.h = h
        self.a = calculate_sigmoid(h)
        return self.a
    
    # Calculates delta/error of a last layer node
    def calculate_error_last(self, target, index_of_node):     
        # Determine the target value
        t = 0
        if(target == index_of_node):
            t = 1
            
        # Set current nodes error with calculated error
        self.error = self.a * (1 - self.a) * (self.a - t)
        
    # Calculates delta/error of a hidden layer node
    def calculate_error(self, k_layer, index_of_node):
        # Calculate summation
        total = 0
        for node in k_layer.get_nodes():
            total += (node.get_error() * node.get_weight(index_of_node))
        
        # Set current nodes error with calculated error 
        self.error = self.a * (1 - self.a) * total

# Layer of nodes class definition
class LayerOfNodes:
    def __init__(self, num_inputs, num_nodes):
        nodes = []
        for i in range(0, num_nodes):
            nodes.append(Node(num_inputs))
        
        self.nodes = nodes
        self.outputs = []
    
    def get_nodes(self):
        return self.nodes
    
    def get_outputs(self):
        return self.outputs
    
    # Goes through each node and updates it's weights
    def update_weights(self):
        for node in self.nodes:
            node.update_weights()
    
    # Goes through each node and calculates it's "a" value
    def calculate_outputs(self, inputs):
        outputs = []
        for node in self.nodes:
            a = node.calculate_a(inputs)
            outputs.append(a)
        self.outputs = outputs
        return self.outputs
    
    # Goes through each last layer node and calculates their errors
    def calculate_errors_last(self, target):
        for i,node in enumerate(self.nodes):
            node.calculate_error_last(target, i)
            
    # Goes through each hidden layer node and calculates their errors
    def calculate_errors(self, k_layer):
        for i,node in enumerate(self.nodes):
            node.calculate_error(k_layer, i + 1)
        
# Neural Net Classifier class definition
class NNClassifier():
    
    def __init__(self, num_inputs, num_nodes_per_layer):
        # Create layers of nodes
        layers = []
        
        first_layer = LayerOfNodes(num_inputs, num_nodes_per_layer[0]) 
        layers.append(first_layer)
        
        for i in range(len(num_nodes_per_layer) - 1):
            layer = LayerOfNodes(num_nodes_per_layer[i], num_nodes_per_layer[i + 1]) 
            layers.append(layer)
            
        self.layers = layers
    
    
    def fit(self, train_data, train_target, test_data, test_target, num_loops):  
        accuracy_list = []
        
        for round in range(num_loops):
            # For each row of attribute data and corresponding target
            for data_row,target in zip(train_data,train_target):
                # Go forward
                next_inputs = data_row

                # Calculate the outputs
                for layer in self.layers:
                    next_inputs = layer.calculate_outputs(next_inputs)

                fires = np.argmax(next_inputs)
                
                if(fires != target):
                    # Go backwards
                    is_last_layer = True
                    # Calculate errors
                    for layer in reversed(self.layers):
                        if(is_last_layer):  
                            layer.calculate_errors_last(target)
                        else:
                            layer.calculate_errors(k_layer)
                        is_last_layer = False
                        k_layer = layer
                        
                    # Update weights
                    for layer in self.layers:
                        layer.update_weights()
    
            
            model = NNModel(train_data, train_target, self.layers)
            predicted = model.predict(test_data)
    
            accuracy_list.append(accuracy_score(predicted, test_target))
        return accuracy_list

# Neural Net Model class definition
class NNModel():
    def __init__(self, train_data, train_target, layers):
        self.train_data = train_data
        self.train_target = train_target
        self.layers = layers

    def predict(self, test_data):
        predicted = []
        
        # For each row of attribute data
        for data_row in test_data:
            next_inputs = data_row
            
            # For layer of nodes
            for layer in self.layers:
                next_inputs = layer.calculate_outputs(next_inputs)
            
            predicted.append(np.argmax(next_inputs))
        return predicted
        
def main():
    idata_train, idata_test, itargets_train, itargets_test = load_iris()
       
    # Use my classifier on Iris
    num_loops = 100
    classifier = NNClassifier(4,[10,3])
    list1 = classifier.fit(idata_train, itargets_train, idata_test, itargets_test, num_loops)
    
    # Create trace 1
    trace1 = go.Scatter(
        x = np.arange(num_loops),
        y = np.asarray(list1),
        mode = 'lines',
        name = 'lines'
    )
    
    data_train, data_test, targets_train, targets_test = load_pima_indian_diabetes()
    
    # Use my classifier on Pima Indian Diabetes
    num_loops = 50
    classifier = NNClassifier(8, [7,2])
    list2 = classifier.fit(data_train, targets_train, data_test, targets_test, 50)
    
    # Create trace 2
    trace1 = go.Scatter(
        x = np.arange(num_loops),
        y = np.asarray(list2),
        mode = 'lines',
        name = 'lines'
    )
    
    data = [trace1, trace2]

    py.iplot(data, filename='line-mode')
    
if __name__ == "__main__":
    main()
