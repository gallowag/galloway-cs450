import pandas as express
from sklearn import datasets
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load iris dataset
def load_iris_data():
    iris = datasets.load_iris()
    return train_test_split(iris.data, iris.target, test_size=0.30)


# Load iris dataset - discretized
def load_discretized_iris_data():
    # Read data
    iris = datasets.load_iris()
    iris_data = []
    row = []
    low = 0
    med = 0
    high = 0
    
    # First, discretize all the data
    for i in range(len(iris.data)):
        row = []
        value = iris.data[i][0]
        if(4.3 <= value <= 5.4):
            row.append("small")
        elif(5.5 <= value <= 6.2):
            row.append("med")
        elif(6.3 <= value <= 7.9):
            row.append("big")
    
        value = iris.data[i][1]
        if(2.0 <= value <= 2.8):
            row.append("small")
        elif(2.9 <= value <= 3.1):
            row.append("med")
        elif(3.2 <= value <= 4.4):
            row.append("big")
    
        value = iris.data[i][2]
        if(1.0 <= value <= 1.9):
            row.append("small")
        elif(3.0 <= value <= 4.9):
            row.append("med")
        elif(5.0 <= value <= 6.9):
            row.append("big")

        value = iris.data[i][3]
        if(0 <= value <= 0.9):
            row.append("low")
        elif(1.0 <= value <= 1.6):
            row.append("med")
        elif(1.7 <= value <= 2.6):
            row.append("high")
        iris_data.append(row)  
    
    data_train, data_test, targets_train, targets_test = train_test_split(iris_data, iris.target, test_size=0.30)
    
    feature_names = ["sepal length", "sepal width", "petel length", "petel width"]
    
    
    return data_train, data_test, targets_train, targets_test, feature_names

# Class definition - DecisionTreeModel
class DecisionTreeModel:
    def __init__(self, tree):
        self.tree = tree
        
    def predict():
        pass

# Class defintion - DecisionTreeClassifier
class DecisionTreeClassifier:
    def __init__(self):
        pass
    
    # Fits the data
    def fit(self, train_data, train_target, featureNames):
        self.data = train_data
        self.targets = train_target
        tree = self.build_tree(train_data, train_target, featureNames)
        model = DecisionTreeModel(tree)
        

    # Recursive function that builds the Id3 tree
    def build_tree(self, data, classes, featureNames):
        nData = len(data)
        nFeatures = len(featureNames)
        # Determining all possible classes
        possible_classes = []
        for target in classes:
            if target not in possible_classes:
                possible_classes.append(target)
                
        # Determine number of occurences of each class
        frequency = np.zeros(len(possible_classes))    
        for p_index, p_class in enumerate(possible_classes):
            for c_class in classes:
                if(p_class == c_class):
                    frequency[p_index] += 1
                
        oneClassLeft = True
        for target in classes:
            if(target != classes[0]):
                oneClassLeft = False
        
        default = classes[np.argmax(frequency)]
        # If we're at an empty branch
        if(nData == 0 or nFeatures == 0):
            return default 
        
        # One class left
        elif(oneClassLeft):
            return classes[0]
        
        # Otherwise split and recurse
        else:
            gain = np.zeros(nFeatures)
 
            for feature in range(nFeatures):
                gain[feature] = self.calculate_info_gain(data, feature, classes)
                
            # Smallest value corresponds to the best feature
            bestFeature = np.argmin(gain)
            
            tree = {featureNames[bestFeature]:{}}
            
            newData = []
            newClasses = []
            values = []
            # Determine branches of best feature
            for row in data:
                if(row[bestFeature] not in values):
                    values.append(row[bestFeature])
            
            # For each branch
            for value in values:
                
                # Filter the data, classes, and feature names according to each branch
                for index, row in enumerate(data):
                    if(row[bestFeature] == value):
                        if(bestFeature == 0):
                            row = row[1:]
                            newNames = featureNames[1:]
                        elif(bestFeature == nFeatures):
                            row = row[:-1]
                            newNames = featureNames[:-1]
                        else:
                            row = row[:bestFeature]
                            row.extend(row[bestFeature + 1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature + 1:])
                        newData.append(row)
                        newClasses.append(classes[index])

                # Recurse with new data, classes and feature names
                subtree = self.build_tree(newData, newClasses, newNames)
                
                tree[featureNames[bestFeature]][value] = subtree
            
            return tree
            
    # Calculates the entropy given a fraction
    def calculate_entropy(self, fraction):
        if(fraction != 0):
            return -fraction * np.log2(fraction)
        else:
            return 0
    
    # Calculates the info gain of an attribute
    def calculate_info_gain(self, data_set, attribute, targets):
        
        # Determining all possible classes
        classes = []
        for target in targets:
            if target not in classes:
                classes.append(target)
                
        # Determing all possible values
        values = []
        for row in data_set:
            if row[attribute] not in values:
                values.append(row[attribute])
        
        entropies = []
        value_counts = np.zeros(len(values))
        
        # For each branch
        for val_index, value in enumerate(values):
            branch = []
            
            # Count all the occurences of the current value 
            # and collect corresponding targets/classes of the current value
            for row_index, row in enumerate(data_set):
                if(row[attribute] == value):
                    value_counts[val_index] += 1
                    branch.append(targets[row_index])
                    
            class_counts = np.zeros(len(classes))
           
            # Count all occurences of each class in branch
            for tar_index, target in enumerate(classes):
                for branch_class in branch:
                    if(target == branch_class):
                        class_counts[tar_index] += 1
                        
            entropy = 0
            weighted_entropies = []
            
            # Calculate entropy of branch
            for count in class_counts:
                entropy += self.calculate_entropy(count / value_counts[val_index])
            
            weighted_entropies.append(entropy * (value_counts[val_index] / len(targets)))
            
        return(sum(weighted_entropies))

def main():
    data_train, data_test, targets_train, targets_test = load_iris_data()
    d_data_train, d_data_test, d_targets_train, d_targets_test, feature_names = load_discretized_iris_data()
    
    # Other implementation
    other_classifier = tree.DecisionTreeClassifier()
    other_model = other_classifier.fit(data_train, targets_train)
    other_predictions = other_model.predict(data_test)
    print("Accuracy of other implementation:", accuracy_score(other_predictions, targets_test))
    
    # My implementation
    my_classifier = DecisionTreeClassifier()
    my_model = my_classifier.fit(d_data_train, d_targets_train, feature_names)

if __name__ == "__main__":
    main()
