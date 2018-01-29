import pandas as express
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load UCI: Car Evaluation dataset
def load_uci_car_evaluation():
    # Read data
    df = express.io.parsers.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=None)
    
    # Set headers
    df.columns = ["buying", "maint", "doors", "persons", "lug_boat",
           "safety", "acceptability"]
    
    # Replace categorical data with numerical data
    obj_df = df.select_dtypes(include=['object']).copy()
    
    cleanup_cols = {"buying":        {"vhigh": 3, "high": 2, "med": 1, "low": 0},
                    "maint":         {"vhigh": 3, "high": 2, "med": 1, "low": 0},
                    "doors":         {"2": 2, "3": 3, "4": 4, "5more": 5},
                    "persons":       {"2": 2, "4": 4, "more": 5},
                    "lug_boat":      {"small": 0, "med": 1, "big": 2},
                    "safety":        {"low": 0, "med": 1, "high": 2},
                    "acceptability": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}
    
    obj_df.replace(cleanup_cols, inplace=True)
    
    # Set up data and targets
    cars = obj_df.values
    
    target = cars[:,6]
    data = cars[:,[0,1,2,3,4,5]].astype(float)
    
    # Scale data
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_data = scaler.transform(data)

    return scaled_data, target

# Load PIMA Indian Dietbetes dataset
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
    
    # Scale data
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    
    return scaled_data, target
    
# Load Automobile MPG dataset
def load_automobile_mpg():
    # Read data
    df = express.io.parsers.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", header=None, 
        delim_whitespace=True)
    
    # Remove rows with missing data ("?")
    cars_2 = np.delete(df.values, [32, 126, 330, 336, 354, 374], 0)
    
    # Set up data and targets
    target = cars_2[:,[0]].flatten()
    data = cars_2[:,[1,2,3,4,5,6,7]].astype(float)
    
    # Scale data
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    
    return scaled_data, target
    
# Class defintion - K_Nearest_Neighbors_Model
class KNNModel:
    def __init__(self, train_data, train_target, k, knn_type):
        self.train_data = train_data
        self.train_target = train_target
        self.k = k
        self.knn_type = knn_type
    
    def predict(self, test_data):
        predicted_target = []
        
        # For each row of test data,
        for i in test_data:
            distances = []
            for j, row in enumerate(self.train_data):
                
                # Calculate it's distance from each row of training data
                distance = sum([(a - b) ** 2 for a, b in zip(i, row)])
                distances.append([distance, self.train_target[j]])
                
            # Sort the distances and retrieve the mode of the 'k' smallest distances
            distances.sort()
            if(self.knn_type == "class"):
                predicted_target.append(get_mode(distances[:self.k:]))
            elif(self.knn_type == "value"):
                predicted_target.append(np.mean(distances[:self.k:]))
            
        return predicted_target

# Class defintion - K_Nearest_Neighbor_Classifier
class KNNClassifier:
    def __init__(self, k, knn_type):
        self.k = k
        self.knn_type = knn_type
    
    def fit(self, train_data, train_target):
        knn_model = KNNModel(train_data, train_target, self.k, self.knn_type)
        return knn_model
    
# Determines the mode of given 2d list (first column is the distances
# and the second is the corresponding data target)
def get_mode(nearest_neighbors):
    frequencies = [0] * 10
    for neighbor in nearest_neighbors:
        frequencies[neighbor[1]] += 1
        
    return frequencies.index(max(frequencies))
    
# Calculate accuracy for values
def calculate_accuracy(list1, list2):
    count = 0
    length = len(list1)
    for i, j in zip(list1, list2):
        
        # If value is within 3 of the other value, view as correct
        if((j - 3) <= i <= (j + 3)):
            count += 1
            
    percent_accuracy = (count / length) * 100
    return percent_accuracy
 
# Using cross validation to find predictions with other knn classifier
def implement_other_knn(data, target, k_fold, k_neighbors, knn_type):
    other_prediction_accuracies = []
    
    # Start cross validation
    kf = KFold(n_splits=k_fold, shuffle=True)

    for train_index, test_index in kf.split(data):
        # Determine training data, testing data, training targets, and testing targets
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
      
        # Other algorithm
        classifier = KNeighborsClassifier(k_neighbors)
        other_model = classifier.fit(data_train, target_train)
        prediction = other_model.predict(data_test)
        
        # Calculate accuracy and add it to list of accuracies
        percent_accuracy = accuracy_score(prediction, target_test)
        other_prediction_accuracies.append(percent_accuracy)
        
    # Return the average of the accuracies
    return np.mean(other_prediction_accuracies)

# Using cross validation to find predictions with my knn classifier
def implement_my_knn(data, target, k_fold, k_neighbors, knn_type):
    my_prediction_accuracy = []
    
    # Start cross validation
    kf = KFold(n_splits=k_fold, shuffle=True)

    for train_index, test_index in kf.split(data):
        # Determine training data, testing data, training targets, and testing targets
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        
        # My algorithm
        classifier = KNNClassifier(k_neighbors, knn_type)
        my_model = classifier.fit(data_train, target_train)
        prediction = my_model.predict(data_test)
        
        # If the class is being predicted, append accuracy using accuracy_score from lib
        if(knn_type == "class"):     
            my_prediction_accuracy.append(accuracy_score(prediction, target_test))
            
        # If the value is being predicted, append accuracy using calculate_accuracy
        elif(knn_type == "value"):
            my_prediction_accuracy.append(calculate_accuracy(prediction, target_test))
    
    # Return the average of the accuracies
    return np.mean(my_prediction_accuracy)
    
# main
def main():
    data_1, target_1 = load_uci_car_evaluation()
    data_2, target_2 = load_pima_indian_diabetes()
    data_3, target_3 = load_automobile_mpg()
    
    # Use data 1
    my_data_1_1 = implement_my_knn(data_1, target_1, 5, 5, "class")
    my_data_1_2 = implement_my_knn(data_1, target_1, 5, 10, "class")
    other_data_1 = implement_other_knn(data_1, target_1, 5, 5, "class")
    
    print("My knn with 5 fold and 5 neigbors:", my_data_1_1)
    print("Other knn with 5 fold and 5 neigbors:", other_data_1)
    print("My knn with 5 fold and 10 neigbors:", my_data_1_2)
     
    # Use data 2
    my_data_2_1 = implement_my_knn(data_2, target_2, 10, 5, "class")
    my_data_2_2 = implement_my_knn(data_2, target_2, 20, 5, "class")
    other_data_2 = implement_other_knn(data_2, target_2, 10, 5, "class")
    
    print("My knn with 10 fold and 5 neigbors:", my_data_2_1)
    print("Other knn with 20 fold and 5 neigbors:", other_data_2)
    print("My knn with 10 fold and 5 neigbors:", my_data_2_2)
    
    # Use data 3
    my_data_3_1 = implement_my_knn(data_3, target_3, 3, 3, "value")
    my_data_3_2 = implement_my_knn(data_3, target_3, 15, 7, "value")
    
    print("My knn with 3 fold and 3 neigbors:", my_data_3_1)
    print("My knn with 15 fold and 7 neigbors:", my_data_3_2)
    
if __name__ == "__main__":
    main()
