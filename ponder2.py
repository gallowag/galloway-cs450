from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Step 1
def load_data():
    iris = datasets.load_iris()
    return iris

# Step 2
def prepare_training_and_test_sets(iris):
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.30)
    
    return data_train, data_test, targets_train, targets_test
    
# Class defintion - K_Nearest_Neighbors_Model
class KNearestNeighborsModel:
    def __init__(self, train_data, train_target, k):
        self.train_data = train_data
        self.train_target = train_target
        self.k = k
    
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
            predicted_target.append(get_mode(distances[:self.k:]))
            
        return predicted_target

# Class defintion - K_Nearest_Neighbor_Classifier
class KNearestNeighborClassifier:
    def __init__(self, k):
        self.k = k
    
    def fit(self, train_data, train_target):
        knn_model = KNearestNeighborsModel(train_data, train_target, self.k)
        return knn_model
    
# Determines the mode of given 2d list (first column is the distances
# and the second is the corresponding data target)
def get_mode(nearest_neighbors):
    frequencies = [0] * 3
    for neighbor in nearest_neighbors:
        frequencies[neighbor[1]] += 1
        
    return frequencies.index(max(frequencies))
    
# Uses existing kNN algorithm to predict targets
def existing_algorithm(train_data, train_target, test_data, k):
    classifier = KNeighborsClassifier(k)
    model = classifier.fit(train_data, train_target)
    return model.predict(test_data)
    
# My implementation of kNN
def my_algorithm(train_data, test_data, train_target, k):
    classifier = KNearestNeighborClassifier(k)
    model = classifier.fit(train_data, train_target)
    return model.predict(test_data)
    
# Calculate accuracy
def calculate_accuracy(list1, list2):
    count = 0
    length = len(list1)
    for i, j in zip(list1, list2):
        if(i == j):
            count += 1
    percent_accuracy = (count / length) * 100
    print("Achieved " + str(count) + "\\" + str(length) + " or " + str(percent_accuracy) + "% accuracy")
    
# Compares my algorithm and exissting algorithm for a given k
def compare_algorithms(train_data, test_data, train_target, test_target, k):
    # Get predicted targets from each algorthm
    predicted_target = existing_algorithm(train_data, train_target, test_data, k)
    my_predicted_target = my_algorithm(train_data, test_data, train_target, k)
    
    # Display results
    print("\nFor k is", k)
    print("Existing Algorithm:")
    calculate_accuracy(predicted_target, test_target)
    print("My Algorithm:")
    calculate_accuracy(my_predicted_target, test_target)
    
# main
def main():
    # Call steps 1 and 2
    iris = load_data()
    training_and_test_sets = prepare_training_and_test_sets(iris)
    
    # Seperate list
    train_data = training_and_test_sets[0]
    test_data = training_and_test_sets[1]
    train_target = training_and_test_sets[2]
    test_target = training_and_test_sets[3]
    
    # Comparing algorithms with different k values
    compare_algorithms(train_data, test_data, train_target, test_target, k=5)
    compare_algorithms(train_data, test_data, train_target, test_target, k=17)
    compare_algorithms(train_data, test_data, train_target, test_target, k=29)
    compare_algorithms(train_data, test_data, train_target, test_target, k=83)
    
if __name__ == "__main__":
    main()
