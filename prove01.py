from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#step 1
def load_data():
    iris = datasets.load_iris()
    
    return iris

#step 2
def prepare_training_and_test_sets():
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.30)
    
    return data_train, data_test, targets_train, targets_test

#step 3
def use_existing_algorithm_to_create_a_model(data_train, targets_train):
    classifier = GaussianNB()
    model = classifier.fit(data_train, targets_train)
    
    return model

#calculate accuracy
def calculate_accuracy(list1, list2):
    count = 0
    length = len(list1)
    for i, j in zip(list1, list2):
        if(i == j):
            count += 1
    percent_accuracy = (count / length) * 100
    print("Achieved " + str(count) + "\\" + str(length) + " or " + str(percent_accuracy) + "% accuracy")

#step 4
def use_that_model_to_make_predictions(model, data_test, targets_test):
    targets_predicted = model.predict(data_test)

    print("Naive Bayes algorithm: ")
    calculate_accuracy(targets_predicted, targets_test)
    
    return targets_predicted

#class defintion - HardCodedModel
class HardCodedModel:
    def __init__(self):
        pass
    
    def predict(self, data_test):
        # Initialize empty targets
        targets = []
        
        # Fill targets with 0's
        for i in data_test:
            targets.append(0)
            
        return targets

#class defintion - HardCodedClassifier
class HardCodedClassifier:
    def __init__(self):
        pass
    
    def fit(self, data_train, targets_train):
        hard_coded_model = HardCodedModel()
        return hard_coded_model
    
#step 5
def implement_your_own_new_algorithm(data_train, data_test, targets_train, targets_test):
    classifier = HardCodedClassifier()
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    
    print("Hard-coded algorithm: ")
    calculate_accuracy(targets_predicted, targets_test)
    
    return targets_predicted
    
#main
def main():
    # Call steps 1 and 2
    iris = load_data
    training_and_test_sets = prepare_training_and_test_sets()
    
    # Seperate list
    data_train = training_and_test_sets[0]
    data_test = training_and_test_sets[1]
    targets_train = training_and_test_sets[2]
    targets_test = training_and_test_sets[3]
    
    # Call steps 3-5
    model = use_existing_algorithm_to_create_a_model(data_train, targets_train)
    targets_predicted = use_that_model_to_make_predictions(model, data_test, targets_test)
    my_targets_predicted = implement_your_own_new_algorithm(data_train, data_test, targets_train, targets_test)
    
#call main    
if __name__ == "__main__":
    main()
