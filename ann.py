import csv

#from sklearn.linear_model import Perceptron

#from sklearn.model_selection import train_test_split
#import numpy as np
#import pandas as pd


'''
def predict(row, weights):
    activation = weights[0]
    for i in range(1,len(row)-1):

        activation += weights[i + 1] * float(row[i])
        print("")
        if activation >= 0.0:
            return 1.0 
        
        else: 
            return 0.0
'''
def train(weights, inputs, target, learning_rate):
    prediction = predict(weights, inputs)
    error = target - prediction
    for i in range(1, len(weights) - 1):
        #print("\nWeight:", weights[i], "\ninputs:", inputs[1])
        #print("Train input:",inputs[i])
        weights[i] += learning_rate * error * float(inputs[i])
    weights[0] += learning_rate * error
    return weights

def predict(weights, inputs):
        weighted_sum = 0
        for input, weight in zip(inputs[1:], weights[1:]):
            print("Predict input:",input, "weight:", weight)
            weighted_sum += float(input) * weight
            print("weighted sum:", weighted_sum)
        weighted_sum += weights[0]
        print("Weighted sum:", weighted_sum)
        if weighted_sum >= 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    training_data = []  # training samples
    training_target = []  # target
    with open("survey lung cancer.csv", "r") as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:  # Training set
            #print(row)
            #print(row[0], row[1], row[2])
            if i == 210:
                break
            
            if row[1].isdigit() is False:
                i+=1
                continue

            else:
                if row[-1] == "YES":
                    row[-1] = 1   # Target value for lung cancer
                    training_target.append(row[-1])

                else:
                    row[-1] = 0   # Target value for lung cancer.
                    training_target.append(row[-1])

                training_data.append(row)
                if i == 209:
                    break
                i+= 1

        print("training_data top row: ",training_data[0])
        print("Number of elements in Y:", len(training_target))
        test_values = []
        test_target = []

        for row in reader:  # Test set
                #print(row)
                #print(row[0], row[1], row[2])
                
            if row[1].isdigit() is False:
                #i+=1
                continue

            else:
                if row[-1] == "YES":
                    row[-1] = 1   # Target value for lung cancer
                    test_target.append(row[-1])

                else:
                    row[-1] = 0   # Target value for lung cancer.
                    test_target.append(row[-1])

                test_values.append(row)
                #i+= 1

        weights = [0.1]
        for i in range(15):
            weights.append(0.1)

        print("Weights:", weights)
        
        learning_rate = 0.1
        
        for i in range(10):
            print("\nFor loop counter for training:", i)
            for inputs, target in zip(training_data, training_target):
                #print("inputs:", inputs, "\ntarget:", target)
                weights = train(weights, inputs, target, learning_rate)
        print("Updated weights:", weights)
            
    accuracy = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for input, target in zip(test_values, test_target):
        prediction = predict(weights, input)

        if prediction == 1 and target == 1:
            true_positive += 1

        elif prediction == 1 and target == 0:
            false_positive += 1

        elif prediction == 0 and target == 1:
            false_negative += 1

        elif prediction == 0 and target == 0:
            true_negative += 1


        print(f"Prediction for {input}: {prediction}, target: {target}")

    accuracy = (true_negative + true_positive)/(true_positive + true_negative + false_positive + false_negative)

    print("Accuracy:", accuracy)
    '''
    weights = [-0.1]
    for i in range(15):
        weights.append(1.0)

    print("Weights:", weights)
    
    learning_rate = 0.1

    
    for i in range(100):
        print("\nFor loop counter for training:", i)
        for inputs, target in zip(training_data, training_target):
            #print("inputs:", inputs, "\ntarget:", target)
            weights = train(weights, inputs, target, learning_rate)
        print("Updated weights:", weights)
            
    print(weights)
    for input, target in zip(training_data, training_target):
        prediction = predict(weights, input)
        print(f"Prediction for {input}: {prediction}, target: {target}")


    '''   
    '''

    def predict(weights, inputs):
        weighted_sum = 0
        for input, weight in zip(inputs[1:], weights[1:]):
            weighted_sum += input * weight
        weighted_sum += weights[0]
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def train(weights, inputs, target, learning_rate):
        prediction = predict(weights, inputs)
        error = target - prediction
        for i in range(1, len(weights) - 1):
            print("\nWeight:", weights[i], "\ninputs:", inputs[1])
            weights[i] += learning_rate * error * inputs[i]
        weights[0] += learning_rate * error
        return weights

    if __name__ == "__main__":
        training_data = []  # training samples
        training_target = []  # labels of class
        with open("survey lung cancer.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                #print(row)
                print(row[0], row[1], row[2])
                
                if row[1].isdigit() is False:
                    training_target.append(row)

                else:
                    if row[-1] == "YES":
                        row[-1] = 1   # Target value for lung cancer

                    else:
                        row[-1] = 0   # Target value for lung cancer.

                training_data.append(row)
                
        
    #print("training_data: ",training_data)
    #print("Y:", training_target)
    weights = [-0.1 , 1 for i in range(15)] 
    
    learning_rate = 0.1

    
    for i in range(100):
        for inputs, target in zip(training_data, training_target):
            print("inputs:", inputs, "\ntarget:", target)
            weights = train(weights, inputs, target, learning_rate)
            
    print(weights)
    for inputs, target in zip(test_data, test_target):
        prediction = predict(weights, inputs)
        print(f"Prediction for {inputs}: {prediction}, target: {target}")

'''