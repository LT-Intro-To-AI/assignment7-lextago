from neural_net_UCI_data import normalize
from neural import *
import pandas as pd
from sklearn.model_selection import train_test_split

#1: parse through

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    tokens = line.split(",")
    out = int(tokens[16])
    output = [0 if out == 1 
              else 0.16 if out == 2 
              else 0.32 if out == 3
              else 0.48 if out == 4
              else 0.64 if out == 5
              else 0.8 if out == 6
              else 1]

    inpt = [float(x) for x in tokens[0:16]]
    return (inpt, output)

with open("zoo_animals.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines()]

# x = data[["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic"]].values
# y = data["class_type"].values


#2a: normalize

td = normalize(training_data)


#6: create neuralnet

zoo_neural = NeuralNet(16, 3, 1)

#7: test_train_split if possible

x_train, x_test, y_train, y_test = train_test_split(td[0], td[1])

#8: train training data

# zoo_train = []

# for x in range(len(x_train)):
#     set = (x_train[x].tolist(),[y_train[x]])
#     zoo_train.append(set)

# zoo_neural.train( zoo_train )

#9: test testing data

# zoo_test = []

# for x in range(len(x_test)):
#     set = (x_test[x].tolist(), [y_test[x]])
#     zoo_test.append(set)

# print(zoo_neural.test_with_expected(zoo_test))

#10: find the accuracy, is there a way?






