from neural_net_UCI_data import normalize
from neural import *
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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

#2a: normalize and split into x and y

td = normalize(training_data)

td_x = []
td_y = []

for i in range(len(td)):
    td_x.append(td[i][0])
    td_y.append(td[i][1])


#6: create neuralnet

zoo_neural = NeuralNet(16, 3, 1)

#7: test_train_split if possible

x_train, x_test, y_train, y_test = train_test_split(td_x, td_y)

#8: train training data

zoo_train = []

for x in range(len(x_train)):
    set = (x_train[x],y_train[x])
    zoo_train.append(set)

zoo_neural.train(zoo_train, iters=10000, print_interval=1000, learning_rate=0.1)

#9: test testing data

zoo_test = []

for x in range(len(x_test)):
    set = (x_test[x], y_test[x])
    zoo_test.append(set)

#10: find the accuracy, is there a way?

test_with_expected = zoo_neural.test_with_expected(zoo_test)

for i in test_with_expected:
    print("Expected: " , i[1][0] , " | Actual: " , round(i[2][0], 2))