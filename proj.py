from typing import Tuple
import pandas as pd
from neural import NeuralNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#1: parse through

data = pd.read_csv("zoo_animals.csv")

x = data[["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize"]].values
y = data["class_type"].values


#2a: normalize

# scaler = StandardScaler().fit(x)

# x = scaler.transform(x)

#6: create neuralnet

zoo_neural = NeuralNet(16, 25, 1)

#7: test_train_split if possible

x_train, x_test, y_train, y_test = train_test_split(x, y)

#8: train training data

zoo_train = []

for x in range(len(x_train)):
    zoo_set = (x_train[x].tolist(),[y_train[x]])
    zoo_train.append(zoo_set)

zoo_neural.train( zoo_train )


#9: test testing data

print(zoo_neural.test_with_expected(x_test.tolist()))

#10: find the accuracy, is there a way?




