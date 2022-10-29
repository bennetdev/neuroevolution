from __future__ import annotations
from typing import List, Union
import numpy as np
import scipy.special
import json
import random


# NeuralNetwork
# - number of inputnodes
# - number of hiddennodes or list of number of hidden nodes
# - number of outputnodes
# - weights between input-layer and hidden-layer
# - weights between hidden-layers (not existent for one hidden layer)
# - weights between hidden-layer and output-layer
# - learning rate
# - activation function to use across neural network
# TODO bias and neuroevolution not working with multiple hidden layers
class NeuralNetwork:
    def __init__(self, inputnodes: int, hiddennodes: Union[int, List[int]], outputnodes: int, learningrate: float):
        # Nodes are initialized as integers
        self.inodes = inputnodes
        # hidden nodes as list (one list element is one layer)
        self.hnodes = hiddennodes if hasattr(hiddennodes, "__iter__") else [hiddennodes]
        self.onodes = outputnodes

        # init of the weights
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes[0], self.inodes))
        self.whh = []
        self.who = np.random.normal(0.0, pow(self.hnodes[-1], -0.5), (self.onodes, self.hnodes[-1]))
        self.lr = learningrate

        # init biases
        self.biasH = np.random.rand(self.hnodes[0], 1) * 2 - 1
        self.biasO = np.random.rand(self.onodes, 1) * 2 - 1

        # if there are multiple hidden layers
        if len(self.hnodes) > 1:
            self._generate_hidden_weights()

        self.activation_function = lambda x: scipy.special.expit(x)


    # Generates the weights between hidden layers if there are multiple hidden layers
    def _generate_hidden_weights(self) -> None:
        #print(self.hnodes)
        for index in range(0, len(self.hnodes) - 1):
            weights = np.random.normal(0.0, pow(self.hnodes[index], -0.5),
                                          (self.hnodes[index], self.hnodes[index + 1])).tolist()
            self.whh.append(weights)
        self.whh = np.array(self.whh)
        #print(self.whh)

    # train the neural network
    # TODO bias
    # Input: input data, Numpy ndarray
    #        Target data, Numpy dnarray
    def train(self, inputs_list: np.ndarray, targets_list: np.ndarray) -> None:
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        biasHCopy = self.biasH.copy()
        np.concatenate((biasHCopy, self.biasH), axis=1)
        hidden_inputs += biasHCopy
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        outputs = [hidden_outputs]
        hidden_layer_outputs = hidden_outputs
        for index, hidden_layer in enumerate(self.whh):
            hidden_layer_inputs = np.dot(hidden_layer, hidden_layer_outputs)
            hidden_layer_outputs = self.activation_function(hidden_layer_inputs)
            outputs.append(hidden_layer_outputs.tolist())
        outputs = np.array(outputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_layer_outputs)
        biasOCopy = self.biasO.copy()
        np.concatenate((biasOCopy, self.biasO), axis=1)
        final_inputs += biasOCopy
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # print(final_outputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        hidden_layer_errors = hidden_errors
        errors = [hidden_errors]
        for hidden_layer in self.whh:
            hidden_layer_errors = np.dot(hidden_layer.T, hidden_layer_errors)
            errors.append(hidden_layer_errors.tolist())
        errors = np.array(errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_layer_outputs))
        for index, hidden_layer in enumerate(self.whh):
            self.whh[index] += self.lr * np.dot(
                (errors[index + 1] * outputs[index + 1] * (1.0 - outputs[index + 1])),
                np.transpose(outputs[index]))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((errors[0] * outputs[0] * (1.0 - outputs[0])),
                                        np.transpose(inputs))

        # update biases
        self.biasO += (output_errors  * (1.0 - final_outputs)) * self.lr
        self.biasH += (hidden_errors * (1.0 - hidden_outputs)) * self.lr

    # query the neural network for specific input
    # Input: List of input values, np.ndarray with only one dimension (gets converted to 2d array)
    # Output: List of all possible outputs with possibility, Numpy
    def query(self, inputs_list: np.ndarray) -> np.ndarray:
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # add bias to hidden_inputs
        biasHCopy = self.biasH.copy()
        np.concatenate((biasHCopy, self.biasH), axis=1)
        hidden_inputs += biasHCopy

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # only executes with multiple hidden layers
        hidden_layer_outputs = hidden_outputs
        for index, hidden_layer in enumerate(self.whh):
            hidden_layer_inputs = np.dot(hidden_layer, hidden_layer_outputs)
            hidden_layer_outputs = self.activation_function(hidden_layer_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_layer_outputs)
        # add bias to final_inputs
        biasOCopy = self.biasO.copy()
        np.concatenate((biasOCopy, self.biasO), axis=1)
        final_inputs += biasOCopy
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    # save neural network to npy files and metadata to json
    # TODO bias
    def save(self) -> None:
        np.save("save/wih.npy", self.wih)
        np.save("save/whh.npy", self.whh)
        np.save("save/who.npy", self.who)
        np.save("save/biasH.npy", self.biasH)
        np.save("save/biasO.npy", self.biasO)
        with open("save/metadata.json", "w") as file:
            json.dump({
                "inputnodes": self.inodes,
                "hiddennodes": self.hnodes[0],
                "hiddenlayers": len(self.hnodes),
                "outputnodes": self.onodes,
                "learningrate": self.lr
            }, file)

    # Neuroevolution
    
    # Function to (deep) copy the neural networks properties (equals crossover for now)
    # TODO bias
    # Output: Neural Network with the same properties as self
    def copy(self) -> NeuralNetwork:
        copy = NeuralNetwork(self.inodes, self.hnodes, self.onodes, self.lr)
        # copy weights of self
        copy.whh = self.whh.copy()
        copy.wih = self.wih.copy()
        copy.who = self.who.copy()
        copy.biasH = self.biasH.copy()
        copy.biasO = self.biasO.copy()

        return copy

    # Function to create child out of two parent neural networks (including self) by crossing over their weights
    # This implementation chooses every weight off a 50/50 chance, TODO only for one hidden layer right now
    # TODO bias
    # Input: parent2, as NeuralNetwork
    # Output: new child, as NeuralNetwork
    def crossover(self, parent2: NeuralNetwork) -> NeuralNetwork:
        child = self.copy()
        for i in range(self.hnodes[0]-1):
            for j in range(self.inodes-1):
                child.wih[i,j] = self.wih[i,j] if random.uniform(0,1) <= 0.5 else parent2.wih[i,j]
        for i in range(self.onodes - 1):
            for j in range(self.hnodes[0]-1):
                child.who[i,j] = self.who[i,j] if random.uniform(0,1) <= 0.5 else parent2.who[i,j]

        for i in range(self.hnodes[0]):
            child.biasH[i] = self.biasH[i] if random.uniform(0,1) <= 0.5 else parent2.biasH[i]

        for i in range(self.onodes):
            child.biasO[i] = self.biasO[i] if random.uniform(0,1) <= 0.5 else parent2.biasO[i]


        return child


    # Alter every weight by MUTATION_RATE chance to a new float having a maximum difference to previous value of 0.1
    # TODO bias
    # Input: chance of mutation, as float
    def mutate(self, MUTATION_RATE: float) -> None:
        # either change value of weight or leave it, depending on randomness and mutation_rate
        # Input: weight value, as float
        # Output: (new) weight value, as float
        def changeValue(value: float) -> float:
            if random.uniform(0.0,1.0) < MUTATION_RATE:
                # re-initialize weight
                return value + random.uniform(-0.5, 0.5)
            else:
                return value

        changeValueFunction = np.vectorize(changeValue)
        self.wih = changeValueFunction(self.wih)
        # if whh not empty <=> multiple hidden layers
        if not(not self.whh):
            self.whh = changeValueFunction(self.whh)
        self.who = changeValueFunction(self.who)
        self.biasO = changeValueFunction(self.biasO)
        self.biasH = changeValueFunction(self.biasH)


# Load neural network from npy files and json
# Output: Neural Network with metadata and weights from files
# TODO bias
def load() -> NeuralNetwork:
    wih = np.load("save/wih.npy")
    whh = np.load("save/whh.npy")
    who = np.load("save/who.npy")
    biasH = np.load("save/biasH.npy")
    biasO = np.load("save/biasO.npy")
    with open("save/metadata.json", "r") as file:
        metadata = json.load(file)
        n = NeuralNetwork(metadata["inputnodes"], metadata["hiddennodes"] * int(metadata["hiddenlayers"]),
                          metadata["outputnodes"], metadata["learningrate"])
        n.wih = wih
        n.whh = whh
        n.who = who
        n.biasH = biasH
        n.biasO = biasO
    return n