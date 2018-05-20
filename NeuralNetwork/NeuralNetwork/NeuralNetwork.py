import numpy as np
import random
import scipy




class NeuralNetwork:

    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons, activationFunction, activationFunctionDerivative, learningRate):
        self.inputNeurons = inputNeurons
        self.hiddenNeurons = hiddenNeurons
        self.outputNeurons = outputNeurons

        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative
        self.learningRate = learningRate

        self.wih = np.random.rand(hiddenNeurons,inputNeurons) - 0.5
        self.hiddenBias = np.random.rand(self.hiddenNeurons,1) - 0.5
        self.who = np.random.rand(outputNeurons,hiddenNeurons) - 0.5
        self.outputBias = np.random.rand(self.outputNeurons,1) - 0.5

    pass

    def query(self, inputList):
       inputs = np.array(inputList,ndmin=2).T

       hiddenInputs = np.dot(self.wih , inputs) + self.hiddenBias
       hiddenOutputs = self.activationFunction(hiddenInputs)

       finalInputs = np.dot(self.who, hiddenOutputs) + self.outputBias
       finalOutputs = self.activationFunction(finalInputs)


       return finalOutputs

    pass


    def train(self, inputList, targetList):

       inputs = np.array(inputList, ndmin=2).T
       targets = np.array(targetList, ndmin=2).T

       hiddenInputs = np.dot(self.wih , inputs) + self.hiddenBias
       hiddenOutputs = self.activationFunction(hiddenInputs)

       finalInputs = np.dot(self.who, hiddenOutputs) + self.outputBias
       finalOutputs = self.activationFunction(finalInputs)

       deltaO = (finalOutputs - targets) * self.activationFunctionDerivative(finalInputs)
       #deltaO = (finalOutputs - targets) * finalOutputs * (1 - finalOutputs)

       self.who = self.who - self.learningRate * np.dot(deltaO, hiddenOutputs.T)
       self.outputBias = self.outputBias - self.learningRate * deltaO;

       deltaH = np.dot(self.who.T,deltaO) * self.activationFunctionDerivative(hiddenInputs)
       self.wih = self.wih - self.learningRate * np.dot(deltaH, inputs.T)
       self.hiddenBias = self.hiddenBias - self.learningRate * deltaH


       pass 



pass


n = NeuralNetwork(4,3,2,lambda x: (1/(1+np.exp(-x))) * (1-(1/(1+np.exp(-x)))),0.5)
n1 = NeuralNetwork(4,3,2,lambda x: np.tanh(x),lambda x: 1/(np.cosh(x)*np.cosh(x)),0.5)
n2 = NeuralNetwork(4,3,2,lambda x: np.maximum(0,x),lambda x: np.heaviside(x,0),0.5)

print(n.query([0.7,0.3,0.1,0.2]))
print(n1.query([0.7,0.3,0.1,0.2]))
print(n2.query([0.7,0.3,0.1,0.2]))

for i in range(50):
    n.train([0.7,0.3,0.1,0.2],[0.99,0.01])
    n1.train([0.7,0.3,0.1,0.2],[0.99,0.01])
    n2.train([0.7,0.3,0.1,0.2],[0.99,0.01])
    pass
print(n.query([0.7,0.3,0.1,0.2]))
print(n1.query([0.7,0.3,0.1,0.2]))
print(n2.query([0.7,0.3,0.1,0.2]))

