
import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class activateRELU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs)
    def backwards(self, gradients):
        self.dinputs = gradients.copy()
        self.dinputs[self.inputs <= 0] = 0

        #apply the activateRelU function to the derivatives - because values that
        #went 0 should not have an impact after this point 

class softMaxFunction:
    def forward(self, inputs):
        newInputs = np.exp(inputs - inputs.max(axis = 1, keepdims = True))
        #The subtrction exist to prevent an overflow
        self.outputs = newInputs/np.sum(newInputs, axis = 1, keepdims = True) 
    def backwards(self, dvalues):
        for index, (outputs, dvalue) in enumerate(self.outputs, dvalues):
            jacobMatrix = np.diagflat(outputs.reshape(-1,1)) - np.dot(outputs, outputs.T)
            self.dinputs[index] = np.dot(jacobMatrix, dvalue)

    def predictions(self):
        #print('predictions')
        #print(self.outputs)
        return np.argmax(self.outputs, axis = 1)

#For binary regression
class sigmoidFunction:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1/(1 + np.exp(-self.inputs))

    def backwards(self, dvalues):
        self.dinputs = dvalues * self.outputs * (1 - self.outputs)

    def predictions(self):
        return (self.outputs > 0.5) * 1

class LinearActivation:
    def forward(self, inputs):
        self.inputs = inputs 
        self.outputs = inputs 

    def backwards(self, dvalues):
        self.dinputs = dvalues.copy()

    def calculatePrediction(self,y, rangeLimit = 250):
        accuracyRange = np.std(y)/rangeLimit
        return np.abs(y - self.outputs) < accuracyRange

    def predictions(self):
        return self.outputs
class FunctionLoss:
    def calculate(self, outputs, targetedClass, includeRegularization = False):
        sampleLoss = self.forward(outputs, targetedClass)
        dataLoss = np.mean(sampleLoss)

        self.accumalatedSum += np.sum(sampleLoss)
        self.accumalatedCount += len(sampleLoss)
        
        if includeRegularization:
            return dataLoss, self.regularizationForward()
        
        return dataLoss 

    def regularizationForward(self):
        lossRegularizationloss = 0 

        for layer in self.layersList:
            #L1 Regularization
            lossRegularizationloss += layer.weightsRegularizationL1 * np.sum(np.abs(layer.weights))
            lossRegularizationloss += layer.biasRegularizationL1 * np.sum(np.abs(layer.bias))
            
            #L2 Regularization
            lossRegularizationloss += layer.weightsRegularizationL2 * np.sum(np.abs(layer.weights**2))
            lossRegularizationloss += layer.biasRegularizationL2 * np.sum(np.abs(layer.bias**2))

        return lossRegularizationloss
    
    def overallLoss(self):
        return self.accumalatedSum / self.accumalatedCount, self.regularizationForward()
    
    def rememberNeuronLayers(self, listOfLayers):
        self.layersList = listOfLayers

    def newPass(self):
        self.accumalatedSum = 0 
        self.accumalatedCount = 0 

class LossCategoricalCrossEntropy(FunctionLoss):
    def forward(self, outputs, targetedClass):
        #There are two types - either they return a single dimension list specifying which index they are 
        #targetting for each dimension or a multi-dimensional list specificying the value they want 
        #IF MULTIDIMENSIONAL IT MUST BE AN NPARRAY
        '''
        ex 1: [1,2,0]
        ex 2, [[0,1,0], [0,0,1], [1,0,0]]
        '''
        try: 
            targetedClass.shape
        except AttributeError: 
            targetedClass = np.array(targetedClass)
        
        if len(targetedClass.shape) == 1: #check to see if the list is single 
            selectedResults = outputs[range(len(outputs)), targetedClass]
        
        if len(targetedClass.shape) == 2: #check to see if the list is multi-dimensional 
            selectedResults = np.sum(outputs * targetedClass, axis = 1, keepdims = True)
        
        selectedResults =np.clip(selectedResults, 1e-7, 1-1e-7)
        
        return -np.log(selectedResults)
    
    def backwards(self, outputs, realY):
        dimensions = outputs.shape
        if len(dimensions) == 1:
            realY = np.eye(dimensions[1])[realY]
        self.dinputs = -realY / outputs
        self.dinputs = self.dinputs/dimensions[0]

class LossBinaryCrossEntropy(FunctionLoss):
        def forward(self, inputs, realY):
            clippedY =  np.clip(inputs, 1e-7, 1-1e-7)

            perSampleLoss = -(realY * np.log(clippedY)) + (1 - realY) * np.log(1 - clippedY)
            #? Left side calculates how wrong the variable is from being 1 and the right side calculate
            #? how wrong it is from being 0 

            return np.mean(perSampleLoss, axis = -1)
        
        def backwards(self, dvalues, realY):
            clippedDvalues = np.clip(dvalues, 1e-7, 1-1e-7)
            dimensions = dvalues.shape
            self.dinputs  = -((realY/clippedDvalues) - (1-realY)/(1-clippedDvalues))/dimensions[1]/dimensions[0]


class MeanSquaredError(FunctionLoss):
    def forward(self, outputs, realY):
        sampleLoss = np.mean((realY - outputs)**2, axis = -1)
        return sampleLoss
    
    def backwards(self, outputs, realY):
        dimensions = outputs.shape
        self.dinputs = -2*(realY - outputs)/dimensions[0]/dimensions[1]

class MeanAbsoluteError(FunctionLoss):
    def forward(self, outputs, realY):
        sampleLoss = np.mean(np.abs((realY - outputs)), axis = -1)
        return sampleLoss

    def backwards(self, dvalues, realY):
        dimensions = dvalues.shape
        self.dinputs = np.sign(realY - dvalues)/dimensions[1]/dimensions[0]
        #! The dinputs is too large - Don't use this

class CategoricalCrossEntropy:
    def __init__(self):
        self.softmax = softMaxFunction()
        self.loss = LossBinaryCrossEntropy()

    def forward(self, inputs, y_true):
        self.softmax.forward(inputs)
        self.outputs = self.softmax.outputs
        self.loss.forward(self.outputs, y_true)
        
        return self.loss.outputs

    def backwards(self, dvalues, y_true):
        dimensions = y_true.shape

        if len(dimensions) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(dimensions[0]), y_true] -= 1
        #Derivative is equal to subtract the desired outputs by 1
        self.dinputs = self.dinputs/dimensions[0]
        #Normalize it

class dropoutLayer:
    def __init__(self, rate):
        self.rate = 1 - rate 
    
    def forward(self, inputs, training = True):
        self.inputs = inputs
        self.binomial = np.random.binomial(1, self.rate, size = inputs.shape)
        if not training: 
            self.outputs = self.inputs.copy()
            return 
        self.outputs = self.inputs * self.binomial
        
    def backwards(self, dvalues):
        self.dinputs = dvalues * self.binomial
   
class neuroNetworkLayer:
    def __init__(self, itemsInArray, Dimensions, weightsRegularizationL1 = 0, biasRegularizationL1 = 0, weightsRegularizationL2 = 0, biasRegularizationL2 = 0):
        self.weights = 0.01 * np.random.randn(itemsInArray, Dimensions)
        #Weights must be as close as possible to zero 
        #! self.weights = np.zeros((itemsInArray, Dimensions))
        self.bias = np.zeros((1,Dimensions))
        #! bias MUST START WITH ZEROS
        self.weightsRegularizationL1 = weightsRegularizationL1
        self.biasRegularizationL1 = biasRegularizationL1
        self.weightsRegularizationL2 = weightsRegularizationL2
        self.biasRegularizationL2 = biasRegularizationL2

        #! Only apply regularization at the second Layer
    
    def forward(self,inputs):
        self.input = inputs 
        self.outputs = np.dot(self.input, self.weights) + self.bias #softMaxFunction(pushForward(np.dot(self.input,self.weights)) + self.bias)
        #print(self.outputs)

    def backwards(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)

        self.dbias = np.sum(dvalues, axis = 0, keepdims = True)
        

        #Backward pass of the regularization
        if self.weightsRegularizationL1 > 0:
            dL1weights = np.ones_like(self.weights)
            dL1weights[self.weights < 0] = -1
            
            self.dweights += self.weightsRegularizationL1 * dL1weights
        
        if self.biasRegularizationL1 > 0:
            dL1weights = np.ones_like(self.bias)
            dL1weights[self.bias < 0] = -1

            self.dbias += self.biasRegularizationL1 * dL1weights
        
        if self.weightsRegularizationL2 > 0:
            self.dweights += self.weightsRegularizationL2 * 2 * self.weights
        
        if self.biasRegularizationL2 > 0:
            self.dbias += self.biasRegularizationL2 * 2 * self.bias

        self.dinputs = np.dot(dvalues, self.weights.T)

class Accuracy:
    def calculate(self, outputs, y):
        predictions = self.compare(outputs, y)
        accuracy = np.mean(predictions)

        self.accumalatedPredictions += np.sum(predictions)
        self.accumalatedCount += len(predictions)

        return accuracy

    def calculateAccumalated(self):
        return self.accumalatedPredictions/self.accumalatedCount

    def newPass(self):
        self.accumalatedPredictions = 0
        self.accumalatedCount = 0

class categoricalAccuracy(Accuracy):
    def __init__(self, binary = False):
        self.binary = binary
        #Binary checks whether is it for a binary output

    def init(self, y):
        pass
        # empty for the sake of consistency
    
    def compare(self, y, predictions):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y

class regressionAccuracy(Accuracy):
    def __init__(self):
        self.precision = None 
        #This is used to indicate the range value for linear regression to be considered accurate
    
    def init(self, y, reinit = False):
        #reinit is for resetting the precision when needed after creating the model
        if self.precision is None or reinit:
            self.precision = np.std(y)/250

    def compare(self, output, y):
        return np.abs(output - y) < self.precision

class NeuroModel:
    def __init__(self):
        self.layers = []
        self.neuronLayers = []
        self.NoOfLayers = 0 

    def add(self, layer):
        self.layers.append(layer)

        if isinstance(layer, neuroNetworkLayer):
            self.neuronLayers.append(layer)

        self.NoOfLayers = len(self.layers)

    def set(self, lossLayer, optimizer, accuracy):
        self.lossLayer = lossLayer
        self.optimize = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.lossLayer.rememberNeuronLayers(self.neuronLayers)
        
        if isinstance(self.layers[-1], softMaxFunction) and isinstance(self.lossLayer, LossCategoricalCrossEntropy):
            self.categoricalCrossEntropyCounter = True
            self.categoricalCrossEntropy = CategoricalCrossEntropy()
        else:
            self.categoricalCrossEntropyCounter = False
        
            

    def train(self, X, y, epochs, printEvery, training = True, batchSize = None):
        self.accuracy.init(y)

        if batchSize is not None:
            steps = len(X)//batchSize
            if steps * batchSize > len(X):
                steps += 1
        
        else:
            steps = 1
            validationSteps = 1

        for epoch in range(epochs + 1):
            self.lossLayer.newPass()
            self.accuracy.newPass()
            
            for step in range(steps): 
           

                Xbatch = X[step * batchSize: (step + 1) * batchSize] if batchSize != None else X
                ybatch = y[step * batchSize: (step + 1) * batchSize] if batchSize != None else y
                output = self.forward(Xbatch, training)

                normalLoss, regularizationLoss = self.lossLayer.calculate(output, ybatch, includeRegularization = True)
                actualLoss = normalLoss + regularizationLoss
                predictions = self.layers[-1].predictions()
                accuracy = self.accuracy.calculate(predictions, ybatch)

                if not step % 100:
                    print('s', step)
                    print('p', accuracy)
                    print('l', actualLoss)
                    print('nl', normalLoss)
                    print('rl', regularizationLoss)
                    print('\n')
            

                self.backwards(output, ybatch)   

                self.optimize.adjustRate()
                for layers in self.neuronLayers:
                    self.optimize.updateWeightsBias(layers)
                self.optimize.updateStep()
            
            normalLoss, regularizationLoss = self.lossLayer.overallLoss()
            actualLoss = normalLoss + regularizationLoss
            accuracy = self.accuracy.calculateAccumalated()
            if not epoch % printEvery:
                print('Validation')
                print('e', epoch)
                print('p', accuracy)
                print('l', actualLoss)
                print('nl', normalLoss)
                print('rl', regularizationLoss)
                print('\n')

    def evaluate(self, Xtest, ytest, batchSize = None):
        self.lossLayer.newPass()
        self.accuracy.newPass()

        if Xtest is not None:
            validationSteps = len(Xtest)//batchSize
            if validationSteps * batchSize > len(Xtest):
                validationSteps += 1

        else:
            validationSteps = 1
        
        for step in range(validationSteps):         
            Xbatch = Xtest[step * batchSize: (step + 1) * batchSize] if batchSize != None else Xtest
            ybatch = ytest[step * batchSize: (step + 1) * batchSize] if batchSize != None else ytest
            output = self.forward(Xbatch, False)

            normalLoss, regularizationLoss = self.lossLayer.calculate(output, ybatch, includeRegularization = True)
            actualLoss = normalLoss + regularizationLoss
            predictions = self.layers[-1].predictions()
            accuracy = self.accuracy.calculate(predictions, ybatch)



        normalLoss, regularizationLoss = self.lossLayer.overallLoss()
        actualLoss = normalLoss + regularizationLoss
        accuracy = self.accuracy.calculateAccumalated()

        print('Test')
        print('p', accuracy)
        print('l', actualLoss)
        print('nl', normalLoss)
        print('rl', regularizationLoss)
        print('\n')

    def forward(self, X, training):
        
        self.layers[0].forward(X)
        
        for i in range(1, self.NoOfLayers):
            if not isinstance(self.layers[i], dropoutLayer):
                self.layers[i].forward(self.layers[i - 1].outputs)
            else:
                self.layers[i].forward(self.layers[i-1].outputs, training)

        return self.layers[-1].outputs
    
    def backwards(self, output, y):
        if not self.categoricalCrossEntropyCounter:
            self.lossLayer.backwards(output, y)
            self.layers[-1].backwards(self.lossLayer.dinputs)
        else:
            self.categoricalCrossEntropy.backwards(output, y)

        for i in range(self.NoOfLayers - 2, -1, -1):
            if self.categoricalCrossEntropyCounter and i == self.NoOfLayers - 2:
                self.layers[i].backwards(self.categoricalCrossEntropy.dinputs)
            else:
                self.layers[i].backwards(self.layers[i + 1].dinputs)

    def predict(self, X):
        output = self.forward(X, False)

        return self.layers[-1].predictions()


'''
X,y = spiral_data(samples = 100, classes = 3)
Xtest, Ytest = spiral_data(samples = 100, classes = 3)

#y = y.reshape(-1,1)
#Ytest = Ytest.reshape(-1,1)

model = NeuroModel()

model.add(neuroNetworkLayer(2, 32, weightsRegularizationL2 = 5e-4, biasRegularizationL2=5e-4))
model.add(activateRELU())
model.add(neuroNetworkLayer(32,3))
model.add(softMaxFunction())

model.set(LossCategoricalCrossEntropy(), op.Optimizer_Adam(rate = 0.02, decay = 5e-7), categoricalAccuracy())
model.finalize()

model.train(X, y, 10001, 100, Xtest, Ytest, batchSize = 50)
'''

