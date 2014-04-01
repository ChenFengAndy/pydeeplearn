import numpy as np

import restrictedBoltzmannMachine as rbm
import theano
from theano import tensor as T

theanoFloat  = theano.config.floatX

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *


def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

# This is a better logical unit
# than having the dbn store the layer values
# this is because the layer values
class MiniBatchTrainer(object):

  def __init__(self, input, nrLayers, initialWeights, initialBiases):
    self.input = input

    # Let's initialize the fields
    # The weights and biases, make them shared variables
    self.weights = []
    self.biases = []
    for i in xrange(nrLayers - 1):
      w = theano.shared(value=np.asarray(initialWeights[i],
                                         dtype=theanoFloat),
                        name='W')
      self.weights.append(w)

      b = theano.shared(value=np.asarray(initialBiases[i],
                                         dtype=theanoFloat),
                        name='b')
      self.biases.append(b)

    # Set the parameters of the object
    # Do not set more than this, these will be used for differentiation in the
    # gradient
    self.params = self.weights + self.biases

    currentLayerValues = self.input
    self.layerValues = [0 for x in xrange(nrLayers)]
    self.layerValues[0] = currentLayerValues
    for stage in xrange(len(self.weights)):
      w = self.weights[stage]
      b = self.biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      if stage != len(self.weights) -1:
        currentLayerValues = T.nnet.sigmoid(linearSum)
      else:
        e_x = T.exp(linearSum - linearSum.max(axis=1, keep_dims=True))
        currentLayerValues = e_x / e_x.sum(axis=1, keep_dims=True)
        # currentLayerValues = T.nnet.softmax(linearSum)

      self.layerValues[stage + 1] = currentLayerValues

  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.layerValues[-1], y)

""" Class that implements a deep belief network, for classification """
class DBN(object):

  """
  Arguments:
    nrLayers: the number of layers of the network. In case of discriminative
        traning, also contains the classifcation layer
        (the last softmax layer)
        type: integer
    layerSizes: the sizes of the individual layers.
        type: list of integers of size nrLayers
    activationFunctions: the functions that are used to transform
        the input of a neuron into its output. The functions should be
        vectorized (as per numpy) to be able to apply them for an entire
        layer.
        type: list of objects of type ActivationFunction
  """
  def __init__(self, nrLayers, layerSizes, activationFunctions,
               dropout=0.5, rbmDropout=0.5, visibleDropout=0.8, rbmVisibleDropout=1):
    self.nrLayers = nrLayers
    self.layerSizes = layerSizes
    # Note that for the first one the activatiom function does not matter
    # So for that one there is no need to pass in an activation function
    self.activationFunctions = activationFunctions

    assert len(layerSizes) == nrLayers
    assert len(activationFunctions) == nrLayers - 1
    self.miniBatchSize = 10

  def train(self, data, labels=None):

    # This depends if you have generative or not
    nrRbms = self.nrLayers - 2

    self.weights = []
    self.biases = []

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))

    # Train the restricted Boltzmann machines that form the network
    currentData = data
    for i in xrange(nrRbms):
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                    rbm.contrastiveDivergence,
                    1, 1,
                    self.activationFunctions[i].value)
      net.train(currentData)

      w = net.weights
      self.weights += [w]
      b = net.biases[1]
      self.biases += [b]

      # Let's update the current representation given to the next RBM
      currentData = net.hiddenRepresentation(currentData)

    # This depends if you have generative or not
    # Initialize the last layer of weights to zero if you have
    # a discriminative net
    lastLayerWeights = np.zeros(shape=(self.layerSizes[-2], self.layerSizes[-1]),
                                dtype=theanoFloat)
    lastLayerBiases = np.zeros(shape=(self.layerSizes[-1]),
                               dtype=theanoFloat)

    self.weights += [lastLayerWeights]
    self.biases += [lastLayerBiases]

    assert len(self.weights) == self.nrLayers - 1
    assert len(self.biases) == self.nrLayers - 1

    self.nrMiniBatches = len(data) / self.miniBatchSize

    # Does backprop for the data and a the end sets the weights
    self.fineTune(sharedData, sharedLabels)

    # TODO: put it back in with dropout
    # self.classifcationWeights = map(lambda x: x * self.dropout, self.weights)
    # here this is float64, and it is composed
    self.classifcationWeights =  self.weights
    self.classifcationBiases = self.biases

  """Fine tunes the weigths and biases using backpropagation.
    data and labels are shared

    Arguments:
      data: The data used for traning and fine tuning
        data has to be a theano variable for it to work in the current version
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  def fineTune(self, data, labels, epochs=100):
    learningRate = 0.1
    # batchLearningRate = learningRate / self.miniBatchSize

    nrMiniBatches = self.nrMiniBatches
    # Let's build the symbolic graph which takes the data trough the network
    # allocate symbolic variables for the data
    # index of a mini-batch
    miniBatchIndex = T.lscalar()

    # The mini-batch data is a matrix
    x = T.matrix('x', dtype=theanoFloat)
    # The labels, a vector
    y = T.matrix('y', dtype=theanoFloat) # labels[start:end] this needs to be a matrix because we output probabilities

    # here is where you can create the layered object
    # the mdb and with it you associate the cost function
    # and you update it's parameters
    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.weights,
                                    initialBiases=self.biases)

    # the error for this batch: it is the sum of the errors obtained from
    # each data instance
    error = T.sum(batchTrainer.cost(y))

    deltaParams = T.grad(error, batchTrainer.params)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # The parameters to be updated
    parametersTuples = zip(batchTrainer.params, deltaParams)
    for param, delta in parametersTuples:
        paramUpdate = np.float32(learningRate) * delta
        newParam = param + paramUpdate
        updates.append((param, newParam))

    mode = theano.compile.MonitorMode(
      # pre_func=inspect_inputs,
      post_func=detect_nan).excluding(
    'local_elemwise_fusion', 'inplace')

    train_model = theano.function(
            # inputs=[miniBatchIndex, momentum],
            inputs=[miniBatchIndex],
            outputs=error,
            updates=updates,
            givens={
                x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
                y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]},
                mode=mode)

    # TODO: early stopping
    for epoch in xrange(epochs):
      print "in if"
      # When you do early stopping you have to return the error on this batch
      # so that you can see when you stop or not
      # you have to pass in the momentum here as well as a parameter for
      # the trainmodel
      for batchNr in xrange(nrMiniBatches):
        error = train_model(batchNr)

    # Let's put the weights back in the dbn class as they are used for classification
    # Note that if you leave it like this you od not have
    # to deal with the random theano stuff
    print "weights"
    for i in xrange(len(self.weights)):
      self.weights[i] = batchTrainer.weights[i].get_value()
      print self.weights[i]

    print "biases"
    for i in xrange(len(self.biases)):
      self.biases[i] = batchTrainer.biases[i].get_value()
      print self.biases[i]

  def classify(self, dataInstaces):
    # TODO: run it on the gpu according to the number of instances
    # I think it is better to just run it on GPU
    # The mini-batch data is a matrix
    dataInstacesConverted = np.asarray(dataInstaces, dtype=theanoFloat)

    x = T.matrix('x', dtype=theanoFloat)
    # TODO: move this to classification weights when you have
    # dropout back in
    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.weights,
                                    initialBiases=self.biases)
    classify = theano.function(
            inputs=[],
            outputs=batchTrainer.layerValues[-1],
            updates={},
            givens={x: dataInstacesConverted})

    lastLayers = classify()

    return lastLayers, np.argmax(lastLayers, axis=1)

