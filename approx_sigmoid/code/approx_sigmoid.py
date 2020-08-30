from random import seed
from random import random
from random import randrange
from math import exp
from csv import reader
import csv

# Network initialization
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range (n_inputs + 1)]} \
                   for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]}  \
                   for i in range (n_outputs)]
    network.append(output_layer)
    return network

# Neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation (sigmoid)
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Values for the piecewise functions
points = list()
slopes = list()
intercepts = list()
y_val = list()

# Split the sigmoid function into the desired amount of segments
def segment_sigmoid (n_segments):
    width = 10.0/(n_segments - 2)
    x_val = -5.0
    prev_sig_val = 0.0
    for i in range(n_segments - 2):
        next_step = x_val + width
        sig_val = 1.0 / (1.0 + exp(-1 * next_step))

        points.append(next_step)
        slopes.append( (sig_val - prev_sig_val) / (next_step - x_val))
        intercepts.append(sig_val - (slopes[-1] * next_step))
        y_val.append(sig_val)

        x_val = next_step
        prev_sig_val = sig_val

# Transfer neuron activation (piecewise linear)
def transfer_pwl(activation):
    if (activation < -5):
        return 0.0
    elif (5 < activation):
        return 1.0
    else:
        for i in range(len(points)):
            if (activation <= points[i]):
                return slopes[i] * activation + intercepts[i]

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Forward propagate input to a network output (piecewise)
def forward_propagate_pw(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            if (layer == network[-1]):
                neuron['output'] = transfer(activation)
            else:
                neuron['output'] = transfer_pwl(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of a neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Calculate the derivative of a neuron output (piecewise)
def transfer_derivative_pw(output):
    if (output < 1.0 / (1.0 + exp(5))):
        return 0.0
    for i in range(len(y_val)):
        if (output < y_val[i]):
            return slopes[i]
    return 1.0

# Back propagate error and store in the neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) -1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i +1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i -1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    csvData = list()
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [row[-1]]
            sum_error += (expected[0]-outputs[0])**2
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        satisfied_error = sum_error/len(train)
        print('>epoch=%d, lrate=%.3f, error=%.10f' % (epoch, l_rate, sum_error/len(train)))
        csvData.append([epoch + 1, sum_error/len(train)])

    with open('data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset.append(row)
    return dataset

# Convert dataset into integers
def ConvertDataset(dataset):
    con_data = list()
    for row in dataset:
        con_row = list()
        for item in row:
            if (item =='N'):
                con_row.append(0.0)
            elif (item == 'O'):
                con_row.append(1.0)
            else:
                con_row.append(float(item))
        con_data.append(con_row)
    return con_data

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# load and prepare data
seed(1)
filename = 'fertility.csv'
dataset = load_csv(filename)
dataset = ConvertDataset(dataset)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_inputs = len(dataset[0]) - 1
network = initialize_network(n_inputs, 2 * n_inputs, 1)

segment_sigmoid (14)

train_network(network, dataset, .15, 5000, 1)
