"""
DOCSTRING
"""
import copy
import numpy

numpy.random.seed(0)

def sigmoid(x):
    """
    compute sigmoid nonlinearity
    """
    output = 1/(1+numpy.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    """
    convert output of sigmoid function to its derivative
    """
    return output*(1-output)

# training dataset generation
int2binary = {}
binary_dim = 8
largest_number = pow(2,binary_dim)
binary = numpy.unpackbits(
    numpy.array([range(largest_number)],dtype=numpy.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2*numpy.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*numpy.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*numpy.random.random((hidden_dim,hidden_dim)) - 1
synapse_0_update = numpy.zeros_like(synapse_0)
synapse_1_update = numpy.zeros_like(synapse_1)
synapse_h_update = numpy.zeros_like(synapse_h)

# training logic
for j in range(10000):
    a_int = numpy.random.randint(largest_number/2)
    a = int2binary[a_int]
    b_int = numpy.random.randint(largest_number/2)
    b = int2binary[b_int]
    c_int = a_int + b_int
    c = int2binary[c_int]
    d = numpy.zeros_like(c)
    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(numpy.zeros(hidden_dim))
    for position in range(binary_dim):
        X = numpy.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = numpy.array([[c[binary_dim - position - 1]]]).T
        layer_1 = sigmoid(numpy.dot(X,synapse_0) + numpy.dot(layer_1_values[-1],synapse_h))
        layer_2 = sigmoid(numpy.dot(layer_1,synapse_1))
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += numpy.abs(layer_2_error[0])
        d[binary_dim - position - 1] = numpy.round(layer_2[0][0])
        layer_1_values.append(copy.deepcopy(layer_1))
    future_layer_1_delta = numpy.zeros(hidden_dim)
    for position in range(binary_dim):
        X = numpy.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        layer_2_delta = layer_2_deltas[-position-1]
        layer_1_delta = ((future_layer_1_delta.dot(synapse_h.T)
                         + layer_2_delta.dot(synapse_1.T))
                         * sigmoid_output_to_derivative(layer_1))
        synapse_1_update += numpy.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += numpy.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        future_layer_1_delta = layer_1_delta
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
