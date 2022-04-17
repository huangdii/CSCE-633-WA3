import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
## dimensions [size][x/y][784][1]
#n = ### YOUR CODE HERE ###
#net = net.SGD([784, n, 10])
net = network.Network([784, 10, 10])

#print(len(list(training_data)), validation_data, test_data)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
