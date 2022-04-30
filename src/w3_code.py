import mnist_loader
import network
import numpy as np
import matplotlib.pyplot as plt

## dimensions [size][x/y][784][1]
#n = ### YOUR CODE HERE ###
#q1##############################
n = (732003960+231002947)%24+4
def question_1a():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network.SGD([784, n, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
########################################################
##q2#####################################################
def question_1b():
    mini_batch = [20,32,64,128,512]
    eta_values = []
    
    # for i in range(-3,10):
        # eta_values.append(i)
        
    for i in mini_batch:
        
    #for j in eta_values:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        net = network.Network([784,n,10])
        net.SGD(training_data, 30, i, 5, test_data=validation_data)
        
#Found best eta_value as 5  and best mini batch size as 10
        
###############################################################
##########q21###########################################
def find_accuracy(test_data, models):
    acc = 0 
    for t_point, t_results in test_data:
        pred = np.zeros(len(models))
        for i,j in enumerate(models):
            predict = np.argmax(j.feedforward(t_point))
            pred[i] = predict
        values, count = np.unique(pred, return_counts = True)
        ind = np.argmax(count)
        pred_final = values[ind]
        
        if isinstance(t_results, int) or isinstance(t_results, np.int64):
            if pred_final == t_results:
                acc = acc+1
        else:
            if pred_final == np.argmax(t_results):
                acc = acc+1
            
    return acc
        
            
            
    
def question_2(batch_size, eta):
    m_values = range(1,22)
    tr_d, va_d, te_d = mnist_loader.load_data()
    test_accuracy_list = []
    train_accuracy_list = []
    for m in m_values:
        models = []
        print(m)
        for i in range(m):
            ran_choice = np.random.choice(len(tr_d[0]),len(tr_d[0]))
            
            m_data = tr_d[0][ran_choice]
            m_results = tr_d[1][ran_choice]
            training_inputs = [np.reshape(x, (784, 1)) for x in m_data]
            training_results = [mnist_loader.vectorized_result(y) for y in m_results]
            training_data = zip(training_inputs, training_results)
            # validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
            # validation_data = zip(validation_inputs, va_d[1])
            test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
            test_data = zip(test_inputs, te_d[1])
            
            net = network.Network([784,n,10])
            net.SGD(training_data, 30, batch_size, eta, test_data=test_data)
            
            models.append(net)
            
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [mnist_loader.vectorized_result(y) for y in tr_d[1]]
        training_data = zip(training_inputs, training_results)
        
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])
        
        test_accuracy = find_accuracy(test_data, models)  
        test_accuracy_list.append(test_accuracy/len(te_d[0]))
        training_accuracy = find_accuracy(training_data, models) 
        train_accuracy_list.append(training_accuracy/len(tr_d[0]))
        print("train accuracy", train_accuracy_list, "test accuracy", test_accuracy_list)
    with open('q2_output.txt','w') as f:
        f.write(f"Test Accuracies: {test_accuracy_list}, Train Accuracy: {train_accuracy_list}")
    return train_accuracy_list, test_accuracy_list

def plot_accuracy(train_accuracy_list, test_accuracy_list):
    plt.figure(figsize = (8,5)) 
    x = np.arange(1, len(train_accuracy_list)+1, 1)
    plt.plot(x,train_accuracy_list, "-x")
    plt.plot(x,test_accuracy_list, "-o")
    plt.xlabel("Ensemble size")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Test"])
    plt.title("Accuracy vs. Ensemble size")        

train_accuracy_list, test_accuracy_list = question_2(10,5) 
plot_accuracy(train_accuracy_list, test_accuracy_list)

#question_1a()
#question_1b()      