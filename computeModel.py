# Create by MrZhang on 2019-11-22

import numpy as np


def initialize_weights(vocab_len, dim):

    parameters = {}

    # np.random.seed(1)

    W1 = np.random.randn(dim, vocab_len)
    W2 = np.random.randn(vocab_len, dim)

    # W1 = np.zeros([dim, vocab_len])
    # W2 = np.zeros((vocab_len, dim))

    assert (W1.shape == (dim, vocab_len))
    assert (W2.shape == (vocab_len, dim))

    parameters["W1"] = W1
    parameters["W2"] = W2

    return parameters

def softmax(input_array):
    return np.exp(input_array) / np.sum(np.exp(input_array), axis=0, keepdims=True)

def matrixMult(input, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z2 = np.dot(W2, np.dot(W1, input.T))
    return Z2

def softOutput(Z2):
    soft_array = softmax(Z2)
    return soft_array.T

def forwardProcess(input_data, parameters):

    # parameters = initialize_weights(vocab_len, dim)
    Z2 = matrixMult(input_data, parameters)
    output_array = softOutput(Z2)

    return output_array

def loss_function(input_data, parameters, truth_words):
    true_index_array = np.argmax(truth_words,axis=1)
    centre_index_array = np.argmax(input_data, axis=1)
    # print(true_index_array)
    # print(centre_index_array)
    V = parameters["W1"]
    U = parameters["W2"]
    vocab_len = np.shape(V)[1]
    # print("the length is:", len(true_index_array))

    loss = 0
    for i in range(len(true_index_array)):
        true_index = true_index_array[i]
        centre_index = centre_index_array[i]
        Vc = V[:, centre_index]
        Uo = U[true_index, :]
        # print("Vc = ", Vc)
        # print("Uo = ", Uo)
        # print("Uo.Vc = ", np.dot(Uo.T, Vc))
        expSum = 0.0
        for j in range(vocab_len):
            Uj = U[j, :]
            expSum += np.exp(np.dot(Uj.T, Vc))
        # print("log UjVc = ", expSum)
        loss += (-np.dot(Uo.T, Vc) + np.log(expSum))
        # print("loss {} = {}".format(i, loss))

    return loss / len(true_index_array)

#########################

def forwardPass(input_data, parameters):

    # parameters = initialize_weights(vocab_len, dim)
    Z2 = matrixMult(input_data, parameters)
    output_array = softOutput(Z2)

    return Z2, output_array

def compute_loss(Z2, truth_data):
    data_num = np.shape(truth_data)[0]
    # print("data numbers :", data_num)
    loss = 0.0
    for i in range(data_num):
        true_word = truth_data[i, :]
        UV = Z2[:, i]
        # print("UV :", UV)
        # print("true word :", true_word)
        loss += -UV[true_word.tolist().index(1)] + np.log(np.sum(np.exp(UV)))
    return loss / data_num

# def backPass(parameters, output_array, truth_data, input_data, learn_rate):
#     errors = np.sum(np.subtract(output_array, truth_data), axis=0)
#     V = parameters["W1"]
#     U = parameters["W2"]
#     centre_index_array = np.argmax(input_data, axis=1)
#     truth_index_array = np.argmax(truth_words, axis=1)
#
#     for i in range(len(centre_index_array)):
#         true_index = truth_index_array[i]
#         centre_index = centre_index_array[i]
#

# vocab_len = 4
# dim = 2
# input_data = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])
# truth_words = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
# parameters = initialize_weights(vocab_len, dim)
# print(parameters["W1"])
# print(parameters["W2"])
# loss = loss_function(input_data, parameters, truth_words)
# print(loss)

def updateParameters(parameters, input_data, output_array, truth_words, learn_rate):
    centre_index_array = np.argmax(input_data, axis=1)
    truth_index_array = np.argmax(truth_words, axis=1)
    V = parameters["W1"]
    U = parameters["W2"]
    dim = np.shape(V)[0]
    vocab_len = np.shape(V)[1]
    # print("V before is :", V)
    # print("U before is :", U)

    for i in range(len(centre_index_array)):
        true_index = truth_index_array[i]
        centre_index = centre_index_array[i]
        Uo = U[true_index, :]
        Vc = V[:, centre_index]
        dVc = np.zeros(dim)
        # print("Uo is:", Uo)
        # print("Vc is:", Vc)
        # print("dVc is :", dVc)
        # dUo = np.zeros(1, vocab_len)
        for j in range(vocab_len):
            Uj = U[j, :]
            # print("Uj is :", Uj)
            Pj = output_array[i, j]
            # print("Pj is:", Pj)
            dVc += Pj * Uj
            # print("dVc is :", dVc)
        dVc = dVc - Uo.T
        # print("dVc is :", dVc)
        dUo = Vc * (1 - output_array[i, true_index])
        # print("dUo is :", dUo)
        V[:, centre_index] -= learn_rate * dVc
        # print("V after is:", V)
        U[true_index, :] -= learn_rate * dUo
        # print("U after is:", U)

    parameters["W1"] = V
    parameters["W2"] = U
    # print("V after is :", V)

    return parameters

# vocab_len = 4
# dim = 2
# input_data = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])
# truth_words = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
# parameters = initialize_weights(vocab_len, dim)
# output_array = forwardProcess(input_data, vocab_len, dim)
# print("output array is :", output_array)
# loss = loss_function(input_data, parameters, truth_words)
# print("loss is :", loss)
# update_parameters = updateParameters(parameters, input_data, output_array, truth_words, learn_rate=0.01)

def train_function(train_data, dim, truth_words, batch_size, epoch, learn_rate):
    data_numbers = np.shape(train_data)[0]
    vocab_len = np.shape(train_data)[1]

    batch_numbers = data_numbers / batch_size

    parameters = initialize_weights(vocab_len, dim)
    # print(parameters)

    for e in range(epoch):
        for i in range(int(batch_numbers)):
            input_data = train_data[i*batch_size : (i+1)*batch_size, :]
            truth_data = truth_words[i*batch_size : (i+1)*batch_size, :]
            # print(input_data)
            output_array = forwardProcess(input_data, parameters)
            loss = loss_function(input_data, parameters, truth_data)
            print("loss in epoch {} batch {} is {}".format(e, i, loss))
            parameters = updateParameters(parameters, input_data, output_array, truth_data, learn_rate)

    return parameters

def train(train_data, dim, truth_words, batch_size, epoch, learn_rate):
    data_numbers = np.shape(train_data)[0]
    vocab_len = np.shape(train_data)[1]
    batch_numbers = data_numbers / batch_size

    parameters = initialize_weights(vocab_len, dim)

    for e in range(epoch):
        if e < 70:
            learn_rate = learn_rate
        else:
            learn_rate = learn_rate / 5
        for i in range(int(batch_numbers)):
            input_data = train_data[i * batch_size: (i + 1) * batch_size, :]
            truth_data = truth_words[i * batch_size: (i + 1) * batch_size, :]
            Z2, output_array = forwardPass(input_data, parameters)
            loss = compute_loss(Z2, truth_data)
            # print("loss in epoch {} batch {} is {}".format(e, i, loss))
            parameters = updateParameters(parameters, input_data, output_array, truth_data, learn_rate)
        print("loss in epoch {} is {}".format(e, loss))

    return parameters

