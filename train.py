# Create by MrZhang on 2019-11-26

import computeModel
import numpy as np

centre_words = np.load('datas/centre_words.npy')
context_words = np.load('datas/context_words.npy')

dim = 100
batch_size = 32
learn_rate = 0.01
epoch = 200

# parameters = computeModel.train_function(centre_words, dim, context_words, batch_size, epoch, learn_rate)
# print(parameters["W1"])

parameters = computeModel.train(centre_words, dim, context_words, batch_size, epoch, learn_rate)
print(parameters["W1"])
np.save('datas/embedding.npy', parameters["W1"])
