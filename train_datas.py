# Create by MrZhang on 2019-11-26

import toolPre
import computeModel
import numpy as np

# sentData = np.load('datas/sentData.npy')
# print(sentData)

data_path = 'datas/testPaper.txt'
save_path = 'datas/train_sentence_data.npy'
toolPre.preproData(data_path, save_path)
vocab_list = toolPre.createVocabList(save_path)
# centre_words, context_words = toolPre.createTrainData(save_path, 2, vocab_list)
# # print(centre_words)
# # print(context_words)
# np.save('datas/centre_words.npy', centre_words)
# np.save('datas/context_words.npy', context_words)
list_path = 'datas/vocab_list.npy'
# file = open(list_path, 'w')
# for word in vocab_list:
#     file.write(str(word))
#     file.write('\n')
# file.close()
vocab_array = np.array(vocab_list)
np.save(list_path, vocab_array)
