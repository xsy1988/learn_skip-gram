# Create by MrZhang on 2019-11-22

import numpy as np
import re

# 对获取到对原始文本进行预处理：
# 去除标点符号，去除 's 之类对影响，然后对文本进行切分，使输出矩阵对每一行数据仅包含一个句子。
# 对每一行句子，去掉首尾对空格符号
# 保存处理后获得的矩阵
def preproData(data_path, save_path):
    fr = open(data_path, 'r')
    sentData = []
    for line in fr:
        line = line.strip('\n')
        line = re.sub(r'[{}]+'.format('“”!,;:?"-'), '', line)
        line = re.sub(r'\d+', '', line)
        line = re.sub(r'[{}]+'.format('\'s'), '', line)
        sentences = line.split('.')
        sentList = [sentence for sentence in sentences if sentence != '']
        for sentence in sentList:
            sentData.append(sentence.strip(' '))
    sentData = np.array(sentData)
    np.save(save_path, sentData)

def createVocabList(data_path):
    dataSet = np.load(data_path)
    vocabSet = set([])
    for sentence in dataSet:
        vocabSet = vocabSet | set(re.split(r'\W+', str.lower(sentence)))
    vocab_list = list(vocabSet)
    vocab_list.remove('')
    vocab_list.sort()
    return vocab_list

# 将一个单词转化为独热向量。
def wordToOneHotVec(word, vocab_list):
    word_vec = np.zeros(len(vocab_list))
    index = vocab_list.index(word)
    word_vec[index] = 1
    return word_vec

# 输入一个句子，根据窗口的大小，生成多组"中心词 -> 关联词"的词对
def generateWordCouple(input_sentence, window_size):
    wordList = re.split(r'\W+', str.lower(input_sentence))
    sentence_len = len(wordList)
    centreWord = []
    contextWord = []
    for i in range(sentence_len):
        for j in range(window_size):
            if i + j + 1 < sentence_len:
                centreWord.append(wordList[i])
                contextWord.append(wordList[i + j + 1])
            if i - j - 1 >= 0:
                centreWord.append(wordList[i])
                contextWord.append(wordList[i - j - 1])
    return centreWord, contextWord

# 将词列表转化为独热向量矩阵
def listToOneHotMat(word_list, vocab_list):
    words_one_hot_mat = []
    for word in word_list:
        if word == '': continue
        words_one_hot_mat.append(wordToOneHotVec(word, vocab_list))
    return words_one_hot_mat

# 对于已经预处理过的训练数据，生成若干对"中心词 -> 关联词"
def createTrainData(train_data_path, window_size, vocab_list):
    train_data = np.load(train_data_path)
    sentNum = np.shape(train_data)[0]
    centreWordList = []
    contextWordList = []
    for i in range(sentNum):
        sentence = train_data[i]
        if len(sentence) < 3: continue
        centreWord, contextWord = generateWordCouple(sentence, window_size)
        centreWordList += centreWord
        contextWordList += contextWord
    centreWordMat = listToOneHotMat(centreWordList, vocab_list)
    contextWordMat = listToOneHotMat(contextWordList, vocab_list)
    return centreWordMat, contextWordMat


