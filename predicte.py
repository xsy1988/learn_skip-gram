# Create by MrZhang on 2019-11-27

import numpy as np
import operator

def cosine_similarly(colA, colB):
    similarly_course = np.dot(colA.T, colB) / (np.linalg.norm(colA) * np.linalg.norm(colB))
    return similarly_course

def get_vocab(vocab_path):
    vocab_array = np.load(vocab_path)
    vocab_list = vocab_array.tolist()
    return vocab_list

def get_embedding_mat(embedding_path):
    return np.load(embedding_path)

def embedding(word, embedding_mat, vocab_list):
    index = vocab_list.index(word)
    return embedding_mat[:, index]

def word_sim_course(word1, word2, vocab_list, emb_mat):
    # vocab_list = get_vocab(vocab_path)
    # emb_mat = get_embedding_mat(emb_path)
    word1_emb = embedding(word1, emb_mat, vocab_list)
    word2_emb = embedding(word2, emb_mat, vocab_list)
    course = cosine_similarly(word1_emb, word2_emb)
    return course

def words_sim_course(target_word, vocab_path, emb_path):
    course_dic = {}
    vocab_list = get_vocab(vocab_path)
    emb_mat = get_embedding_mat(emb_path)
    for word in vocab_list:
        if word == target_word: continue
        course = word_sim_course(target_word, word, vocab_list, emb_mat)
        if course < 0.2: continue
        course_dic[word] = course
    # course_dic = sorted(course_dic.values(), reverse=True)
    # course_dic = sorted(course_dic.items(), key=operator.itemgetter(1), reverse=True)
    return course_dic

def most_sim_words(course_dic, count):
    course_dic = sorted(course_dic.items(), key=operator.itemgetter(1), reverse=True)
    most_sim = {}
    i = 1
    for key, value in enumerate(course_dic):
        if i > count:
            break
        else:
            most_sim[key] = value
            i += 1
    return most_sim

list_path = 'datas/vocab_list.npy'

embedding_path = 'datas/embedding.npy'
word = 'window'
course_dic = words_sim_course(word, list_path, embedding_path)
# print(course_dic)
most_5_sim = most_sim_words(course_dic, 5)
print(most_5_sim)
print(most_5_sim[0])
