
# coding: utf-8

# In[39]:

import numpy as np
import matplotlib.pyplot as plt
from math import log
from numpy.linalg import inv
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

#
# # # Preprocessing for data
#
# # In[91]:
#
# train_file = open('/home/seungkwan/train-processed.csv')
# test_file = open('/home/seungkwan/test-processed.csv')
#
# K = 10000
#
# train_sentence = []
# test_sentence = []
# boW = {}
#
# train_y = []
# test_y = []
#
#
# # In[92]:
#
# for line in train_file:
#     label, sentence = line.split(',')
#     train_sentence.append(sentence)
#     for word in sentence.split():
#         if word in boW:
#             boW[word] += 1 if label == '1' else  -1
#         else:
#             boW[word] = 1 if label == '1' else -1
#     train_y.append(int(label))
#
# for line in test_file:
#     label, sentence = line.split(',')
#     test_sentence.append(sentence)
#     for word in sentence.split():
#         if word in boW:
#             boW[word] = boW[word] + 1 if label == '1' else boW[word] - 1
#         else:
#             boW[word] = 1 if label == '1' else -1
#     test_y.append(int(label))
#
#
# # In[93]:
#
# N = len(train_sentence)
# train_x = np.zeros((K, N))
# train_y = np.array(train_y, np.float)
#
# test_x = np.zeros((K, len(test_sentence)))
# test_y = np.array(test_y, np.float)
#
# print(train_x.shape, train_y.shape ,test_x.shape,test_y.shape)
#
#
# # In[94]:
#
# boWlist = list(boW.items())
# boWlist.sort(key=lambda a: a[1], reverse=True)
# boWlist = boWlist[:K//2] + boWlist[-(K//2):]
#
#
# # In[95]:
#
# for i in range(K):
#     boWlist[i] = (boWlist[i][0], i)
#
# word_2_index = dict(boWlist)
#
# for i, sentence in enumerate(train_sentence):
#     here = np.zeros(K)
#     for word in sentence.split():
#         if word in word_2_index:
#             here[word_2_index[word]] += 1
#     train_x[:, i] = here[:]
#
# for i, sentence in enumerate(test_sentence):
#     here = np.zeros(K)
#     for word in sentence.split():
#         if word in word_2_index:
#             here[word_2_index[word]] += 1
#     test_x[:, i] = here[:]
#
#
# # # Naive Bayes Classifier
#
# # In[96]:
#
# model = BernoulliNB()
# model.fit(train_x.T, train_y)
#
# prediction = model.predict(test_x.T)
# print(prediction.shape, test_y.shape)
# result = np.equal(prediction, test_y)
#
# print("accuracy = ",np.sum(result) / test_y.shape[0])
# np.save('accuracy%d' % K, np.sum(result) / test_y.shape[0])


# Plot Accuracy

# In[104]:

accuracy = []


K = [1000, 1500, 2000, 2500, 3000, 5000, 7000, 8000, 9000, 10000, 12000, 13500, 15000, 17500, 20000]

for k in K:
    accuracy.append(np.load('accuracy%d.npy' % k).tolist())

print(accuracy)
plt.plot(K, accuracy)
#plt.axis([0, 17500, 0.76, 0.79])
plt.xlabel('K')
plt.ylabel('accuracy')
plt.show()


# In[ ]:



