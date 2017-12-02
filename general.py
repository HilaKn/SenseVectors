# import multiprocessing
# for i in xrange(0,200):
#     j = i*2
#     print j

# my_str = "hello whitespace  tab and another         3 tabs"
# my_list = my_str.split()
# print my_list
#
# dict = {1:2, 2:6, 3:7}
# print len(dict)
#
# with open("demo", 'w') as f:
#     for i in xrange(0,10):
#         f.write(str(i)+'\n')
# class Person:
#     def __init__(self,first,last):
#         self.first_name = first
#         self.last_name = last
#
# p1 = Person("Hila","Kneller")
# p2 = Person("Dub", "Dubon")
# p3 = Person("Tal","Taltal")
# dic_1 = {0:p1,1:p2,2:p3}
# dic_2 = {0:p2,1:p1,2:p3}
#
# p2.last_name = "new_new"
# dic_1[0].first_name = "Pila"
# print dic_1
# print dic_2

#
# with open("/home/h/data/HeiPLAS-release/HeiPLAS-dev.txt") as f:
#     input_list = [line.split() for line in f.readlines()]
#     HeiPLAS_dev_adectives = set([item[1] for item in input_list])
#
#
# with open("/home/h/data/HeiPLAS-release/HeiPLAS-test.txt") as f:
#     input_list = [line.split() for line in f.readlines()]
#     HeiPLAS_test_adectives = set([item[1] for item in input_list])
#
# HeiPLAS_dev_adectives.update(HeiPLAS_test_adectives)
# with open("HeiPLAS_adj_list",'w') as f:
#     f.write("\n".join(HeiPLAS_dev_adectives))
#
# with open("/home/h/Documents/Hila/Research/code/SenseVectors/adj_dic") as f:
#     f.readline()
#     input_list = [line.split() for line in f.readlines() ]

# min_occur_for_clustering = [500,600,700,800,900,1000]#[3,5,10,20,30,40,50,100,120,140,150,160,180,200,220,250,300,350,400]
#
# for x in min_occur_for_clustering:
#     print "Min occurrences for clustering = {}".format(x)
#     print "-----------------------------------"
#     input_list = [item for item in input_list if len(item)>0]
#     multi_sense_adjectives = [item[0] for item in input_list if int(item[1])  > x]
#
#     multi_sense_from_dev = len([item for item in HeiPLAS_dev_adectives if item in multi_sense_adjectives])
#     multi_sense_from_test = len([item for item in HeiPLAS_test_adectives if item in multi_sense_adjectives])
#
#     print "HeiPLAS dev:\nTotal adjectives = {}\nHas multi senses: {}".format(len(HeiPLAS_dev_adectives),multi_sense_from_dev)
#     print "HeiPLAS test:\nTotal adjectives = {}\nHas multi senses: {}".format(len(HeiPLAS_test_adectives),multi_sense_from_test)


#
# with open ("output_v7/wiki_update_with_labels_v7","r") as f:
#
#     counter = 0
#     for line in f:
#
#         if line.find("<1>") > -1 or line.find("<2>") > -1 or line.find("<0>") > -1:
#             print line
#         counter += 1

#
# import numpy as np
# a = np.arange(60.).reshape(3,4,5)
# print a
# b = np.arange(24.).reshape(4,3,2)
# print b

# a = ('1','2','3')
# str = "\t".join(a)
# print str
import gensim
import numpy as np
from operator import itemgetter

model = gensim.models.KeyedVectors.load("/home/h/data/word2vec/word2vec_text", mmap='r')  # mmap the large matrix as read-only
# model = gensim.models.KeyedVectors.load("models/from_parsed_wiki/models/normed_we_300_5", mmap='r') .wv # mmap the large matrix as read-only
model.syn0norm = model.syn0

word_1= 'high'
word_2 = 'degree'
word_3 = 'low'
#
# print model.similarity(word_1,word_2)
# print model.similarity(word_3,word_2)
# print model.n_similarity([word_1,word_3],[word_2])
# print model.n_similarity([word_1,word_3, 'intense','bad','mild'],[word_2])
# print model.n_similarity([word_1,word_3, 'intense','bad','mild'],['height'])
# print model.most_similar(positive=[word_1,word_3],negative=[word_2])
# print model.most_similar(positive=['hot','temperature'],negative=['cold'])
# print model.most_similar(positive=['hot','temperature'],negative=['big'])
# print model.similarity(word_1,word_2)
# print model.similarity(word_2,word_3)
#
# print model.n_similarity([word_1,word_3],[word_2])

# words = ['hot', 'large', 'beautiful','smart']
adjectives =['high','low', 'intense','bad','mild']# ['regular','standard']
nouns = ['pain', 'storm', 'point', 'opinion','risk','headache','fever', 'desire','anxiety','emotion','temperature','heat','hope']
# nouns = ['pain', 'point','risk','headache','fever']

print model.n_similarity(nouns,['degree'])

print '+++++++++++++++'
from adj_noun_attr import AdjNounAttribute
with open("/home/h/data/HeiPLAS-release/HeiPLAS-dev.txt") as f:
    input_list = [line.split() for line in f.readlines()]
data_set = [AdjNounAttribute(item[1],item[2],item[0].lower()) for item in input_list]


data_set = [samp for samp in data_set if samp.adj in model.vocab
            and samp.noun in model.vocab and samp.attr in model.vocab and samp.attr not in ["good"]]
print "Total samples = [{}]".format(len(data_set))

    # 2. Load all attribute names
attributes = [samp.attr for samp in data_set]
unique_attr = set(attributes)
print "Number of attributes: [{}]".format(len(unique_attr))
attributes= list(unique_attr)

print adjectives
attr_score = [(attr,model.n_similarity(adjectives,[attr])) for attr in attributes]
sorted_data = sorted(attr_score,key=itemgetter(1),reverse=True)
print sorted_data[:10]

print nouns
attr_nouns_score = [(attr,model.n_similarity(nouns,[attr])) for attr in attributes]
sorted_noun_data = sorted(attr_nouns_score,key=itemgetter(1),reverse=True)
print sorted_noun_data[:10]
print "---------"
# for word in words:
#     print word
#     most_sim = model.most_similar(word,topn=5)
#     most_sim_words = [sim_word[0] for sim_word in most_sim]
#
#     attr_score = [(attr,model.n_similarity(most_sim_words,[attr])) for attr in attributes]
#     sorted_data = sorted(attr_score,key=itemgetter(1),reverse=True)
#
#     print most_sim_words
#     print sorted_data[:10]
# print model.n_similarity(['young','old','newborn','brand-new'],['recent'])

import numpy as np
# score = np.dot(model.word_vec(word_1,use_norm=False),model.word_vec(word_2,use_norm=False))
# print score



# import gensim
# model = gensim.models.KeyedVectors.load("models/from_parsed_wiki/models/normed_we_300_5", mmap='r') .wv # mmap the large matrix as read-only
# model.syn0norm = model.syn0
# from nltk.stem.porter import  PorterStemmer
# st = PorterStemmer()
# word_1= st.stem("intrusiveness")
# word_2 = st.stem("accurate")
# print word_1
# print word_2
# print model.similarity(word_1,word_2)

# if word in model.vocab:
#     print "in"
# else:
#     print "no"

