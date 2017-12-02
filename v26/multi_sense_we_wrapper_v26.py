import gensim
from sklearn.externals import joblib
import logging
from adj_clustering_v7 import K_CLUSTERS
import numpy as np

class MultiSenseWE:

    def __init__(self,before_clustering_we_file,normed_we_file, multi_sense_ad_file, adj_clusters_folder):
        self.we_file = normed_we_file
        self.pre_cluster_we_file = before_clustering_we_file
        self.multi_senses_adj_file = multi_sense_ad_file
        self.adj_clusters_folder = adj_clusters_folder
        self.multi_sense_adj={} #all the adjectives that have multi sense representation
        self.vocab = {}#should hold all the words that have vectors including original adjectives with multi sense vectors(e.g. "high" for" high_<1>")

    def set_model(self):
         # load pre-trained, before clustering normalized word2ec
        self.org_model = gensim.models.KeyedVectors.load(self.pre_cluster_we_file, mmap='r').wv  # mmap the large matrix as read-only
        self.org_model.syn0norm = self.org_model.syn0

        # load pre-trained, normalized word2ec
        self.model = gensim.models.KeyedVectors.load(self.we_file, mmap='r').wv  # mmap the large matrix as read-only
        self.model.syn0norm = self.model.syn0

        #load adjectives list with multi-sense representation
        with open(self.multi_senses_adj_file,'r') as f:
            senses_adj_list = f.read().splitlines()
            self.multi_sense_adj = dict.fromkeys(senses_adj_list)
        print "Total multi adj = {}".format(len(self.multi_sense_adj))

        #generate list of all the words with word vectors
        self.vocab = dict(self.model.vocab , **self.multi_sense_adj)
        print "VOCAB SIZE =[{}]".format(len(self.vocab))


#TODO: (27/9) debug this!!!!!
    def predict_label(self,word, clusters_matrix):
        word_vec = self.org_model.word_vec(word)
        cosine_sim = np.dot(word_vec,clusters_matrix.T)
        label = np.argmax(cosine_sim)
        return label

    def get_adj_label(self, adj,context):
        file_name = self.adj_clusters_folder +'/'+ adj
        clusters_matrix = np.loadtxt(file_name)
        label = self.predict_label(context,clusters_matrix)
        # kmeans=joblib.load(file_name)
        # cluster_labels = kmeans.predict(self.org_model.word_vec(context).reshape(1,-1))
        # label = cluster_labels[0]
        new_adj = "{}_<{}>".format(adj,label)
        # print "new_adj = {}".format(new_adj)
        return new_adj


    def adj_vec_by_context(self,adj,context):
        # print "adj_by_context: adj = [{}]. noun=[{}]".format(adj,context)
        adj_label = adj
        if adj in self.multi_sense_adj:
            adj_label = self.get_adj_label(adj,context)
            # print "adj_label = {}".format(adj_label)

        return self.word_vec(adj_label)


    def get_adj_name(self,adj,context):
        adj_label = adj
        if adj in self.multi_sense_adj:
            adj_label = self.get_adj_label(adj,context)
            # print "adj_label = {}".format(adj_label)

        return adj_label

    def word_vec(self,word):
        # if word.find("<") > -1:
        #     print "word = {}".format(word)
        # if word in self.model.vocab:
        #     print "[{}] in self.model.vocab".format(word)
        # if word in self.vocab:
        #     print "[{}] in self.vocab".format(word)

        # print "self.model.word_vec([{}]) = [{}]".format(word,self.model.word_vec(word)[0:5])

        return self.model.word_vec(word)


    # get all the representations of specified adjective
    def all_adj_vecs(self,adj):
        word_vecs = []
        if adj in self.multi_sense_adj:
            file_name = self.adj_clusters_folder +'/'+ adj
            clusters_matrix = np.loadtxt(file_name)
            k = clusters_matrix.shape[0]
            formatted_adj=["{}_<{}>".format(adj,i) for i in xrange(0,k)]
            word_vecs = [self.word_vec(adj) for adj in formatted_adj]

        word_vecs.append(self.word_vec(adj))#in any case add the general adjective vector
        return word_vecs
