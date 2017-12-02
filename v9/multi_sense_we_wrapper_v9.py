import gensim
from sklearn.externals import joblib
import logging
from adj_clustering_v7 import K_CLUSTERS

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


    def get_adj_label(self, adj,context):
        file_name = self.adj_clusters_folder +'/'+ adj +'.pkl'
        kmeans=joblib.load(file_name)
        cluster_labels = kmeans.predict(self.org_model.word_vec(context).reshape(1,-1))
        label = cluster_labels[0]
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

    def word_vec(self,word):
        if word.find("<") > -1:
            print "word = {}".format(word)
        # if word in self.model.vocab:
        #     print "[{}] in self.model.vocab".format(word)
        # if word in self.vocab:
        #     print "[{}] in self.vocab".format(word)

        # print "self.model.word_vec([{}]) = [{}]".format(word,self.model.word_vec(word)[0:5])

        return self.model.word_vec(word)


    #get all the representations of specified adjective
    def all_adj_vecs(self,adj):
        file_name = self.adj_clusters_folder +'/'+ adj +'.pkl'
        if adj in self.multi_sense_adj:
            formatted_adj=["{}_<{}>".format(adj,i) for i in xrange(0,K_CLUSTERS)]
            word_vecs = [self.word_vec(adj) for adj in formatted_adj]

        else:
            word_vec = self.word_vec(adj)
            word_vecs = [word_vec]*K_CLUSTERS
        return word_vecs
