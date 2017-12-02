import cPickle
import os
import argparse
import gensim
import gzip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import math
from collections import defaultdict
import pickle
from save_adj_dict_v7 import AdjContext, ADJ_DIC_FOLDER
import save_adj_dict_v7 as s
import json
import jsonpickle
from itertools import groupby
import itertools
import io

TOKEN_ID_COLUMN = 0
TOKEN_COLUMN = 1
POS_COLUMN = 3
DEP_ID_COLUMN = 6
DEP_RELATION_COLUMN = 7

ADJ_TAG = 'JJ'
NOUN_TAGS = ['NN','NNS','NNP']#Todo: consider adding proper nouns
K_CLUSTERS = 3
MIN_ITEMS_FOR_CLUSTERING = 1000
NON_RELEVANT_HEAD = ['.',',']
MIN_ADJ_NOUN_OCCURRENCE = 1
adj_clusters_folder = "adj_clusters"
output_folder = "output_v7"
adj_analysis_folder = "adj_analysis"


def output_clusters():
    global label_to_contexts, cluster_id, i, label, contexts_ids, j, contexts_words, f
    # print clusters for manual analysis
    label_to_contexts = defaultdict(list)
    for cluster_id in xrange(0, K_CLUSTERS):
        contexts_ids = [i for i, label in enumerate(kmeans.labels_) if cluster_id == label]
        contexts_words = set([filtered_contexts_list[j].head_noun for j in contexts_ids])
        label_to_contexts[cluster_id] = contexts_words
    with open("{}/{}/{}/{}".format(output_folder, adj_clusters_folder, adj_analysis_folder, adj), 'w') as f:
        for label, contexts_words in label_to_contexts.iteritems():
            f.write("label\t{}\n".format(label))
            f.write("\n".join([(str(label) + "\t" +word) for word in contexts_words]))
            f.write("\n")
    label_to_contexts.clear()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('sentences_input_file',help='input file path - sentences format')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',help='the genrated file for WE training after adjectives clustering and labeling file path')
    parser.add_argument('--analyze_adj_clusters','-a',default=False,action='store_true', help='output contexts clusters for later analysis')
    args = parser.parse_args()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder+'/'+ adj_clusters_folder):
        os.makedirs(output_folder+'/'+adj_clusters_folder)

    if not os.path.exists(output_folder+'/'+ adj_clusters_folder+'/'+adj_analysis_folder):
        os.makedirs(output_folder+'/'+adj_clusters_folder+'/'+adj_analysis_folder)

    #load the trained word vectors
    print "Loading word vectors from {}".format(args.word_embeddings_file)
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    # with open(s.ADJ_DIC_FILE, 'rb') as f1 , open(s.SENT_TO_ADJ_CONTEXT_FILE,'rb') as f2:
    #     adj_dic_json= json.load(f1)
    #     adj_dic = jsonpickle.decode(adj_dic_json)
        # sentences_dic_json = json.load(f2)
        # sentence_to_adj_context = jsonpickle.decode(sentences_dic_json)

    with open("HeiPLAS_adj_list", 'r') as f:
        HeiPLAS_adj  = dict.fromkeys(f.read().splitlines())

    #remove adjective that occur less than MIN_ITEMS_FOR_CLUSTERING times
    # print "Filter out adjectives with less than {} relevant contexts".format(MIN_ITEMS_FOR_CLUSTERING)
    # filtered_adj_dic = {adj: contexts_list for adj, contexts_list in adj_dic.iteritems()
    #                     if len(contexts_list) >MIN_ITEMS_FOR_CLUSTERING
    #                     and adj in HeiPLAS_adj}

    sent_to_labeled_adj = defaultdict(list)

    # print "Start clustering {} adjectives".format(len(filtered_adj_dic))
    # print "Write to file the the filtered adjectives"
    #cluster each adjective to senses
    for filename in os.listdir(ADJ_DIC_FOLDER):
        with open(ADJ_DIC_FOLDER + '/' + filename) as f:
            adj = filename
            adj_contexts_json= json.load(f)
            adj_contexts= jsonpickle.decode(adj_contexts_json)
        filtered_contexts = [context for context in adj_contexts
                             if len(adj_contexts) >MIN_ITEMS_FOR_CLUSTERING and adj in HeiPLAS_adj]


        sorted_contexts_list = sorted(filtered_contexts, key=lambda x: x.head_noun, reverse=True)
        grouped_contexts_list = [list(grouped_contexts) for head_noun, grouped_contexts in
                                      groupby(sorted_contexts_list, lambda x: x.head_noun)]
        filtered_contexts_list = list(itertools.chain.from_iterable([contexts for contexts in grouped_contexts_list
                                                                         if len(contexts)>MIN_ADJ_NOUN_OCCURRENCE]))
        if len(filtered_contexts_list) < 1:
            continue
        print "Clustering adj: [{}] with [{}] contexts.".format(adj, len(filtered_contexts_list))

        clustering_input = np.array([model.word_vec(context.head_noun) for context  in filtered_contexts_list])
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0).fit(clustering_input)

        if (args.analyze_adj_clusters):
            output_clusters()

        for i in xrange(0,len(filtered_contexts_list)):
            context = filtered_contexts_list[i]
            context.update_label( kmeans.labels_[i])
            sent_to_labeled_adj[context.sentence_id].append(context)

        joblib.dump(kmeans, '{}/{}/{}.pkl'.format(output_folder,adj_clusters_folder,adj))

    print "Finished clustering all adjectives"

    #update corpus from original sentences file
    print "Start updating corpus with new adjectives labels"
    with io.open(args.sentences_input_file, 'r',encoding = 'utf8') as fi, io.open(output_folder+"/"+args.sentences_output_file,'w',encoding = 'utf8')as fo:
        sentence = []
        sentence_id = 0

        for line in fi:
            if sent_to_labeled_adj.has_key(sentence_id):
                line_data = line.split()
                for context in sent_to_labeled_adj[sentence_id]:
                    line_data[context.token_id-1] = context.label
                output = ' '.join(line_data)+'\n'
                fo.write(output)

            else:
                fo.write(line)

            sentence_id += 1
            if (sentence_id % 100000 == 0):
                print "update corpus:  sentence {}".format(sentence_id)
                # break
    print "Finished generating new sentences file"