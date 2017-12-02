import cPickle
import os
import argparse
import gensim
import gzip
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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
import sys
import operator
#DBSCAN + Agglomerative
#single vector per noun (wihtout multiple counting)
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
MIN_ADJ_NOUN_OCCURRENCE = 10
adj_clusters_folder = "adj_clusters"
output_folder = "output_v18"
adj_analysis_folder = "adj_analysis"
adj_first_clustering_folder = "first_clustering"
#
#
# def output_clusters():
#     global label_to_contexts, cluster_id, i, label, contexts_ids, j, contexts_words, f
#     # print clusters for manual analysis
#     label_to_contexts = defaultdict(list)
#     for cluster_id in xrange(0, K_CLUSTERS):
#         contexts_ids = [i for i, label in enumerate(clustering_alg.labels_) if cluster_id == label]
#         contexts_words = set([filtered_contexts_list[j].head_noun for j in contexts_ids])
#         label_to_contexts[cluster_id] = contexts_words
#     with open("{}/{}/{}/{}".format(output_folder, adj_clusters_folder, adj_analysis_folder, adj), 'w') as f:
#         for label, contexts_words in label_to_contexts.iteritems():
#             f.write("label\t{}\n".format(label))
#             f.write("\n".join([(str(label) + "\t" +word) for word in contexts_words]))
#             f.write("\n")
#     label_to_contexts.clear()

def output_clusters(k,final_labeling):
    global label_to_contexts, cluster_id,  label, contexts_ids, contexts_words, f
    # print clusters for manual analysis
    label_to_contexts = defaultdict(list)
    for i, label in enumerate(final_labeling):
        head_noun = filtered_contexts_list[i].head_noun
        label_to_contexts[label].append(head_noun)


    with open("{}/{}/{}/{}".format(output_folder, adj_clusters_folder, adj_analysis_folder, "{}_{}".format(adj,k)), 'w') as f:
        for label, contexts_words in label_to_contexts.iteritems():
            f.write("label\t{}\n".format(label))
            unique_contexts_words = set(contexts_words)
            f.write("\n".join([(str(label) + "\t" +word) for word in unique_contexts_words]))
            f.write("\n")
    label_to_contexts.clear()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('sentences_input_file',help='input file path - sentences format')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',help='the generated file for WE training after adjectives clustering and labeling file path')
    parser.add_argument('--analyze_adj_clusters','-a',default=False,action='store_true', help='output contexts clusters for later analysis')
    args = parser.parse_args()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder+'/'+ adj_clusters_folder):
        os.makedirs(output_folder+'/'+adj_clusters_folder)

    if not os.path.exists(output_folder+'/'+ adj_clusters_folder+'/'+adj_analysis_folder):
        os.makedirs(output_folder+'/'+adj_clusters_folder+'/'+adj_analysis_folder)

    if not os.path.exists(output_folder+'/'+adj_first_clustering_folder):
        os.makedirs(output_folder+'/'+adj_first_clustering_folder)


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
    for filename in ['blue','hot','long','right']:#os.listdir(ADJ_DIC_FOLDER):
        with open(ADJ_DIC_FOLDER + '/' + filename) as f, open("adj_num_of_occurrence", 'a') as wf:
            adj = filename
            adj_contexts_json= json.load(f)
            adj_contexts= jsonpickle.decode(adj_contexts_json)
            wf.write(adj + '\t' + str(len(adj_contexts)) + '\n')
        filtered_contexts = [context for context in adj_contexts
                             if len(adj_contexts) >MIN_ITEMS_FOR_CLUSTERING and adj in HeiPLAS_adj]


        sorted_contexts_list = sorted(filtered_contexts, key=lambda x: x.head_noun, reverse=True)
        grouped_contexts_list = [list(grouped_contexts) for head_noun, grouped_contexts in
                                      groupby(sorted_contexts_list, lambda x: x.head_noun)]
        # filtered_contexts_list = list(itertools.chain.from_iterable([contexts for contexts in grouped_contexts_list
        #                                                                  if len(contexts)>MIN_ADJ_NOUN_OCCURRENCE]))

        filtered_contexts_list = []
        unique_contexts = []
        data_mapper = {}
        index = 0
        for i,contexts in enumerate(grouped_contexts_list):
            if len(contexts)>MIN_ADJ_NOUN_OCCURRENCE:
                for j in xrange(len(filtered_contexts_list),len(filtered_contexts_list)+len(contexts)):
                    data_mapper[j] = index
                filtered_contexts_list.extend(contexts)
                unique_contexts.append(contexts[0])
                index +=1

        print "max i = [{}]".format(i)
        print "data mapper contains [{}] items".format(len(data_mapper))

        if len(unique_contexts) < 1:
            continue
        print "Clustering adj: [{}] with [{}] contexts.".format(adj, len(filtered_contexts_list))

        clustering_input = np.array([model.word_vec(context.head_noun) for context  in unique_contexts])
        print "input is ready"
        print clustering_input.shape

        try:

            print "DBSCAN clustering [{}] for".format(adj)
            first_clustering = DBSCAN(eps = 0.4,min_samples=5, metric='cosine', algorithm='brute',  n_jobs=-1).fit(clustering_input)
            first_clustering_k = len(list(set(first_clustering.labels_)))
            print "done clustering [{}] with [{}] clusters".format(adj,first_clustering_k)

            label_to_contexts_vecs = defaultdict(list)
            outliers = []
            for i, label in enumerate(first_clustering.labels_):
                if label != -1:
                    context_vec = clustering_input[[i],:]
                    label_to_contexts_vecs[label].append(context_vec)
                else:
                    outliers.append(i)

            label_to_matrix = {label:np.array(context_vecs).squeeze() for label,context_vecs in label_to_contexts_vecs.iteritems()}
            print "label 0 matrix shape = {}".format(label_to_matrix[0].shape)
            label_to_avg = {label:np.average(matrix,axis=0) for label,matrix in label_to_matrix.iteritems()}
            print "label 0 avg vector shape= {}".format(label_to_avg[0].shape)
            sorted_labels = sorted(label_to_avg.items(), key=operator.itemgetter(0))
            print "clustering outlier shape = {}".format(clustering_input[[outliers[0]],:].squeeze().shape)
            sorted_labels.extend([(-1,clustering_input[[outlier_idx],:].squeeze()) for outlier_idx in outliers])
            agglomerative_input = np.array([item[1] for item in sorted_labels])
            print "agglomerative input shape = {}".format(agglomerative_input.shape)
            k= first_clustering_k-1
            print "Agglomerative clustering [{}] for [{}] clusters".format(adj,k)
            clustering_alg = AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='complete').fit(agglomerative_input)
            print "done clustering [{}] for [{}] clusters".format(adj,k)

            unique_labeling = []
            outlier_counter = 0
            for i in xrange(0,len(unique_contexts)):
                if first_clustering.labels_[i] != -1:
                    unique_labeling.insert(i,clustering_alg.labels_[first_clustering.labels_[i]])
                else:
                    label =clustering_alg.labels_[k+outlier_counter]
                    unique_labeling.insert(i,label)
                    outlier_counter += 1

            print len(unique_labeling)
            print "after unique_labeling."
            final_labeling = [unique_labeling[data_mapper[i]] for i in xrange(0,len(filtered_contexts_list))]
            if (args.analyze_adj_clusters):
                print "start file writing"
                output_clusters(k,final_labeling)
                print "done file writing"
            print "done clustering"
        except :
            print "Failed to cluster adjective: [{}]".format(adj)
            print sys.exc_info()
            continue
        finally:
            print "Finally"
        #
        # if (args.analyze_adj_clusters):
        #     print "start file writing"
        #     output_clusters()
        #     print "done file writing"

    #     print "before sent_to_labeled_adj"
    #     for i in xrange(0,len(filtered_contexts_list)):
    #         context = filtered_contexts_list[i]
    #         context.update_label( final_labeling[i])
    #         sent_to_labeled_adj[context.sentence_id].append(context)
    #
    #     print "after sent_to_labeled_adj"
    #     joblib.dump(clustering_alg, '{}/{}/{}.pkl'.format(output_folder,adj_clusters_folder,adj))
    #     joblib.dump(clustering_alg, '{}/{}/{}.pkl'.format(output_folder,adj_first_clustering_folder,adj))
    #     print "finished joblib dump"
    # print "Finished clustering all adjectives"

    #update corpus from original sentences file
    # print "Start updating corpus with new adjectives labels"
    # with io.open(args.sentences_input_file, 'r',encoding = 'utf8') as fi, io.open(output_folder+"/"+args.sentences_output_file,'w',encoding = 'utf8')as fo:
    #     sentence = []
    #     sentence_id = 0
    #
    #     for line in fi:
    #         if sent_to_labeled_adj.has_key(sentence_id):
    #             line_data = line.split()
    #             for context in sent_to_labeled_adj[sentence_id]:
    #                 line_data[context.token_id-1] = context.label
    #             output = ' '.join(line_data)+'\n'
    #             fo.write(output)
    #
    #         else:
    #             fo.write(line)
    #
    #         sentence_id += 1
    #         if (sentence_id % 100000 == 0):
    #             print "update corpus:  sentence {}".format(sentence_id)
    #             # break
    # print "Finished generating new sentences file"