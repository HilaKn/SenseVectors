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
from nltk.stem import WordNetLemmatizer

# multiple DBSCAN runnings + label outliers
#single vector per noun (wihtout multiple counting)
#apply lemmatizer on the nouns (e.g. consider 'car' and 'cars' as the same noun)
#run for all HeiPLAS adjectives
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
output_folder = "output_v24"
adj_analysis_folder = "adj_analysis"
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
    adj_files = [file for file in os.listdir(ADJ_DIC_FOLDER)]
    adj_to_cluster = [adj for adj in HeiPLAS_adj if adj in adj_files]
    # print "Start clustering {} adjectives".format(len(filtered_adj_dic))
    # print "Write to file the the filtered adjectives"
    #cluster each adjective to senses
    lemmatizer = WordNetLemmatizer()
    for filename in adj_to_cluster:#for filename in ['blue','hot','long','right','cold','clean','dark','light']:#os.listdir(ADJ_DIC_FOLDER):
        with open(ADJ_DIC_FOLDER + '/' + filename) as f, open("adj_num_of_occurrence", 'a') as wf:
            adj = filename
            adj_contexts_json= json.load(f)
            adj_contexts= jsonpickle.decode(adj_contexts_json)
            wf.write(adj + '\t' + str(len(adj_contexts)) + '\n')

        for context in adj_contexts:
            head_noun_lemma = lemmatizer.lemmatize(context.head_noun)
            context.head_noun = head_noun_lemma

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


        print "data mapper contains [{}] items".format(len(data_mapper))

        if len(unique_contexts) < 1:
            continue
        print "Clustering adj: [{}] with [{}] contexts. and [{}] unique contexts".format(adj, len(filtered_contexts_list),len(unique_contexts))

        clustering_input = np.array([model.word_vec(context.head_noun) for context  in unique_contexts])
        print "input is ready"
        print clustering_input.shape

        try:

            print "DBSCAN clustering [{}] for".format(adj)
            clustering_alg = DBSCAN(eps = 0.4,min_samples=5, metric='cosine', algorithm='brute',  n_jobs=10).fit(clustering_input)
            k_1 = len(set(clustering_alg.labels_))
            print "done clustering [{}] with [{}] clusters".format(adj,k_1)
            outlier_idx = [idx for idx,label in enumerate(clustering_alg.labels_) if label == -1]#save all indexes of outlier samples
            outlier_input = np.array([clustering_input[i] for i in outlier_idx])
            print "DBSCAN clustering [{}] outliers".format(len(outlier_input))
            clustering_alg_2 = DBSCAN(eps = 0.5,min_samples=5, metric='cosine', algorithm='brute',  n_jobs=10).fit(outlier_input)
            k_2 = len(set(clustering_alg_2.labels_))
            outlier_idx_2= [idx for idx,label in enumerate(clustering_alg_2.labels_) if label == -1]#save all indexes of outlier samples
            outlier_vecs_2 = [outlier_input[i] for i in outlier_idx_2]
            print "done clustering [{}] with [{}] clusters".format(adj,k_2)
            label_id_gap = k_1-(1 if -1 in clustering_alg.labels_ else 0)

            label_to_contexts_vecs = defaultdict(list)
            outliers = []
            for i, label in enumerate(clustering_alg.labels_):
                if label != -1:
                    context_vec = clustering_input[[i],:]
                    label_to_contexts_vecs[label].append(context_vec)

            for i, label in enumerate(clustering_alg_2.labels_):
                if label != -1:
                    context_vec = outlier_input[[i],:]
                    label_to_contexts_vecs[label+label_id_gap].append(context_vec)

            import operator
            print "Generate label to avg vector dictionary"
            label_to_matrix = {label:np.array(context_vecs).squeeze() for label,context_vecs in label_to_contexts_vecs.iteritems()}
            label_to_avg = {label:np.average(matrix,axis=0) for label,matrix in label_to_matrix.iteritems()}
            sorted_labels = sorted(label_to_avg.items(), key=operator.itemgetter(0))
            sorted_labels_avg = np.array([item[1] for item in sorted_labels])
            print "Done generate label to avg vector dictionary"

            clustering_labels = clustering_alg.labels_
            for i,org_i in enumerate(outlier_idx):
                if clustering_alg_2.labels_[i] != -1:
                    clustering_labels[org_i]=clustering_alg_2.labels_[i] +label_id_gap
                else:
                    #TODO: find best cluster for outlier
                    print "try to find the best cluster for outlier"
                    cosine_sim_matrix = np.dot(clustering_input[org_i],sorted_labels_avg.T)
                    max_sim_row = np.argmax(cosine_sim_matrix)
                    clustering_labels[org_i]=max_sim_row

            print "before final labeling"
            final_labeling = [clustering_labels[data_mapper[i]] for i in xrange(0,len(filtered_contexts_list))]
            if (args.analyze_adj_clusters):
                print "start file writing"
                output_clusters(label_id_gap+k_2,final_labeling)
                print "done file writing"
            print "done clustering"
        except :
            print "Failed to cluster adjective: [{}]".format(adj)
            print sys.exc_info()
            continue
        finally:
            print "Finally"


        print "before sent_to_labeled_adj"
        for i in xrange(0,len(filtered_contexts_list)):
            context = filtered_contexts_list[i]
            context.update_label( final_labeling[i])
            sent_to_labeled_adj[context.sentence_id].append(context)

        print "after sent_to_labeled_adj"
        joblib.dump(clustering_alg, '{}/{}/{}.pkl'.format(output_folder,adj_clusters_folder,adj))
        print "finished joblib dump"
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

    print "DONE!"