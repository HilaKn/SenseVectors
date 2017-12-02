#This script is for checking weather the noun context of adejctives
# can help in extracting all the adjective possible attributes
#by similarity measure between the nouns and the attributes vectors.

import gensim
import argparse
import os
import numpy as np
from operator import itemgetter

ADJ_ATTR_FOLDER = "adj_attr_avg_output"

def get_attributes_list():
    with open(args.Hei_PLAS_dev_file, 'r') as f:
        rows = f.readlines()
        attributes = [row.split()[0].rstrip().lower() for row in rows]
    unique_attr = set(attributes)
    print "Number of attributes: [{}]".format(len(unique_attr))
    return list(unique_attr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract all possible attributes of adjectives')

    parser.add_argument('Hei_PLAS_dev_file',help='path to HeiPLAS dev file')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    parser.add_argument('adj_clusters_folder',help='path to the adjective clusters folder')
    args = parser.parse_args()

    if not os.path.exists(ADJ_ATTR_FOLDER):
        os.makedirs(ADJ_ATTR_FOLDER)

     # 1. Load WE vectors
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    # 2. Load all attribute names
    attributes = get_attributes_list()
    attributes = [attr for attr in attributes if attr in model.vocab]


    adj_list = ['abstract', 'abstruse']
    # sim = model.n_similarity(adj_list,['temperature'])
    # print sim
    cosine_sims = np.array([model.n_similarity(adj_list,[attr]) for attr in attributes])
    attr_score = [(attributes[att_index],score) for att_index,score in enumerate(cosine_sims)]
    sorted_data = sorted(attr_score,key=itemgetter(1),reverse=True)
    for data in sorted_data:
        row = '\t'.join(str(i) for i in data) + '\n'
        print row

    # 3.From adj_analysis_folder:
    # for filename in os.listdir(args.adj_clusters_folder):
    #     # 3.1. load list of all noun contexts
    #     with open(args.adj_clusters_folder + '/' + filename) as f:
    #         rows = f.readlines()
    #
    #     print "start processing [{}]".format(filename)
    #     noun_contexts = [row.split('\t')[1].rstrip() for row in rows if row.split('\t')[0] != "label"]
    #     cosine_sims = np.array([model.n_similarity(noun_contexts,attr) for attr in attributes]).squeeze()
    #     attr_score = [(attributes[att_index],score) for att_index,score in enumerate(cosine_sims)]
    #     sorted_data = sorted(attr_score,key=itemgetter(1),reverse=True)
    #     with open (ADJ_ATTR_FOLDER + '/' + filename + '_noun', 'w') as f:
    #         for data in sorted_data:
    #             row = '\t'.join(str(i) for i in data) + '\n'
    #             print>>f, row
