#This script is for checking weather the noun context of adejctives
# can help in extracting all the adjective possible attributes
#by similarity measure between the nouns and the attributes vectors.

import gensim
import argparse
import os
import numpy as np
from operator import itemgetter

ADJ_ATTR_FOLDER = "adj_attr_output"

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
    attr_vecs = np.array([model.word_vec(attr) for attr in attributes ])
    attr_vecs.squeeze()
    attr_noun_score = {}



    # 3.From adj_analysis_folder:
    for filename in os.listdir(args.adj_clusters_folder):
        # 3.1. load list of all noun contexts
        with open(args.adj_clusters_folder + '/' + filename) as f:
            rows = f.readlines()
        print "start processing [{}]".format(filename)
        noun_contexts = [row.split('\t')[1].rstrip() for row in rows if row.split('\t')[0] != "label"]
        noun_vecs = np.array([model.word_vec(noun) for noun in noun_contexts])  #no need to filter out of vocab nouns cause it was already done in the preprocess phase - only
                                                                              #nuns with WE were counted as contexts
        noun_vecs.squeeze()

        cosine_sims = np.dot(noun_vecs, attr_vecs.T)

        index_to_max_noun_per_att = np.argmax(cosine_sims,axis=0)#take the row index of the max score for each attribute
        attr_noun_score = [(attributes[att_index],noun_contexts[noun_index],cosine_sims[noun_index,att_index])
                    for att_index,noun_index in enumerate(index_to_max_noun_per_att)]
        sorted_data = sorted(attr_noun_score,key=itemgetter(2),reverse=True)
        with open (ADJ_ATTR_FOLDER + '/' + filename + '_noun', 'w') as f:
            for data in sorted_data:
                row = '\t'.join(str(i) for i in data) + '\n'
                print>>f, row

        adj_cosine_sims = np.dot(model.word_vec(filename),attr_vecs.T)
        sorted_indexes = np.argsort(adj_cosine_sims)
        adj_attr_score = [(attributes[att_index],adj_cosine_sims[att_index])
                    for att_index in sorted_indexes]

        sorted_data = sorted(adj_attr_score,key=itemgetter(1),reverse=True)
        with open (ADJ_ATTR_FOLDER + '/' + filename, 'w') as f:
            for data in sorted_data:
                row = '\t'.join(str(i) for i in data) + '\n'
                print>>f, row
    #     For each adjective file:

    #       3.2  for each attribute search for the closest noun and save the similarity score
    #       3.3. sort the attributes decanting according to the score
    #       3.4. print results to file (attribute,noun,score) list



    #Later phase - build evaluation scheme based on wordnet attributes links :
    #
