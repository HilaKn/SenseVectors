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
import json

TOKEN_ID_COLUMN = 0
TOKEN_COLUMN = 1
POS_COLUMN = 3
DEP_ID_COLUMN = 6
DEP_RELATION_COLUMN = 7

NO_HEAD_NOUN_TAG = '<no_head_noun_tag>'
ADJ_TAG = 'JJ'
NOUN_TAGS = ['NN','NNS','NNP']#Todo: consider adding proper nouns
NON_RELEVANT_HEAD = ['.',',']
output_folder = "output_v5"
ADJ_DIC_FILE = "adj_dic_object_pickle"
SENT_TO_ADJ_CONTEXT_FILE = "sentence_to_adj_context_object_pickle"

class AdjContext:

    def get_head_noun(self, dependency_row_data, full_sentence_data):
        # the head noun is not setting correctly sometimes by the parser.
        # so here is a simple heuristic to take the closest noun to the
        # adjective as the head noun in case the head noun is the root
        # or some non reasonable token (e.g. '.')
        head_id = int(dependency_row_data[DEP_ID_COLUMN])
        if head_id == 0 or full_sentence_data[head_id - 1][TOKEN_COLUMN] in NON_RELEVANT_HEAD:
            noun_candidates = [int(token[TOKEN_ID_COLUMN]) for token in full_sentence_data
                               if token[POS_COLUMN] in NOUN_TAGS]
            head_word = NO_HEAD_NOUN_TAG
            if len(noun_candidates) > 0:
                id = int(dependency_row_data[TOKEN_ID_COLUMN])
                closest_noun = (noun_candidates[0], abs(noun_candidates[0] - id))
                for cand in noun_candidates:
                    dist = abs(cand - id)
                    if dist < closest_noun[1]:
                        closest_noun = (cand, dist)
                head_id = closest_noun[0]
                head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]

        #If the dep relation is not to a noun:
        #1. If relation is conj go recursivly to search for the head noun
        #2. Else: set to unrecognized head (to be removed later
        elif full_sentence_data[head_id - 1][POS_COLUMN] not in NOUN_TAGS:
            head_word = NO_HEAD_NOUN_TAG
            if dependency_row_data[DEP_RELATION_COLUMN] == "conj":
                head_word = self.get_head_noun(full_sentence_data[head_id - 1],full_sentence_data)

        else:
             head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]

        return head_word

    def __init__(self,dependency_row_data, full_sentence_data,sentence_id):
        self.adj = dependency_row_data[TOKEN_COLUMN].lower()
        self.sentence_id = sentence_id
        self.token_id = int(dependency_row_data[TOKEN_ID_COLUMN])
        self.label_id = 0
        self.label = self.adj.lower()

        head_noun = self.get_head_noun(dependency_row_data, full_sentence_data)
        self.head_noun = head_noun.lower()
        
    def update_label(self,label_id):
        self.label_id = label_id
        self.label = "{}_<{}>".format(self.adj,label_id)

def generate_adj_data_mapper():
    adj_dict = defaultdict(list)
    sentence_to_adj_context = defaultdict(list)
    sentence_id = 0

    with gzip.open(args.parsed_input_file, 'rb') as f:
        parsed_sentence = []
        for line in f:
            if line != '\n':
                line_data = line.split('\t')
                parsed_sentence.append(line_data)
            else:
                for token_data in parsed_sentence:
                    if token_data[POS_COLUMN] == ADJ_TAG and token_data[TOKEN_COLUMN].lower() in model.vocab:  # if this is an adjective token
                        adj_context = AdjContext(token_data, parsed_sentence,sentence_id)
                        #to save memory - save only contexts that have WE otherwise it can't be
                        # clustered anyway
                        if adj_context.head_noun in model.vocab:
                            adj_dict[adj_context.adj].append(adj_context)
                            sentence_to_adj_context[sentence_id].append(adj_context)
                sentence_id += 1
                parsed_sentence = []
                if (sentence_id % 100000 == 0):
                    print "finished pre-process sentence {}".format(sentence_id)
                    # break

    print "End generate_adj_data_mapper method"
    return adj_dict,sentence_to_adj_context


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('parsed_input_file',help='input file path - parsed data')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')

    args = parser.parse_args()


    #load the trained word vectors
    print "Loading word vectors from {}".format(args.word_embeddings_file)
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    adj_dic,sentence_to_adj_context = generate_adj_data_mapper()

    print "Saving objects to file using cPickle"
    with open(ADJ_DIC_FILE, 'wb') as f1, open(SENT_TO_ADJ_CONTEXT_FILE,'wb') as f2:
        cPickle.dump(adj_dic, f1)
        cPickle.dump(sentence_to_adj_context, f2)