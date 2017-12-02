import os
import argparse
import gensim
import gzip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import math
from collections import defaultdict

TOKEN_ID_COLUMN = 0
TOKEN_COLUMN = 1
POS_COLUMN = 3
DEP_ID_COLUMN = 6
DEP_RELATION_COLUMN = 7

ADJ_TAG = 'JJ'
NOUN_TAGS = ['NN','NNS','NNP']#Todo: consider adding proper nouns
K_CLUSTERS = 3
MIN_ITEMS_FOR_CLUSTERING = 100
NON_RELEVANT_HEAD = ['.',',']
adj_clusters_folder = "adj_clusters"
output_folder = "output_v4"

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

            if len(noun_candidates) > 0:
                id = int(dependency_row_data[TOKEN_ID_COLUMN])
                closest_noun = (noun_candidates[0], abs(noun_candidates[0] - id))
                for cand in noun_candidates:
                    dist = abs(cand - id)
                    if dist < closest_noun[1]:
                        closest_noun = (cand, dist)
                head_id = closest_noun[0]
                head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]
            else:
                # print "no noun candidates"
                head_word = '<ROOT>'
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
        self.head_noun = head_noun
        
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
                    break

    print "End generate_adj_data_mapper method"
    return adj_dict,sentence_to_adj_context


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('parsed_input_file',help='input file path - parsed data')
    parser.add_argument('sentences_input_file',help='input file path - sentences format')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',help='the genrated file for WE training after adjectives clustering and labeling file path')
    args = parser.parse_args()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder+'/'+ adj_clusters_folder):
        os.makedirs(output_folder+'/'+adj_clusters_folder)

    #load the trained word vectors
    print "Loading word vectors from {}".format(args.word_embeddings_file)
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    adj_dic,sentence_to_adj_context = generate_adj_data_mapper()

    #remove adjective that occur less than MIN_ITEMS_FOR_CLUSTERING times
    print "Filter out adjectives with less than {} relevant contexts".format(MIN_ITEMS_FOR_CLUSTERING)
    filtered_adj_dic = {adj: contexts_list for adj, contexts_list in adj_dic.iteritems() if len(contexts_list) >MIN_ITEMS_FOR_CLUSTERING}

    print "Write to file the the filtered adjectives"
    with open(output_folder+"/filtered_adj_dic", 'w') as f:
        f.write("adjectives count: {}\n\n".format(len(filtered_adj_dic)))
        for adj, data in filtered_adj_dic.iteritems():
            f.write("{}\t{}\n".format(adj, len(data)))

    print "Start clustering {} adjectives".format(len(filtered_adj_dic))
    #cluster each adjective to senses
    for adj,contexts_list in filtered_adj_dic.iteritems():
        print "Clustering adj: [{}] with [{}] contexts".format(adj, len(contexts_list))
        clustering_input = np.array([model.word_vec(context.head_noun) for context  in contexts_list])
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0).fit(clustering_input)
        for i in xrange(0,len(contexts_list)):
            contexts_list[i].update_label( kmeans.labels_[i])
        joblib.dump(kmeans, '{}/{}/{}.pkl'.format(output_folder,adj_clusters_folder,adj))

    print "Finished clustering all adjectives"

    #update corpus from original sentences file
    print "Start updating corpus with new adjectives labels"
    with open(args.sentences_input_file, 'rb') as fi, open(output_folder+"/"+args.sentences_output_file,'w')as fo:
        sentence = []
        sentence_id = 0
        for line in fi:
            if sentence_to_adj_context.has_key(sentence_id):
                line_data = line.split()
                for context in sentence_to_adj_context[sentence_id]:
                    line_data[context.token_id-1] = context.label
                fo.write(' '.join(line_data))

            sentence_id += 1
            if (sentence_id % 100000 == 0):
                print "update corpus:  sentence {}".format(sentence_id)
                break
    print "Finished generating new sentences file"