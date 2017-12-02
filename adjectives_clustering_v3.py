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
output_folder = "output_v3"

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
        return head_id, head_word

    def __init__(self,dependency_row_data, full_sentence_data,sentence_id):
        self.adj = dependency_row_data[TOKEN_COLUMN].lower()
        self.sentence_id = sentence_id
        self.token_id = int(dependency_row_data[TOKEN_ID_COLUMN])
        self.label_id = 0
        self.label = self.adj.lower()

        head_id, head_noun = self.get_head_noun(dependency_row_data, full_sentence_data)
        self.head_id = head_id
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
                    if token_data[POS_COLUMN] == ADJ_TAG:  # if this is an adjective token
                        adj_context = AdjContext(token_data, parsed_sentence,sentence_id)
                        adj_dict[adj_context.adj].append(adj_context)
                        sentence_to_adj_context[sentence_id].append(adj_context)
                sentence_id += 1
                parsed_sentence = []
                if (sentence_id % 100000 == 0):
                    print "finished pre-process sentence {}".format(sentence_id)
                    # break
    #
    # with open(output_folder+"/adj_dic", 'w') as f:
    #     f.write("adjectives count: {}\n\n".format(len(adj_dict)))
    #     for adj, data in adj_dict.iteritems():
    #         f.write("{}\t{}\n".format(adj, len(data)))

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

    adj_dic,sentence_to_adj_context = generate_adj_data_mapper()

    #load the trained word vectors
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    #remove adjective that occur less than MIN_ITEMS_FOR_CLUSTERING times
    #remove adjectives that occure less that MIN_ITEMS_FOR_CLUSTERING with head noun with word vector representation
    # temp_filtered_adj_dic = {adj: contexts_list for adj, contexts_list in adj_dic.iteritems() if len(contexts_list) >MIN_ITEMS_FOR_CLUSTERING}
    filtered_adj_dic={}
    for adj,contexts_list in adj_dic.iteritems():
        if len(contexts_list)>MIN_ITEMS_FOR_CLUSTERING:
            have_wv_contexts = [context for context in contexts_list if context.head_noun in model.vocab]
            if (len(have_wv_contexts)) > MIN_ITEMS_FOR_CLUSTERING:
                filtered_adj_dic[adj] = have_wv_contexts
            else:
                print "1_removing adj: {}".format(adj)
        else:
            print "2_removing adj: {}".format(adj)


    with open(output_folder+"/filtered_adj_dic", 'w') as f:
        f.write("adjectives count: {}\n\n".format(len(filtered_adj_dic)))
        for adj, data in filtered_adj_dic.iteritems():
            f.write("{}\t{}\n".format(adj, len(data)))

    #map adjective to 1/0 - has multi senses or not
    adj_to_multi = dict.fromkeys(adj_dic.keys(),0)
    for adj in filtered_adj_dic.keys():
        adj_to_multi[adj] = 1

    with open("{}/adj_senses_flag_{}".format(output_folder,MIN_ITEMS_FOR_CLUSTERING),'w') as f:
        for adj, flag in adj_to_multi.iteritems():
            f.write("{}\t{}\n".format(adj,str(flag)))


    #cluster each adjective to senses
    for adj,contexts_list in filtered_adj_dic.iteritems():
        "Clustering adj: {}".format(adj)
        clustering_input = np.array([model.word_vec(context.head_noun) for context  in have_wv_contexts])
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0).fit(clustering_input)
        for i in xrange(0,len(have_wv_contexts)):
            have_wv_contexts[i].update_label( kmeans.labels_[i])
        joblib.dump(kmeans, '{}/{}/{}.pkl'.format(output_folder,adj_clusters_folder,adj))

    print "done clustering"
    #update corpus
    # with gzip.open(args.input_file, 'rb') as fi, open(output_folder+"/"+args.sentences_output_file,'w')as fo:
    #     sentence = []
    #     sentence_id = 0
    #     for line in fi:
    #         if line != '\n':
    #             line_data = line.split('\t')
    #             word = line_data[TOKEN_COLUMN].lower()
    #             pos = line_data[POS_COLUMN]
    #             if pos == ADJ_TAG and filtered_adj_dic.has_key(word):
    #                 token_id = int(line_data[TOKEN_ID_COLUMN])
    #                 contexts = filtered_adj_dic[word]
    #                 for context in contexts:
    #                     if context.sentence_id == sentence_id and token_id == context.token_id:
    #                         word = context.label
    #                         break
    #
    #             sentence.append(word)
    #         else:
    #             fo.write(' '.join(sentence).strip(".")+'\n')
    #             sentence_id += 1
    #             sentence = []
    #             if (sentence_id % 100000 == 0):
    #                 print "update corpus:  sentence {}".format(sentence_id)
    #                 break

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
                # break
    print "done generating new sentences file"