import os
import argparse
import gensim
import gzip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import math

TOKEN_ID_COLUMN = 0
TOKEN_COLUMN = 1
POS_COLUMN = 3
DEP_ID_COLUMN = 6
DEP_RELATION_COLUMN = 7

ADJ_TAG = 'JJ'
NOUN_TAG = 'NN'
K_CLUSTERS = 3
MIN_ITEMS_FOR_CLUSTERING = 20

class AdjContext:

    def __init__(self,adj,head_noun, sentence_id, token_id ):
        self.adj = adj.lower()
        self.head_noun = head_noun.lower()
        self.sentence_id = sentence_id
        self.token_id = token_id
        self.label_id = 0
        self.label = adj.lower()


    def __init__(self, parsed_token, sentence_id):
        self.adj = parsed_token.word.lower()
        self.head_noun = parsed_token.dep_word.lower()
        self.sentence_id = sentence_id
        self.token_id = parsed_token.id
        self.label_id = 0
        self.label = self.adj.lower()

    def update_label(self,label_id):
        self.label_id = label_id
        self.label = "{}_<{}>".format(self.adj,label_id)

class ParsedToken:

    def __init__(self,dependency_row_data, full_sentence_data):
        self.id = int(dependency_row_data[TOKEN_ID_COLUMN])
        self.word = dependency_row_data[TOKEN_COLUMN]
        self.pos = dependency_row_data[POS_COLUMN]

        dep_id = int(dependency_row_data[DEP_ID_COLUMN])
        if dep_id == 0 or full_sentence_data[dep_id-1][TOKEN_COLUMN] in ['.',',']: #if it's the root search for the closest noun in the sentence
            noun_candidates = []
            for token in full_sentence_data:
                if token[POS_COLUMN] == NOUN_TAG:
                    noun_candidates.append(int(token[TOKEN_ID_COLUMN]))
            if len(noun_candidates)>0:
                closest_noun = (noun_candidates[0], abs(noun_candidates[0]-self.id))
                for cand in noun_candidates:
                    dist =abs(cand-self.id)
                    if dist <closest_noun[1]:
                        closest_noun = (cand,dist)
                dep_id = closest_noun[0]
                dep_word = full_sentence_data[dep_id-1][TOKEN_COLUMN]
            else:
                dep_word = '<ROOT>'
        else:
             dep_word = full_sentence_data[dep_id-1][TOKEN_COLUMN]
        self.dep_id = dep_id#int(dependency_row_data[DEP_ID_COLUMN])
        self.dep_word = dep_word#full_sentence_data[self.dep_id-1][TOKEN_COLUMN]

        # self.dep_word = full_sentence_data[self.dep_id-1][TOKEN_COLUMN]


def generate_adj_data_mapper():
    adj_dict = {}
    sentence_id = 0

    with gzip.open(args.input_file, 'rb') as f:
        parsed_sentence = []
        count_bad_head = 0
        for line in f:
            if line != '\n':
                line_data = line.split('\t')
                parsed_sentence.append(line_data)
            else:
                for token_data in parsed_sentence:
                    if token_data[POS_COLUMN] == ADJ_TAG:  # if this is an adjective token
                        if parsed_sentence[int(token_data[DEP_ID_COLUMN])-1][TOKEN_COLUMN]  in ['.','isceral',]:
                            count_bad_head+=1
                        parsed_adj = ParsedToken(token_data, parsed_sentence)
                        adj_context = AdjContext(parsed_adj, sentence_id)
                        if not adj_dict.has_key(adj_context.adj):
                            adj_dict[adj_context.adj] = []
                        adj_dict[adj_context.adj].append(adj_context)

                sentence_id += 1
                parsed_sentence = []
                if (sentence_id % 100000 == 0):
                    print "finished pre-process sentence {}".format(sentence_id)
                    print "bad head = {}".format(str(count_bad_head))
                    break
    with open("output/adj_dic", 'w') as f:
        f.write("adjectives count: {}\n\n".format(len(adj_dict)))
        # print "adjectives count: {}".format(len(adj_dict))
        for adj, data in adj_dict.iteritems():
            f.write("{}\t{}\n".format(adj, len(data)))
            # print "{} - {} occurrences".format(adj,len(data))
            # print "parsed sentecnes = {}".format(count)
            # model = gensim.models.KeyedVectors.load(word2vec_text_normed_path, mmap='r') .wv # mmap the large matrix as read-only
            # model.syn0norm = model.syn0
            # load the trained word vectors

    return adj_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('input_file',help='input file path - parsed dat')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',help='the genrated file for WE training after adjectives clustering and labeling file path')
    args = parser.parse_args()

    if not os.path.exists('output'):
        os.makedirs('output')
    #load wiki parsed data
    # count = 0
    # sentences = []
    # with open('/home/h/data/wikipedia/wikipedia.corpus.nodups.clean.tokenized', 'r') as f:
    #
    #     for line in f:
    #         if count<100:
    #
    #             print line
    #             sentences.append(line)
    #         count+=1
    # print "{} sentences".format(count)
    #
    adj_clusters_folder = "adj_clusters"

    adj_dic = generate_adj_data_mapper()

    #load the trained word vectors
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    #remove adjective that occur less than MIN_ITEMS_FOR_CLUSTERING times
    filtered_adj_dic = {adj: contexts_list for adj, contexts_list in adj_dic.iteritems() if len(contexts_list) >MIN_ITEMS_FOR_CLUSTERING}
    with open("output/filtered_adj_dic", 'w') as f:
        f.write("adjectives count: {}\n\n".format(len(filtered_adj_dic)))
        for adj, data in filtered_adj_dic.iteritems():
            f.write("{}\t{}\n".format(adj, len(data)))

    #map adjective to 1/0 - has multi senses or not
    adj_to_multi = dict.fromkeys(adj_dic.keys(),0)
    for adj, contexts_list in adj_dic.iteritems():
        if len(contexts_list)>MIN_ITEMS_FOR_CLUSTERING:
            adj_to_multi[adj] = 1

    with open("output/adj_senses_flag_{}".format(MIN_ITEMS_FOR_CLUSTERING),'w') as f:
        for adj, flag in adj_to_multi.iteritems():
            f.write("{}\t{}\n".format(adj,str(flag)))


    if not os.path.exists('output/'+ adj_clusters_folder):
        os.makedirs('output/'+adj_clusters_folder)

    #cluster each adjective to senses
    for adj,contexts_list in filtered_adj_dic.iteritems():
        # stop_iter = False
        # for context in contexts_list:
        #     if context.head_noun not in model.vocab:
        #         print "{} not in vocab, skipping adjective: {}".format(context.head_noun,adj)
        #         stop_iter = True
        #         break
        # if stop_iter:
        #     continue
        has_wv_contexts = [context for context in contexts_list if context.head_noun in model.vocab]
        if (len(has_wv_contexts))<10:
            print "removing adj: {}".format(adj)
            continue
        clustering_input = np.array([model.word_vec(context.head_noun) for context  in has_wv_contexts])
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0).fit(clustering_input)
        for i in xrange(0,len(has_wv_contexts)):
            has_wv_contexts[i].update_label( kmeans.labels_[i])
        joblib.dump(kmeans, 'output/{}/{}.pkl'.format(adj_clusters_folder,adj))


    #update corpus
    with gzip.open(args.input_file, 'rb') as fi, open("output/"+args.sentences_output_file,'w')as fo:
        sentence = []
        sentence_id = 0
        for line in fi:
            if line != '\n':
                line_data = line.split('\t')
                word = line_data[TOKEN_COLUMN].lower()
                pos = line_data[POS_COLUMN]
                if pos == ADJ_TAG:
                    token_id = int(line_data[TOKEN_ID_COLUMN])
                    contexts = adj_dic[word]
                    for context in contexts:
                        if context.sentence_id == sentence_id and token_id == context.token_id:
                            word = context.label
                            break

                sentence.append(word)
            else:
                fo.write(' '.join(sentence).strip(".")+'\n')
                sentence_id += 1
                sentence = []
                if (sentence_id % 100000 == 0):
                    print "update corpus:  sentence {}".format(sentence_id)
                    break