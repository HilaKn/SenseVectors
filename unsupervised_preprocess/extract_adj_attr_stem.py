#This script is for checking weather the noun context of adejctives
# can help in extracting all the adjective possible attributes
#by similarity measure between the nouns and the attributes vectors.

import gensim
import argparse
import os
import numpy as np
from operator import itemgetter
from nltk import stem
from adj_noun_attr import AdjNounAttribute
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import  PorterStemmer
from nltk.stem import RegexpStemmer

ADJ_ATTR_FOLDER = "adj_attr_output_stem"
ATTR_RESULTS_FILE = "attr_results"
STEM_RESULTS_FILE = "stem_results_file"

def lcs(S,T):
    if S == 'evil' and T == 'devilish':
        print "here"
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])


    if len(list(lcs_set)) > 0:
        return len(list(lcs_set)[0])
    else:
        return 0
    # return list(lcs_set)

def get_attributes_list(data_set):
    attributes = [samp.attr for samp in data_set]
    unique_attr = set(attributes)
    print "Number of attributes: [{}]".format(len(unique_attr))
    return list(unique_attr)

def read_HeiPLAS_data(file_path):
    with open(file_path) as f:
        input_list = [line.split() for line in f.readlines()]
    data = [AdjNounAttribute(item[1],item[2],item[0].lower()) for item in input_list]
    return data


def predict_attr(data_set,org_attributes,adj_vecs, attr_vecs,output_file):
    adj_attr_cosine = np.dot(adj_vecs, attr_vecs.T)
    index_to_max_att_per_adj = np.argmax(adj_attr_cosine,
                                         axis=1)  # take the row index of the max score for each attribute

    lcs_attr_idx =[]
    for idx,samp in enumerate(data_set):
        attr_lcs = [lcs(attr,samp.adj) for attr in org_attributes]
        lcs_len  = max(attr_lcs)
        attr_idx = attr_lcs.index(lcs_len)
        # attr = org_attributes[attr_idx]
        lcs_attr_idx.append(attr_idx)
        if lcs_len >4:
            # if samp.attr == org_attributes[attr_idx]  :
            index_to_max_att_per_adj[idx] = attr_idx
            # elif org_attributes[index_to_max_att_per_adj[idx]]== samp.attr:
            #     print " ".join ([samp.attr.upper(), samp.adj, samp.noun,org_attributes[attr_idx].upper()])

    correct = 0.0
    results = []
    for samp_id, attr_idx in enumerate(index_to_max_att_per_adj):
        if data_set[samp_id].attr == org_attributes[attr_idx]:
            correct += 1
        res = AdjNounAttribute(data_set[samp_id].adj, data_set[samp_id].noun, org_attributes[attr_idx])
        results.append(res)
    print "According to adj-attr similarity:"
    print "Total samples: [{}]. Correct: [{}]. Accuracy: [{}]".format(len(data_set), correct, correct / len(data_set))
    with open(output_file, 'w') as f:
        for res in results:
            row = '\t'.join([res.attr.upper(), res.adj, res.noun]) + '\n'
            f.write(row)

def my_stem(word):
    st =RegexpStemmer('ness$|ity$|ment', min=4)
    if word.endswith('acy'):
        stem = word[:-2]
        stem += 'te'
    elif word.endswith('cy'):
        stem = word[:-2]
        stem+= 't'

    elif word.endswith('ility'):
        stem = word[:-5]
        stem+= 'le'
        if stem not in model.vocab:
            stem = word[:-3]

    # elif word.endswith('ality'):
    #     stem = word[:-5]
    #     if stem not in model.vocab:
    #         stem = word[:-3]

    elif word.endswith('ce'):
        stem = word[:-2]
        stem += 't'

    else:
        stem = st.stem(word)
        if stem.endswith('i'):
            stem = stem[:-1] + 'y'
    return stem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract all possible attributes of adjectives')

    parser.add_argument('Hei_PLAS_file',help='path to HeiPLAS dev file')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    args = parser.parse_args()

    if not os.path.exists(ADJ_ATTR_FOLDER):
        os.makedirs(ADJ_ATTR_FOLDER)

     # 1. Load WE vectors
    model = gensim.models.KeyedVectors.load(args.word_embeddings_file, mmap='r') .wv # mmap the large matrix as read-only
    model.syn0norm = model.syn0

    data_set = read_HeiPLAS_data(args.Hei_PLAS_file)
    data_set = [samp for samp in data_set if samp.adj in model.vocab
               and samp.noun in model.vocab and samp.attr in model.vocab and samp.attr not in ["good"]]
    print "Total samples = [{}]".format(len(data_set))

    # 2. Load all attribute names
    attributes = get_attributes_list(data_set)
    attr_vecs = np.array([model.word_vec(attr) for attr in attributes ])
    attr_vecs.squeeze()

    # 3. convert to stem form
    # st = LancasterStemmer()
    #st = PorterStemmer()
    # st =RegexpStemmer('ness$|ity$|', min=4)
    stemmed_attr = [my_stem(attr) if my_stem(attr) in model.vocab else attr  for attr in attributes ]
    stem_attr_vecs = np.array([model.word_vec(attr) for attr in stemmed_attr ])
    stem_attr_vecs.squeeze()

    #4. get adj vectors
    adj_vecs = np.array([model.word_vec(samp.adj) for samp in data_set])
    adj_vecs.squeeze()

    predict_attr(data_set,attributes,adj_vecs,attr_vecs,ATTR_RESULTS_FILE)
    predict_attr(data_set,attributes,adj_vecs,stem_attr_vecs,STEM_RESULTS_FILE)


    #Predict according to the heiger similarity score