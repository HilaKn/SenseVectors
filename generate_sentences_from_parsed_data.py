import argparse
from gensim.models.wrappers import FastText
from gensim.models.word2vec import LineSentence, Word2Vec
import multiprocessing
import math
import time
import  os
import logging
import gzip

MAX_CORES_TO_USE = 40
EPOCHS = 5
TOKEN_COLUMN = 1

#generate sentences file from the parsed wiki data for later WE training
if __name__ == '__main__':
    start_time = time.time()
    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Generate text file with sentences from parser output.')

    parser.add_argument('input_file',help='input file path containing parser output in .gz format')
    parser.add_argument('output_file',help='output file for the extracted sentences')
    args = parser.parse_args()

    with gzip.open(args.input_file, 'rb') as fi, open(args.output_file,'w')as fo:
        sentence = []
        sentence_id = 0
        for line in fi:
            if line != '\n':
                word = line.split('\t')[TOKEN_COLUMN].lower()
                sentence.append(word)
            else:
                fo.write(' '.join(sentence).strip(".")+'\n')
                sentence_id += 1
                sentence = []
                if (sentence_id % 100000 == 0):
                    print "finished process sentence {}".format(sentence_id)
                    # break

    print "Done generating sentences from: {}.\nOutput is under: {}.".format(args.input_file,args.output_file)
