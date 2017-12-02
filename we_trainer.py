import argparse
from gensim.models.wrappers import FastText
from gensim.models.word2vec import LineSentence, Word2Vec,KeyedVectors
import multiprocessing
import math
import time
import  os
import logging

MAX_CORES_TO_USE = 40
EPOCHS = 5

if __name__ == '__main__':
    start_time = time.time()
    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('input_file',help='input file path for the word embeddings training')
    parser.add_argument('output_file', help='output file path for the word embeddings model')
    parser.add_argument('--dimension',default=300, help='input file path for the word embeddings training')
    parser.add_argument('--lr',default=0.025, help = 'initial learning rate')
    parser.add_argument('--parallelism','-p', default=False,action='store_true', help='use multi threaded training or not (according to number of cores)')
    parser.add_argument('--window', default=5,type=int, help='size of the context window')
    args = parser.parse_args()

    free_cores =1
    if args.parallelism:
        cores = multiprocessing.cpu_count()
        uptime_data = os.popen("uptime").read().split()
        load_avg = float(uptime_data[-3].strip(','))#take the load average of the last minute(the third from the end)
        used_cores = math.ceil(load_avg/cores)
        free_cores = min(cores - used_cores,MAX_CORES_TO_USE)
        print "running with {} threads".format(free_cores)


    sentences = LineSentence(args.input_file)


    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences, size = args.dimension,alpha = args.lr,window = args.window,
                     workers=free_cores, iter=EPOCHS)

    model.save('{}_{}_{}'.format(args.output_file, args.dimension,args.window))

    model.init_sims(replace = True)
    model.save('{}_{}_{}_normed'.format(args.output_file,args.dimension,args.window))

    end_time = time.time()

    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
