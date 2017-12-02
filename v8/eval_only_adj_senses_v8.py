import os
from multi_sense_we_wrapper_v8 import MultiSenseWE
from adj_noun_attr import AdjNounAttribute
import gensim
from torch.autograd import Variable
import torch
import numpy as np
import random
import operator
from scipy import spatial
import argparse
import torch.nn as nn
import torch.nn.functional as F
import logging

BATCH_SIZE = 5
D_IN = 300 #The dimension of 2 concatenated word2vec for adjective and noun
D_OUT = 300 #word2vec dimension
EPHOCS = 30
K = 5 #check within top K to evaluate the results
#
# dev_file_path = "/home/h/Documents/Hila/Research/dataset/HeiPLAS-release/HeiPLAS-dev.txt"
# test_file_path = "/home/h/Documents/Hila/Research/dataset/HeiPLAS-release/HeiPLAS-test.txt"
# word2vec_bin_path = "/home/h/Documents/Hila/Research/dataset/word2vec/GoogleNews-vectors-negative300.bin.gz"
# word2vec_text_normed_path = "/home/h/Documents/Hila/Research/dataset/word2vec/word2vec_text"
correct_predictions_file = "true_predictions"
false_prediction_file = "false_predictions"
test_results = "test_results"



class Model(nn.Module):
    def __init__(self, D_in, D_out):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(D_in,D_out,bias=False)
        weights = np.identity(D_out)
        self.linear_1.weight.data = torch.Tensor(weights)

    def forward(self, x):
        return self.linear_1(x)


def read_HeiPLAS_data(file_path):
    with open(file_path) as f:
        input_list = [line.split() for line in f.readlines()]
    data = [AdjNounAttribute(item[1],item[2],item[0].lower()) for item in input_list]
    return data


#########MAIM#############
def batch_training(batch_size = BATCH_SIZE, epochs = EPHOCS):
    running_loss = 0.0
    indices = range(y_train.shape[0])

    for epoch in range(epochs):
        print "Epoch: {}".format(epoch)
        random.shuffle(indices)
        for i in xrange(batch_size, y_train.shape[0] + batch_size, batch_size):
            if i >= y_train.shape[0]:
                current_indecies = indices[i - batch_size:y_train.shape[0] - 1]
            else:
                current_indecies = indices[i - batch_size:i]

            x = Variable(torch.Tensor(x_train[current_indecies]))
            y = Variable(torch.Tensor(y_train[current_indecies]), requires_grad=False)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = nn_model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            print(epoch, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:  #
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print "Finished batch training"

def online_training(epochs = EPHOCS ):
    running_loss = 0.0
    indices = range(y_train.shape[0])
    for epoch in range(epochs):
        print "Epoch: {}".format(epoch)
        random.shuffle(indices)
        for i in indices:

            x = Variable(torch.Tensor(x_train[[i]]))
            y = Variable(torch.Tensor(y_train[[i]]), requires_grad=False)

            # pytorch doesn't support directly in training without batching so this is kind of a hack
            x.unsqueeze(0)
            y.unsqueeze(0)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = nn_model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            # print(epoch, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.data[0]
            # if i % 100 == 99:  #
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0
    print "Finished online training"


def test():

    weights = nn_model.linear_1.weight.data.numpy()

    filtered_test_samp = [samp for samp in test_triplets
                          if samp.adj in we_wrapper.vocab and samp.noun in we_wrapper.vocab and samp.attr in we_wrapper.vocab]
    print "after filter missing words, testing samples: " + str(len(filtered_test_samp))

    x_test = np.array([we_wrapper.adj_vec_by_context(samp.adj,samp.noun) for samp in filtered_test_samp])
    y_test = np.array([we_wrapper.word_vec(samp.attr) for samp in filtered_test_samp])
    attr_vecs = {attr: we_wrapper.word_vec(attr) for attr in attributes if attr in we_wrapper.vocab}

    print "attr_vecs size = {}".format(len(attr_vecs))
    print "x test shape: " + str(x_test.shape)
    print "y_test: " + str(y_test.shape)
    print "weights shape: {}".format(weights.shape)

    x_test_matrix = np.dot(weights, np.transpose(x_test))
    print "x_test matrix shape = {}".format(x_test_matrix.shape)

    # check P@1 and P@5 accuracy
    correct = 0.0
    top_5_correct = 0.0
    correct_pred =[]
    false_pred = []
    results = []
    for i in xrange(0, x_test_matrix.shape[1]):
        y_pred = x_test_matrix[:, [i]]
        #calculate cosine similarity for normalized vectors
        cosine_sims = {attr: np.dot(y_pred.T, attr_vecs[attr]) for attr in attr_vecs.keys()}
        sorted_sims = dict(sorted(cosine_sims.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
        most_sim_attr = max(sorted_sims, key=lambda i: sorted_sims[i])
        if most_sim_attr == filtered_test_samp[i].attr:
            correct += 1
            correct_pred.append(filtered_test_samp[i])
        else:
            false_pred.append((filtered_test_samp[i],most_sim_attr))
        if filtered_test_samp[i].attr in sorted_sims.keys():
            top_5_correct += 1
        results.append((filtered_test_samp[i],most_sim_attr))
    print "correct: {} from total: {}. Accuracy: {}".format(correct, y_test.shape[0], correct / y_test.shape[0])
    print "top 5 correct: {} from total: {}. Accuracy: {}".format(top_5_correct, y_test.shape[0],
                                                                  top_5_correct / y_test.shape[0])

    file = open(args.output_folder +'/' +correct_predictions_file,'w')
    for item in correct_pred:
        string = ' '.join([item.attr.upper(),item.adj, item.noun])
        print >>file,string

    file = open(args.output_folder +'/' + false_prediction_file,'w')
    for item in false_pred:
        string = ' '.join([item[0].attr.upper(),item[0].adj, item[0].noun,item[1].upper()])
        print >>file,string

    file = open(args.output_folder +'/' +test_results,'w')
    for item in results:
        string = ' '.join([item[0].attr.upper(),item[0].adj, item[0].noun,item[1].upper()])
        print >>file,string



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec model.')

    parser.add_argument('dev_file',help='dev input file')
    parser.add_argument('test_file', help='test input file')
    parser.add_argument('we_file', help='word embeddings normed model file')
    parser.add_argument('adj_with_sense_file',help='list of adjectives with multi-sense representation file')
    parser.add_argument('adj_clusters_folder',help='adjectives clusters models folder')
    # parser.add_argument('clusters_per_adj',help='number of generated clusters per adjective with multi-sense representation')
    parser.add_argument('output_folder', help='path to the output folder')
    parser.add_argument('org_we_file', help='path to the original we model file - before adjectives clustering')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    dev_triplets = read_HeiPLAS_data(args.dev_file)
    attributes = {triplet.attr for triplet in dev_triplets}
    test_triplets= read_HeiPLAS_data(args.test_file)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # load pre-trained, normalized word2ec
    we_wrapper = MultiSenseWE(args.org_we_file, args.we_file,args.adj_with_sense_file,args.adj_clusters_folder)
    we_wrapper.set_model()

    #Filter samples that their words are missing from the word2vec vocabulary
    print "before filter missing words, training samples: " + str(len(dev_triplets))
    filtered_samp = [samp for samp in dev_triplets
                     if samp.adj in we_wrapper.vocab and samp.noun in we_wrapper.vocab and samp.attr in we_wrapper.vocab ]
    print "after filter missing words, training samples: " + str(len(filtered_samp))

    attributes = [attr for attr in attributes if attr in we_wrapper.vocab ]

    #
    # for samp in filtered_samp:
    #     print "[{}] [{}]".format(samp.adj,samp.noun)
    #generate trainig vectors
    x_train = np.array([we_wrapper.adj_vec_by_context(samp.adj,samp.noun) for samp in filtered_samp])
    y_train = np.array([we_wrapper.word_vec(samp.attr) for samp in filtered_samp])

    print "x shape: "+ str(x_train.shape)
    print "y_train: " + str(y_train.shape)

     #prepare NN model
    nn_model = Model(D_IN, D_OUT)
    criterion = torch.nn.MSELoss(size_average=True)#Mean Square Error loss
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-4)



    # batch_training(BATCH_SIZE)

    online_training()

    print "before filter missing words, testing samples: " + str(len(test_triplets))
    #Filter test samples that their words are missing from the word2vec vocabulary
    filtered_test_samp = [samp for samp in test_triplets
                          if samp.adj in we_wrapper.vocab and samp.noun in we_wrapper.vocab and samp.attr in we_wrapper.vocab]
    print "after filter missing words, testing samples: " + str(len(filtered_test_samp))
    
    test()

    # recall_test()

    print "Done!!!!!"