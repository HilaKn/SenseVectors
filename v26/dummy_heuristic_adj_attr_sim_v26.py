import main as m
import gensim
from AdjNounAtt import AdjNounAttribute
import numpy as np

K =5
predictions_file = "dummy_heuristic_adj_attr_test_results"

class Predictor:
    def __init__(self,dev_set,attributes):
        self.dev_set = dev_set
        self.dev_adjectives  = [item.adj for item in dev_set]
        self.dev_nouns = [item.noun for item in dev_set]
        self.attributes = list(attributes)
        self.attr_vecs = np.array([model.word_vec(attr) for attr in self.attributes]).squeeze()

    def predict(self,samp):
        sim = np.dot(model.word_vec(samp.adj),self.attr_vecs.T)
        attr_ids = sim.argsort()[-K:][::-1]
        attr_id = np.argmax(sim)
        adj_attrs = [self.attributes[i] for i in attr_ids]


        return adj_attrs






dev_set = m.read_HeiPLAS_data(m.dev_file_path)
test_set = m.read_HeiPLAS_data(m.test_file_path)

model = gensim.models.KeyedVectors.load(m.word2vec_text_normed_path, mmap='r')  # mmap the large matrix as read-only
model.syn0norm = model.syn0

dev_filtered_samp = [samp for samp in dev_set
                     if samp.adj in model.vocab and samp.noun in model.vocab and samp.attr in model.vocab ]
filtered_test_samp = [samp for samp in test_set
                          if samp.adj in model.vocab and samp.noun in model.vocab and samp.attr in model.vocab]

attributes = list(set([samp.attr for samp in dev_filtered_samp]))
predictor = Predictor(dev_filtered_samp,attributes)

correct = 0.0
correct_in_K = 0.0
predictions = []
for test in filtered_test_samp:
    print "{} {}".format(test.adj, test.noun)
    adj_preds = predictor.predict(test)
    predictions.append((AdjNounAttribute(test.adj,test.noun,test.attr),adj_preds[0]))
    if adj_preds[0] == test.attr:
        correct += 1
    if test.attr in adj_preds:
        correct_in_K += 1

print "correct = {}, total: {}, accuracy: {}".format(correct, len(filtered_test_samp), correct/len(filtered_test_samp))
print "correct_in_{} = {}, total: {}, accuracy: {}".format(K, correct_in_K, len(filtered_test_samp), correct_in_K/len(filtered_test_samp))

file = open(predictions_file,'w')
for item in predictions:
    string = ' '.join([item[0].attr.upper(),item[0].adj, item[0].noun,item[1].upper()])
    print >>file,string