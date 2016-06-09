'''
Trains a RNN on medical diagnosis of diseases dataset
   data is obtained from various online sources
   Memory network needs to predict the disease using many symptoms listed as 
   natural language sentences
   
'''

from __future__ import print_function
from functools import reduce
import re
import tarfile
import os.path
import pickle
import h5py
import pdb
from itertools import izip_longest

import random
import numpy as np
np.random.seed(1337)  # for reproducibility
random.seed(1337)

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, Callback
from utils import create_vectors_dataset, get_spacy_vectors
from glove import Glove
from spacy.en import English



RNN = recurrent.LSTM
NUM_HIDDEN_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 10
DROPOUT_FACTOR = 0.5
print('RNN / HIDDENS = {}, {}'.format(RNN, NUM_HIDDEN_UNITS))

max_len = 500
word_vec_dim = 300
vocab_size = 2350

training_set_file = 'data/training_set.dat'
test_set_file = 'data/test_set.dat'

train_stories = pickle.load(open(training_set_file,'r'))
test_stories = pickle.load(open(test_set_file,'r'))

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q) for fact,q in train_stories]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q) for fact,q in test_stories]

answer_vocab = sorted(reduce(lambda x, y: x | y, (set([answer]) for _, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
answer_dict = dict((word, i) for i, word in enumerate(answer_vocab))
print('Answers dict len: {0}'.format(len(answer_dict)))

# I need to check also if this exist
#word_vectors_dir = 'word_vectors/glove.42B.300d.txt'
#word_vectors_model = Glove.load_stanford(word_vectors_dir)
nlp = English()


print('Build model...')

model = Sequential()
model.add(GRU(output_dim = NUM_HIDDEN_UNITS, activation='tanh', 
               return_sequences=True, input_shape=(max_len, word_vec_dim)))
model.add(Dropout(DROPOUT_FACTOR))
model.add(GRU(NUM_HIDDEN_UNITS, return_sequences=False))
model.add(Dense(vocab_size, init='uniform',activation='softmax'))

#json_string = model.to_json()
#model_file_name = 'models/lstm_num_hidden_units_' + str(NUM_HIDDEN_UNITS) + '_num_lstm_layers_' + str(2) + '_dropout_' + str(0.3)
#open(model_file_name  + '.json', 'w').write(json_string)

print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Compilation done...')

print('Training')

base_dir = '.'

NUM_DATA_TRAIN = len(train_stories)
NUM_DATA_TEST = len(test_stories)

random.shuffle(train_stories)
valid_stories = train_stories[int(len(train_stories)*0.95):]
train_stories = train_stories[:int(len(train_stories)*0.95)]
print('Validation size: {0}'.format(len(valid_stories)))
print('Training size: {0}'.format(len(train_stories)))

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

acc_hist = []
acc_hist.append(0.)
show_batch_interval = 50
for k in xrange(EPOCHS):
    for b,train_batch in enumerate(zip(grouper(train_stories, BATCH_SIZE, fillvalue=train_stories[-1]))):
        X,Y = get_spacy_vectors(train_batch[0], answer_dict, 
                                         max_len, nlp)
            
        loss = model.train_on_batch(X, Y)
        if b % show_batch_interval == 0:
            print('Epoch: {0}, Batch: {1}, loss: {2}'.format(k,b,loss))
    
    X,Y = get_spacy_vectors(valid_stories, answer_dict, 
                                        max_len, nlp)
    loss, acc = model.evaluate(X, Y, batch_size=BATCH_SIZE)
    print('Epoch{0}, Valid loss / valid accuracy = {1:.4f} / {2:.4f}'.format(k,loss, acc))
    #Logging results
    with open(base_dir + '/logs/log_{0}_{1}_drop_{2}.txt'.format(
                'GRU',str(NUM_HIDDEN_UNITS),str(DROPOUT_FACTOR)),'a') as fil:
        fil.write(str(loss) + ' ' + str(acc) + '\n')
    #Saving model
    if max(acc_hist) < acc:
        model.save_weights(base_dir + '/models/weights_{0}_{1}_drop_{2}.hdf5'.format(
                'GRU',str(NUM_HIDDEN_UNITS),str(DROPOUT_FACTOR)),overwrite=True)
    acc_hist.append(acc)
    
# Obtaining test results
# Evaluatin Best 5 accuracy and best accuracy
SAVE_ERRORS = False

acc_5 = 0.
acc = 0.
for b,test_batch in enumerate(zip(grouper(test_stories, BATCH_SIZE, fillvalue=test_stories[-1]))):
    X,Y = get_spacy_vectors(test_batch[0], answer_dict, 
                                     max_len, nlp)
    answers_test = Y if b == 0 else np.vstack((answers_test,Y))
    preds = model.predict(X)
    # Saving in order to make some more visualizations
    all_predictions = preds if b == 0 else np.vstack((all_predictions,preds))
    if b % 50 == 0:
        print('Batch: {0}'.format(b))

all_predictions = all_predictions[:len(test_stories)]
answers_test = answers_test[:len(test_stories)]
for k,(pred,answer) in enumerate(zip(all_predictions,answers_test)):
    prediction = np.argsort(pred)[-5:][::-1]
    pred_words = [answer_dict.keys()[answer_dict.values().index(pred)] for pred in prediction]
    answer_word = answer_dict.keys()[answer_dict.values().index(answer.argmax())] 
    if answer_word in pred_words:
        acc_5 += 1.
    if pred_words[0] == answer_word:
        acc += 1.

all_err = -np.log(all_predictions[range(all_predictions.shape[0]),answers_test.argmax(axis=1)])

np.savetxt('logs/error.dat',all_err)
          
acc /= len(test_stories)
acc_5 /= len(test_stories)
print('Accuracy: {0}'.format(acc))
print('5 most prob. accuracy: {0}'.format(acc_5))
