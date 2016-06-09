'''
Trains a RNN on medical diagnosis of diseases dataset
   data is obtained scrapping from various online sources
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

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, Callback
from utils import create_vectors_dataset

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


RNN = recurrent.LSTM
NUM_HIDDEN_UNITS = 312
BATCH_SIZE = 32
EPOCHS = 30
DROPOUT_FACTOR = 0.5
print('RNN / HIDDENS = {}, {}'.format(RNN, NUM_HIDDEN_UNITS))

max_len = 500
word_vec_dim = 300

training_set_file = 'data/training_set.dat'
test_set_file = 'data/test_set.dat'

training_set_file, test_set_file = input_files
train_word_file, test_word_file = vector_files

train_stories = pickle.load(open(training_set_file,'r'))
test_stories = pickle.load(open(test_set_file,'r'))

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q) for fact,q in train_stories]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q) for fact,q in test_stories]

vocab = sorted(reduce(lambda x, y: x | y, (set(story + [answer]) for story, answer in train_stories + test_stories)))

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _ in train_stories + test_stories)))


print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

answer_vocab = sorted(reduce(lambda x, y: x | y, (set([answer]) for _, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
answer_dict = dict((word, i) for i, word in enumerate(answer_vocab))
print('Answers dict len: {0}'.format(len(answer_dict)))

inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)


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

NUM_DATA_TRAIN = inputs_train.shape[0]
NUM_DATA_TEST = inputs_test.shape[0]

random_choice_train = np.random.choice(inputs_train.shape[0], NUM_DATA_TRAIN, replace=False)
X = inputs_train[random_choice_train]
Y = answers_train[random_choice_train]

random_choice_test = np.random.choice(inputs_test.shape[0], NUM_DATA_TEST, replace=False)
tX = inputs_test[random_choice_test]
tY = answers_test[random_choice_test]

print('Training')

base_dir = '.'
checkpointer = ModelCheckpoint(filepath=base_dir + '/models/weights_{0}_{1}_drop_{2}.hdf5'.format(
                'GRU',str(NUM_HIDDEN_UNITS),str(DROPOUT_FACTOR)), verbose=1, save_best_only=True)

#Logger class
class saveMetrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        with open(base_dir + '/logs/log_{0}_{1}_drop_{2}.txt'.format(
                'GRU',str(NUM_HIDDEN_UNITS),str(DROPOUT_FACTOR)),'a') as fil:
            fil.write(str(logs.get('loss')) + ' ' + 
                str(logs.get('acc')) + ' ' + str(logs.get('val_loss')) + ' '
             + str(logs.get('val_acc')) + '\n')

saver = saveMetrics()            
#TODO: Careful with validation split -> can leave diseases never seen by the network
model.fit(X, Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05,
                                    callbacks=[checkpointer,saver])
loss, acc = model.evaluate(tX, tY, batch_size=BATCH_SIZE)

print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
