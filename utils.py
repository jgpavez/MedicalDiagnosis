from __future__ import print_function
from functools import reduce
import tarfile
import numpy as np
import re
import pdb
import pickle
import h5py
from keras.preprocessing.sequence import pad_sequences
from glove import Glove


def check_repeated(name,repeated_list):
    name = name.lower().strip()
    return name if not (name in repeated_list) else repeated_list[name]

def process_title(word):
    return re.sub(r'\W+', ' ', word).strip().lower()

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False,repeated_list=None):
    '''Parse stories

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        spl = line.split(' ', 1)
        if len(spl) > 1:
            nid, line = spl
        else:
            continue
        try:
            nid = int(nid)
        except ValueError:
            pdb.set_trace()
        if nid == 0:
            story = []
        if '\t' in line:
            supporting, a = line.split('\t')
            a = map(process_title,a.split(','))
            options = [] if len(a) == 1 else list(set(a[1:]))
            a = a[0]    
            substory = None
            # Provide all the substories
            if supporting:
                story.append([tokenize(supporting) + [u'.']])
            substory = [x for x in story if x]
            # TODO: I should have done the lower in previous processing steps
            if not substory:
                continue
            data.append((substory, a.lower(), map(lambda x:x.lower(),options)))
        else:
            sent = tokenize(line)
            story.append([sent + [u'.']])
    return data


def get_stories(f, only_supporting=False, max_length=None,repeated_list=None, min_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting,repeated_list=repeated_list)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [[flatten(reversed(story)), answer, options] for story,answer,options in data if not max_length or len(flatten(story)) < max_length]
    # At least two facts
    print(len(data))
    if min_length: 
        data = filter(lambda x: len(x[0]) > min_length, data)
    print(len(data))
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx))  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def get_spacy_vectors(data, answer_dict, story_maxlen, model):
    X = []
    Y = []
    for story,answer in data:
        story = story[:story_maxlen] if len(story) > story_maxlen else story
        x = [model(unicode(w)).vector for w in story]
        X.append(x)
        if not answer_dict is None:
            y = np.zeros(len(answer_dict))
            y[answer_dict[answer]] = 1
            Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen,dtype='float32'),
            np.array(Y))

def get_word_vectors(data, answer_dict, story_maxlen, model):
    X = []
    Y = []
    for story,answer in data:
        story = story[:story_maxlen] if len(story) > story_maxlen else story
        x = [model.word_vectors[model.dictionary[w]] for w in story if 
             not model.dictionary.get(w) is None]
        X.append(x)
        if not answer_dict is None:
            y = np.zeros(len(answer_dict))
            y[answer_dict[answer]] = 1
            Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen,dtype='float32'),
            np.array(Y))
     
def create_vectors_dataset(input_files, vector_files, max_len=500):
    print('Creating word vectors file')

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

    # I need to check also if this exist
    word_vectors_dir = 'word_vectors/glove.42B.300d.txt'
    word_vectors_model = Glove.load_stanford(word_vectors_dir)

    inputs_train, answers_train = get_word_vectors(train_stories, answer_dict, 
                                                   max_len, word_vectors_model)
    inputs_test, answers_test = get_word_vectors(test_stories, answer_dict, max_len,
                                                 word_vectors_model)

    with h5py.File(train_word_file,'w') as train_f:
        _ = train_f.create_dataset('inputs',data=inputs_train)
        _ = train_f.create_dataset('answers',data=answers_train)
    with h5py.File(test_word_file,'w') as test_f:
        _ = test_f.create_dataset('inputs',data=inputs_test)
        _ = test_f.create_dataset('answers',data=answers_test)
        
    return (inputs_train, answers_train),(inputs_test, answers_test)

def save_vectors_dict(input_files):


    # I need to check also if this exist
    filename = 'word_vectors/glove.42B.300d.txt'
    word_vectors_dict = 'word_vectors/glove_dict.hdf5'
    dct = {}
    vectors = array.array('d')

    # Read in the data.
    with io.open(filename, 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            tokens = line.split(' ')

            word = tokens[0]
            entries = tokens[1:]

            dct[word] = i
            vectors.extend(float(x) for x in entries)
            
    print('Saving to hf5 file')
    with h5py.File(word_vectors_dict,'w') as vector_f:
        _ = vector_f.create_dataset('vectors',data=dct)

        