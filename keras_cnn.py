from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence

import numpy as np
# import theano

from collections import Counter
import linecache, sys, math, re

#
# foundation paper  :   http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
# codebase          :   https://github.com/yoonkim/CNN_sentence/blob/master/conv_net_sentence.py
#

# parameters:
np.random.seed(31337)

# max # of tokens we will accept in the input vector
maxlen = 100

# num of training examples per batch
batch_size = 32

# dimension of embeddings produced from word input vectors
embedding_dims = 50

# number of training epochs
nb_epoch = 2

emb_index = dict()

num_toks = []
num_examples =0

def load_vocab_index():
    f = open('amazon_auto_beauty_sport_train.vocab',"r")

    line_num=1
    for a_line in f:
        toks = a_line.split(" ")
        emb_index[toks[0]] = line_num
        line_num += 1

# creates NP vectors of size max(toks, max_len)
def get_token_indicies(toks,tok_not_found=0, max_index=5000):

    return_val = []

    for a_tok in toks:

        if (len(return_val) <= maxlen):
            idx = emb_index.get(a_tok)

            if (idx == None or idx >= max_index):
                idx = tok_not_found

            return_val.append(idx)
        else:
            break

    return return_val

def get_embeding(word):
    line_num = emb_index.get(word)

    # if word doesn't have an embedding, return random vector
    if (line_num == None):
        emb = np.random.random_sample(size=embedding_dims)

    else:
        emb = linecache.getline('amazon_auto_beauty_sport_train.vectors', line_num)
        toks = emb.strip().split(" ")
        emb = toks[1:]

        for k,v in enumerate(emb):
            emb[k] = float(v)

    return np.array(emb)

def make_embedding_vector(toks):

    return_val = np.array([])

    val = 0
    for i in range(maxlen):
        if (i <= len(toks)-1):
            emb = get_embeding(toks[i])
        else:
            emb = np.array(np.random.random_sample(size=embedding_dims))

        return_val = np.concatenate((emb, return_val), axis=0)
        val = i

    return return_val

# via https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
def make_nn (nb_classes):

    # number of filters applied in 1D convolution
    nb_filter = 500

    # width of filters
    filter_length = 3

    # number of hidden nodes in full connected layer
    hidden_dims = 250

    model = Sequential()

    # 1) max_features - (words * emb dim)
    # 2) embedding_dims - # of dims in the embedding
    # 3) input_length=maxlen - max # of words in input seq

    model.add(Embedding((embedding_dims * maxlen), embedding_dims, input_length=maxlen))
    model.add(Dropout(0.25))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1
                            ))


    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=2))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims, activation='tanh'))
    model.add(Dropout(0.25))

    # We project onto a 3 unit output layer, and squash it with a sigmoid:
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  class_mode='categorical')

    return model

def make_vector (tok_arr):
    num_toks.append(len(toks))

    #print ("{} max toks so far".format(max_toks))

    if (label == 0):
        label_arr = [1,0,0]
    elif (label == 1):
        label_arr = [0,1,0]
    elif (label == 2):
        label_arr = [0,0,1]

    # peel off the training label
    #trainy_vals.append(label_arr)

    # create the embedding vector for the entire document
    #tok_embs = make_embedding_vector(toks)

    # another fun detail: the indicies of the words cannot exceed the max length of the vector (word emb * max toks)
    # so if you have max toks = 100, and word embed
    example_vec = get_token_indicies(toks,max_index=5000)

    #print ("{}".format(toks))
    #print ("{}".format(tok_embs))

    #print ("target size {} : padded single doc vector shape {}".format((maxlen * embedding_dims), tok_embs.shape))

    #all_vecs.append(example_vec)
    #print ("tok_embs shape : {}".format(tok_embs.shape))
    #print ("all vector shape : {}".format(all_vecs.shape))

    return [example_vec, label_arr]

if __name__ == "__main__":

    trainy_vals=[]
    testy_vals=[]

    file_train_data = {'auto_train_lines.csv':0, 'beauty_train_lines.csv':1 , 'sport_train_lines.csv':2}
    file_test_data = {'auto_test_lines.csv':0, 'beauty_test_lines.csv':1, 'sport_test_lines.csv':2}

    load_vocab_index()

    all_vecs = []
    num_examples = 0

    #
    # for each training file
    #
    for train_file, label in file_train_data.items():

        print ("processing training file : {} {}".format(train_file, label))

        #
        # for the first n examples (speeds up debugging data proc code)
        #
        for i in range(1,101,1):
            a_sent = linecache.getline(train_file, i)

            # split out tokens
            toks = re.split('\W+',a_sent)
            X_vec, Y_vec = make_vector(toks)
            all_vecs.append(X_vec)
            trainy_vals.append(Y_vec)

    print ("num examples : {}".format(num_examples))

    print ("unpadded arr len {}".format(len(all_vecs)))
    # split the oen giant vector into 5000-dim vectors per example
    all_vecs = sequence.pad_sequences(all_vecs, maxlen=maxlen)
    print ("padded shape {}".format(all_vecs.shape))

    for a_vec in all_vecs:
        print (a_vec.shape)

    # check to make sure we got the right number of examples after the padding
    nb_samples = all_vecs.shape[0]
    shape_1 = all_vecs.shape[1]
    print ("{} samples  : shape 0".format(nb_samples))
    print ("{}          : shape 1".format(shape_1))

    for idx in range(0):
        # print ("{} : LX {} {} \nLY {} {} ".format(idx, len(X_train[idx]), X_train[idx], len(y_train[idx]), y_train[idx]))
        print ("{} : train SH1 {} y val SH1 {} ".format(idx, all_vecs[idx], trainy_vals[idx]))

    max_toks = np.max(num_toks)

    print ("{} training instances".format(len(all_vecs)))

    print ("FINAL {} max toks".format(max_toks))
    print ("FINAL {} min toks".format(np.min(num_toks)))
    print ("FINAL {} mean toks".format(np.mean(num_toks)))
    print ("FINAL {} median toks".format(np.median(num_toks)))

    print (Counter(num_toks))

    import theano
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'

    model = make_nn(len(file_train_data))
    #plot(model, to_file='model.png')

    print ('fitting model')
    model.fit(all_vecs,
              trainy_vals,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              verbose=1,
              show_accuracy=True,
              validation_split=0.1)

    print ("Done fitting model")

    score = model.evaluate(testx_vals, testy_vals)

    print ("Evaluated model")