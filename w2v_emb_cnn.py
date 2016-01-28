from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence

import numpy as np

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

# max dim of embedding vector
max_dim_emb_vector = embedding_dims * maxlen

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

    emb = []
    # if word doesn't have an embedding, return random vector
    if (line_num == None):
        emb = np.random.random_sample(size=embedding_dims)

    else:
        emb_str = linecache.getline('amazon_auto_beauty_sport_train.vectors', line_num)
        toks = emb_str.strip().split(" ")
        # emb = toks[1:]

        for a_val in toks[1:]:
            emb.append(float(a_val))

    return np.array(emb).astype('float32')

#
# given an array of tokens, and a numeric label
#
# return a single np array with all of the w2v embeddings concatenated for the tokens
# and a one hot vector for the label
#
def make_w2v_embedding_vector(toks,label):

    num_toks.append(len(toks))

    return_val = np.array([])

    if (label == 0):
        label_arr = [1,0,0]
    elif (label == 1):
        label_arr = [0,1,0]
    elif (label == 2):
        label_arr = [0,0,1]

    val = 0
    # for all of the tokens within the max length:
    #   get the embedding for each token
    #   if there's no embedding for the token, create a random vector
    for i in range(maxlen):
        if (i <= len(toks)-1):
            emb = get_embeding(toks[i])
        else:
            emb = np.array(np.random.random_sample(size=embedding_dims))

        return_val = np.concatenate((emb, return_val), axis=0)
        val = i

    #return_val = sequence.pad_sequences(return_val, maxlen=maxlen)
    return_val = return_val.reshape(return_val.shape[0],1,1)

    return [return_val, label_arr]

# via https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
def make_w2v_embedding_nn (nb_classes):

    model = Sequential()

    # 1) max_features - (words * emb dim)
    # 2) embedding_dims - # of dims in the embedding
    # 3) input_length=maxlen - max # of words in input seq

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=100,
                            filter_length=3,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=500,
                            #input_dim=max_dim_emb_vector,
                            #input_length=max_dim_emb_vector
                            input_shape=(5000,5000)
                            ))


    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=2))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(250, activation='tanh'))
    model.add(Dropout(0.25))

    # We project onto a 3 unit output layer, and squash it with a sigmoid:
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  class_mode='categorical')

    return model

def make_keras_embedding_nn (nb_classes):

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

def make_word_index_vector (tok_arr, label):
    num_toks.append(len(tok_arr))

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
    example_vec = get_token_indicies(tok_arr,max_index=5000)

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

    # each file is a separate class, # of files = # of classes
    file_train_data = {'auto_train_lines.csv':0, 'beauty_train_lines.csv':1 , 'sport_train_lines.csv':2}
    file_test_data = {'auto_test_lines.csv':0, 'beauty_test_lines.csv':1, 'sport_test_lines.csv':2}

    load_vocab_index()

    all_vecs = []
    #num_examples = 0

    #
    # for each training file
    #
    for train_file, label in file_train_data.items():

        print ("processing training file : {} {}".format(train_file, label))

        count_lines = open(train_file)
        #num_lines = sum(1 for line in count_lines)

        num_lines = 100

        count_lines.close()

        print ("{} lines in file {}".format(num_lines,train_file))

        #
        # for the first n examples (speeds up debugging data proc code)
        #
        for i in range(1,num_lines,1):
            a_sent = linecache.getline(train_file, i)

            # split out tokens
            toks = re.split('\W+',a_sent)
            X_vec, Y_vec = make_w2v_embedding_vector(toks, label)

            X_vec = sequence.pad_sequences(X_vec, maxlen=5000)

            all_vecs.append(X_vec)
            trainy_vals.append(Y_vec)

            num_examples += 1

            if (i % 1000 == 0):
                print ('.', end="")

        print("")

    print ("num examples : {}".format(num_examples))
    print ("unpadded arr len {}".format(len(all_vecs)))

    # split the oen giant vector into 5000-dim vectors per example
    # all_vecs = sequence.pad_sequences(all_vecs, maxlen=5000)
    # print ("padded shape {}".format(all_vecs.shape))

    # print ("shape 0: {}, shape 1: {}".format(all_vecs.shape[0], all_vecs.shape[1]))
    #train_X = all_vecs.reshape(all_vecs.shape[0], 1 ,all_vecs.shape[1])

    num_train = int(len(all_vecs) * 0.7)
    train_X = all_vecs[:num_train]
    test_X = all_vecs[num_train:]

    #train_X = sequence.pad_sequences(train_X, maxlen=maxlen)
    #test_X = sequence.pad_sequences(test_X, maxlen=maxlen)

    #print ("train X shape : {}".format(train_X.shape))

    train_Y = trainy_vals[:num_train]
    test_Y = trainy_vals[num_train:]

    #for a_vec in all_vecs:
    #    print (a_vec.shape)

    # check to make sure we got the right number of examples after the padding
    # all_vecs = np.array(all_vecs)
    # nb_samples = all_vecs.shape[0]
    # shape_1 = all_vecs.shape[1]
    # print ("{} samples  : shape 0".format(nb_samples))
    # print ("{}          : shape 1".format(shape_1))

    #for idx in range(len(all_vecs)):
    #    print ("{} : train SH0 {} y val SH0 {} ".format(idx, all_vecs[idx].shape(0), trainy_vals[idx].shape(0)))

    max_toks = np.max(num_toks)

    print ("{} training instances".format(len(train_X)))
    print ("{} labels instances".format(len(train_Y)))
    print ("{} testing instances".format(len(test_X)))
    print ("{} label instances".format(len(test_Y)))

    print ("FINAL {} max toks".format(max_toks))
    print ("FINAL {} min toks".format(np.min(num_toks)))
    print ("FINAL {} mean toks".format(np.mean(num_toks)))
    print ("FINAL {} median toks".format(np.median(num_toks)))

    print (len(set([len(a) for a in train_X] + [len(train_Y)])))

    #for a in train_X:
    #    print ("shape X: {}".format(a.shape()))

    #for a in train_Y:
    #    print ("shape Y: {}".format(a.shape()))

    print (Counter(num_toks))

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    print ("train X shape : {}".format(train_X.shape))
    print ("train Y shape : {}".format(train_Y.shape))

    # import theano
    # theano.config.optimizer='fast_compile'
    # theano.config.exception_verbosity='high'

    model = make_w2v_embedding_nn(len(file_train_data))
    #plot(model, to_file='model.png')

    print ('fitting model')
    model.fit(train_X,
              train_Y,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              verbose=1,
              show_accuracy=True,
              validation_split=0.1)

    model.save_weights("amzn_weights_local_cnn.keras", overwrite=True)

    print ("Done fitting model")

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    print ("test X shape : {}".format(test_X.shape))
    print ("test Y shape : {}".format(test_Y.shape))

    print(model.evaluate(test_X, test_Y))

    print ("Evaluated model")
