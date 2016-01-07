from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adagrad, SGD

import numpy as np
import math

print ("Start")

model = Sequential()
model.add(Dense(input_dim=1, output_dim=40, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(input_dim=40, output_dim=1, init='uniform'))
model.add(Activation('linear'))

print ("Defined model")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
# adagrad = Adagrad(lr=0.01, epsilon=1e-6)

print ("Created optimizer")

model.compile(loss='mean_squared_error', optimizer=sgd)

print ("Compiled model")

np.random.seed(1000) # for repro

num_examples = 4000
test_pct = 0.80
num_test = test_pct * num_examples

all_x = np.float32(np.random.uniform(-2*math.pi, 2*math.pi, (1, num_examples))).T
np.random.shuffle(all_x)
trainx_vals = all_x[:num_test]
testx_vals = all_x[num_test:]
trainy_vals = np.sin(trainx_vals)
testy_vals = np.sin(testx_vals)

#print trainy_vals

print("Created training data")

model.fit(trainx_vals, trainy_vals, nb_epoch=20000)

print ("Fit model")

score = model.evaluate(testx_vals, testy_vals)

print ("Evaluated model")

print score