#import required libraries
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#read the input file and convert the characters into lowercase
filename = "shortjokes.csv"
text = open(filename).read().lower()

chars = sorted(list(set(text))) #list of distinct characters from the input file
#mapping from character to index and vice-versa
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
#total number of characters in the input file
no_of_chars = len(text) 
#total number of unique characters in the input file
vocab_size = len(chars)

# preprocessing of data
seq_length = 100
step = 100
#data 
X_data = [] 
Y= []
for i in range(0, n_chars - seq_length, step):
    input = text[i:i + seq_length] 
    output = raw_text[i + seq_length]
    X_data.append([char_to_int[char] for char in input])
    Y.append(char_to_int[output])
no_of_patterns = len(X_data)
# reshape X to be [samples, time steps, features] as LSTM requires input in that format
X = numpy.reshape(X_data, (no_of_patterns, seq_length, 1)) #reshape the input
X = X / float(vocab_size) #normalize input to values between 0 and 1
# one hot encode the output variable
y = np_utils.to_categorical(Y)

# define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="best_weights_1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

#load the weights for text generation
filename = "best_weights_1.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


#function to generate text.
'''
Pick a random seed and number og characters for the model to generate. Input to model = seed seq and it outputs a character
Iteratively, update the seed sequence to include the generated character and trim off the first character
'''
def generate_text(seed,chars):
# pick a random seed
    for k in range(seed):
        start = numpy.random.randint(0, len(X)-1)
        start_seed = X_data[start]
        print ("\"",''.join([int_to_char[value] for value in start_seed]),"\"")
        # generate characters
        for i in range(chars):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(vocab_size)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            sys.stdout.write(result)
            start_seed.append(index)
            start_seed = start_seed[1:len(start_seed)]


#call function to generate output
generate_text(100,50)
