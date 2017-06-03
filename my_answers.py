import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    
    # list comprehension,take slices with step size of 1 from input series 
    X = [np.array(series[i:i+window_size]) for i in range(len(series)-window_size)]
    y = [np.array(series[i+window_size]) for i in range(len(series)-window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    # LSTM layer with 5 hidden units, input shape is (window_size, 1), 1 is representing step size of 1
    model.add(LSTM(5, input_shape=(window_size,1)))
    # fully connected layer of size 1 to output numerical value
    model.add(Dense(1))
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text

    # using re module to replace all characters except for alphabetc and ['!', ',', '.', ':', ';', '?'] with space
    import re
    text = re.sub(r'[^a-zA-Z!,.:;?]', ' ', text)
    # reshape extra spaces as a result of last step with only one space character
    text = text.replace('  ',' ')
    # remove as many non-english characters and character sequences as you can 
    

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # number of individual pairs as a function of text size and step size
    pairs = int(np.ceil(len(text)/step_size) - window_size) 
    # list comprehension,take slices from input text with step size input from user 
    inputs = [text[i*step_size:i*step_size+window_size] for i in range(pairs)]
    outputs = [text[i*step_size+window_size] for i in range(pairs)]
    
    return inputs,outputs
