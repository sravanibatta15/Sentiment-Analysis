import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import nltk
from log_code import Logger
logger = Logger.get_logs('rnn')
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Flatten,Embedding,Masking,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu,sigmoid,softmax
import pickle

class BIDIRECTION:
    def __init__(self):
        try:
            self.model=Sequential()
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def rnn_model(self,dic_size,maxlen,train_ind,train_dep,valid_ind,valid_dep):
        try:
            self.model.add(Embedding(input_dim=dic_size, output_dim=5, input_length=maxlen))
            self.model.add(Masking(mask_value=0.0))

            self.model.add(Bidirectional(SimpleRNN(units=3, return_sequences=True, name='Hidden_layer_1')))
            self.model.add(Bidirectional(SimpleRNN(units=4, return_sequences=True, name='Hidden_layer_2')))
            self.model.add(Bidirectional(SimpleRNN(units=5, return_sequences=False, name='Hidden_layer_3')))
            self.model.add(Dense(units=1, activation='sigmoid', name='output_layer'))

            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.model.summary()

            self.model.fit(train_ind, train_dep, epochs=20, batch_size=100,validation_data=(valid_ind, valid_dep))
            self.model.summary()

            with open('review.pkl', 'wb') as f:
                pickle.dump(self.model, f)

        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")