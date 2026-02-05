import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import nltk
from log_code import Logger
logger = Logger.get_logs('word')
import warnings
warnings.filterwarnings('ignore')
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

class EMBEDDINGS:
    def __init__(self,data):
        try:
            self.data=data
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def vectors(self):
        try:
            # a=[]
            # for i in self.data:
            #     if i not in a:
            #         a.append(i)
            # print(f'length:{len(a)}')
            dic_size = 10000
            vec = [one_hot(i, dic_size) for i in self.data]
            t = []
            for i in vec:
                t.append(len(i))
            p=max(t)
            vectors = pad_sequences(vec, maxlen=p, padding='post')
            return vectors,dic_size,p
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")