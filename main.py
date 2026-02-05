import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import nltk
from log_code import Logger
logger = Logger.get_logs('main')
import warnings
warnings.filterwarnings('ignore')
import string
string.punctuation
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from lemmatization import LEMMA
from embedding import EMBEDDINGS
from RNN import BIDIRECTION
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SENTIMENT:
    def __init__(self,path):
        try:
            logger.info(f'logger started successfully.....')
            self.path=path
            self.df=pd.read_csv(self.path)
            self.df = self.df.iloc[:10000]
            logger.info(f'Dataset:{self.df.sample(5)}')
            logger.info(f'values count:{self.df["sentiment"].value_counts()}')
            logger.info(f'shape:{self.df.shape}')
            logger.info(f'column names:{self.df.columns}')
            logger.info(f'null values:{self.df.isnull().sum()}')
            self.df['sentiment']=self.df['sentiment'].map({'negative':0,'positive':1}).astype(int)
            logger.info(f'data:{self.df.sample(5)}')
            self.lemma=LEMMA(self.df['review'])
            self.lem=WordNetLemmatizer()
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def clean_data(self):
        try:
            self.cleaned_data=self.lemma.data_cleaning()
            #print(self.cleaned_data)
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def word2vec(self):
        try:
            self.embedding = EMBEDDINGS(self.cleaned_data)  # pass data here
            self.input_vectors,self.dic_size,self.maxlen= self.embedding.vectors()  # call vectors()
            print(self.input_vectors)
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def data_splitting(self):
        try:
            self.train_ind_data = self.input_vectors[:7000]
            self.valid_ind_data = self.input_vectors[7000:9000]
            self.test_ind_data = self.input_vectors[9000:]

            self.train_dep_data = self.df['sentiment'][:7000]
            self.valid_dep_data = self.df['sentiment'][7000:9000]
            self.test_dep_data = self.df['sentiment'][9000:]

            logger.info(f'training independent:{self.train_ind_data.shape}')
            logger.info(f'validation independent:{self.valid_ind_data.shape}')
            logger.info(f'testing independent:{self.test_ind_data.shape}')

            logger.info(f'training dependent:{self.train_dep_data.shape}')
            logger.info(f'validation dependent:{self.valid_dep_data.shape}')
            logger.info(f'testing dependent:{self.test_dep_data.shape}')
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def bidirectional_rnn(self):
        try:
            self.train_dep_label=np.array(self.train_dep_data).reshape(-1,1)
            self.valid_dep_label=np.array(self.valid_dep_data).reshape(-1,1)
            self.bidirection=BIDIRECTION()
            self.a=self.bidirection.rnn_model(self.dic_size,self.maxlen,self.train_ind_data,self.train_dep_label,self.valid_ind_data,self.valid_dep_label)
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def testing_data(self):
        try:
            with open('review.pkl','rb') as f:
                self.m=pickle.load(f)
                self.labels = ['negative', 'positve']
                review = ['The product is very good I recommend other to use it']
                self.text = review[0].lower()
                self.text = ''.join([i for i in self.text if i not in string.punctuation])
                self.text = ' '.join([self.lem.lemmatize(i) for i in self.text.split() if i not in stopwords.words('english')])
                self.v = [one_hot(i, self.dic_size) for i in [self.text]]
                self.p = pad_sequences(self.v, maxlen=self.maxlen, padding='post')
                print(self.labels[np.argmax(self.m.predict(self.p))])
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

if __name__ == '__main__':
    try:
        path=r'C:\Users\sravs\Downloads\sentiment_analysis\IMDB Dataset.csv'
        obj = SENTIMENT(path)
        obj.clean_data()
        obj.word2vec()
        obj.data_splitting()
        obj.bidirectional_rnn()
        obj.testing_data()
    except Exception:
        exc_type, exc_msg, exc_line = sys.exc_info()
        logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")