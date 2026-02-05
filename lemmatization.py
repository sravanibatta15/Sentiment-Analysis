import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import nltk
from log_code import Logger
logger = Logger.get_logs('lemma')
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


class LEMMA:
    def __init__(self,data):
        try:
            self.data=data
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def data_cleaning(self):
        try:
            logger.info(f'lemmatization started')
            stop_words = set(stopwords.words('english'))
            lem = WordNetLemmatizer()
            cleaned_reviews = []
            for text_data in self.data:
                text_data = str(text_data)
                cleaned_data = ''
                s = ''
                for i in text_data:
                    if i not in string.punctuation:
                        s += i
                for k in s.split():
                    word = k.lower()
                    if word not in stop_words:
                        lemma = lem.lemmatize(word)
                        cleaned_data += ' ' + lemma
                cleaned_reviews.append(cleaned_data.strip())
            return cleaned_reviews
        except Exception:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")