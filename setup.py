import pandas as pd
import numpy as np

import re
from itertools import chain
from unidecode import unidecode

from pyvi import ViTokenizer
from vncorenlp import VnCoreNLP

# raw data
DATA_PATH = './data/raw_data_2.csv'
VI_DATA_PATH = './data/vn.csv'

# data paths
FULL_PATH = './data/full.csv'
TRAIN_PATH = './data/train.csv'
VAL_PATH = './data/val.csv'
TEST_PATH = './data/test.csv'

# labels path
ALL_LABELS_PATH = './data/labels.csv'

# stopword path
STOPWORDS_PATH = './data/vietnamese-stopwords.txt'
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    STOPWORDS = file.readlines()
STOPWORDS = [sw.strip() for sw in STOPWORDS if sw.strip()]

# all labels
ALL_LABELS = pd.read_csv(ALL_LABELS_PATH)['0'].tolist()
NUM_LABELS = len(ALL_LABELS)

# VNCoreNLP
# vncorenlp = VnCoreNLP('vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')

# TF-IDF
TFIDF_PATH = 'models/tfidf_word_5000.pkl'

# Tokenizer max_seq_len
MAX_SEQUENCE_LENGTH = 200
MODEL_NAME = 'vinai/phobert-base'
# MODEL_NAME = 'bert-base-multilingual-cased'
# MODEL_NAME = 'distilbert-base-cased'