import pandas as pd
import numpy as np

import re
from unidecode import unidecode
from itertools import chain

from vncorenlp import VnCoreNLP

# Paths
DATA_PATH = '/data/raw_data_2.csv'
VI_DATA_PATH = '/data/vn.csv'
FULL_PATH = '/data/full.csv'
TRAIN_PATH = '/data/train.csv'
VAL_PATH = '/data/val.csv'
TEST_PATH = '/data/test.csv'

ALL_LABELS_PATH = '/data/labels.csv'

STOPWORDS_PATH = '/data/vietnamese-stopwords.txt'

# Stopwords
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    STOPWORDS = file.readlines()
STOPWORDS = [sw.strip() for sw in STOPWORDS if sw.strip()]

# All labels
ALL_LABELS = pd.read_csv(ALL_LABELS_PATH)['0'].tolist()
NUM_LABELS = len(ALL_LABELS)

# VNCoreNLP
vncorenlp = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
