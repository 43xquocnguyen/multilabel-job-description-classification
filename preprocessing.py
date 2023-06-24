# from googletrans import Translator
from vncorenlp import VnCoreNLP
from itertools import chain
import re
from unidecode import unidecode
import numpy as np
from transformers import AutoTokenizer, BertModel
from tqdm.auto import tqdm
import tensorflow as tf
import torch
import underthesea
import pickle

    
with open('./data/vietnamese-stopwords.txt', 'r',encoding='utf-8') as file:
    vi_stopwords = file.readlines()
vi_stopwords = [sw.strip() for sw in vi_stopwords if sw.strip()]

def is_word(element):
    pattern = r'^\w+$'
    return re.match(pattern, element) is not None

# Remove special characters & stopwords
def remove_characters(txt_data):
    cleaned = []
    for token in txt_data:
        if token not in vi_stopwords:
            # if token.replace('_', '').isalpha():
            if is_word(unidecode(token)):
                # cleaned_token = re.sub(r"[-()\"#/@;:<>{}`+=~|!?,]", "", token)
                # cleaned.append(cleaned_token)
                cleaned.append(token)
    return cleaned

def preprocess(txt):
    # Lower case
    txt = txt.lower()
    # Tokenization

    txt = underthesea.word_tokenize(txt)
    txt = [token.replace(' ','_') for token in txt]

    # txt = VnCoreNLP.tokenize(txt)
    # txt = list(chain.from_iterable(txt))

    # Special character removal
    txt = remove_characters(txt)
    txt = list(filter(lambda x: x != '', txt))

    txt = ' '.join(txt).replace('_', ' ')
    return txt.strip()

def tokenize_data(text_data, tokenizer, max_length):
    input_ids = []
    attn_mask = []
    # token_type_ids = []

    for txt_data in tqdm(text_data):

        tokenized = tokenizer.encode_plus(
            txt_data,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        input_ids.append(tokenized['input_ids'])
        attn_mask.append(tokenized['attention_mask'])
        # token_type_ids.append(tokenized['token_type_ids'])

    return torch.tensor(input_ids), torch.tensor(attn_mask)#, torch.tensor(token_type_ids)


def embed(input_ids, attn_mask, encoder):
    with torch.no_grad():
        outputs = encoder(input_ids, attention_mask=attn_mask)
    return outputs[0]


def extract_feature_for_DL(X, tokenizer, max_length, encoder):
    X = [preprocess(x) for x in X]
    ids, attn_mask = tokenize_data(X, tokenizer, max_length=max_length)
    embeddeds = embed(ids, attn_mask, encoder=encoder)
    return embeddeds



def extract_feature_for_ML(X):
    X = [preprocess(x) for x in X]
    with open('model/tfidf_5000_ML.pkl', 'rb') as file:
        tfidf_loaded = pickle.load(file)
    return tfidf_loaded.transform(X)
