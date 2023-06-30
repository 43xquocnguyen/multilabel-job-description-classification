from tqdm.auto import tqdm
import torch
import pickle

from setup import *
import preprocessing as P



def tokenize_data(text_data, tokenizer, max_length):
    input_ids = []
    attn_mask = []
    # token_type_ids = []

    for txt_data in tqdm(text_data):

        tokenized = tokenizer.encode_plus(
            txt_data,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(tokenized['input_ids'])
        attn_mask.append(tokenized['attention_mask'])

    return torch.tensor(input_ids), torch.tensor(attn_mask)

def extract_features_ML(X):
    X = [P.preprocess(x) for x in X]
    with open(TFIDF_PATH, 'rb') as file:
        tfidf = pickle.load(file)
    return tfidf.transform(X)


def extract_features(method, X):
    if method == 'Machine Learning':
        X_features = extract_features_ML(X)
    else:
        X_features = tokenize_data(X)

    return X_features