import torch
import pickle

from setup import *
import preprocessing as P
from models import *
from utils import *

def tokenize_data(X, pretrained_model_name, max_length=MAX_SEQUENCE_LENGTH):
    print(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False)
    encoding = tokenizer.encode_plus(
        X,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt',
    )

    return {
      'X': X,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_masks': encoding['attention_mask'].flatten()
    }
    

def extract_features_ML(X, fe):
    if fe == 'TF-IDF':
        with open(TFIDF_PATH, 'rb') as file:
            tfidf = pickle.load(file)
        return tfidf.transform([X])
    elif fe == 'fastText':
        raise ValueError('Hiện tại chưa code xong phương pháp trích xuất đặc trưng fastText cho ML')
    else:
        raise ValueError('Hiện tại chưa code xong phương pháp trích xuất đặc trưng GloVe cho ML')


def predict(X, method, model, fe, st):
    # Preprocessing
    X_preprocessed = P.preprocess(X)

    # Feature extracting
    if method == 'Machine Learning':
        X_features = extract_features_ML(X_preprocessed, fe)
        model = load_model_ML(model)
        y_pred = model.predict(X_features)
    else:
        pretrained_model_name = get_backbone_name(fe)
        model = load_model_DL(model, pretrained_model_name, fe)
        X_features = tokenize_data(X_preprocessed, pretrained_model_name)
        _, _, pred = get_prediction(model, X_features['input_ids'], X_features['attention_masks'])
        y_pred = pred.cpu().numpy()

    return return_label(y_pred)
