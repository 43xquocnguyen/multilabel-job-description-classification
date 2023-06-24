import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from googletrans import Translator
# from vncorenlp import VnCoreNLP
from itertools import chain
import re
from unidecode import unidecode
from transformers import AutoTokenizer, BertModel
from tqdm.auto import tqdm
import tensorflow as tf
import torch
import preprocessing as P
import pickle

st.set_page_config(layout="wide")

def return_label(y):
    y = y.flatten()
    job_labels = []
    for i in range(0, NUM_LABELS):
        if y[i] == 1:
            job_labels.append(labels[i])

    return job_labels

def threshold(y):
    return (y >= 0.5) + 1 - 1

def predicting(model, features, type):
    if type == 'ML':
        preds = model.predict(features)
        preds = [return_label(y) for y in preds]
    else:
        preds = model.predict(features)
        preds = [threshold(a) for a in preds]
        preds = [return_label(y) for y in preds]

    return preds
# from vncorenlp import VnCoreNLP
# vncorenlp = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
data = []
# cities_lst = pd.read_csv('data/provinces.csv').Provinces
# industries_lst = pd.read_csv('data/job_labels.csv')['0']
# levels_lst = pd.read_csv('data/levels.csv').name
# dict_job = dict(industries_lst)

# Predict
labels = pd.read_csv('data/job_labels.csv')
labels = labels['0'].tolist()
NUM_LABELS = len(labels)

path_DL1 = 'model/XLMBert_bGRU.h5'
path_DL2 = 'model/XLMBert_bLSTM.h5'
path_DL3 = 'model/XLMBert_customized.h5'
path_DL4 = 'model/XLMBert_TextCNN.h5'
 
path_ML1 = 'model/LR_ML.pkl'
path_ML2 = 'model/SGD_ML.pkl'
path_ML3 = 'model/SVC_ML.pkl'

MAX_SEQUENCE_LENGTH = 200
MODEL_NAME = 'bert-base-multilingual-cased'
# MODEL_NAME = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)#, output_hidden_states=True)

tags_index = []
tags = []
# Title
st.markdown("<h1 style='text-align: center;'>Predict label from description</h1>", unsafe_allow_html=True)
# st.title("Thông tin tuyển dụng", style="text-align: center")

# st.write("Tiêu đề:")
# title_job = st.text_input("Tiêu đề:", placeholder = 'Title', label_visibility = "collapsed")

# comp_name_col, comp_city_col = st.columns([0.7,0.3])
# with comp_name_col:
#     st.write("Tên công ty:")
#     comp_name = st.text_input("Tên công ty:", placeholder = 'Company name', label_visibility = "collapsed")
# with comp_city_col:
#     st.write("Địa chỉ: ")
#     city = st.multiselect("Địa chỉ: ", cities_lst, label_visibility = "collapsed")

# level_col, exp_years_col = st.columns([0.4,0.6])
# with exp_years_col:
#     st.write("Kinh nghiệm:")
#     input_exp_years = st.radio("Kinh nghiệm:", ('Từ ... đến...', 'Không yêu cầu'), horizontal = True, label_visibility = "collapsed")
#     if input_exp_years == 'Từ ... đến...':
#         min_year, max_year = st.slider('Kinh nghiệm:', 0, 50, (5, 15), key='exp_year', label_visibility = "collapsed")
# with level_col:
#     st.write("Cấp độ: ")
#     level = st.selectbox('Cấp độ: ',('a','b','c','d','e'), label_visibility = "collapsed")



# st.write('Mức lương:')
# min_sal_col, max_sal_col =st.columns(2)

# with min_sal_col:
#     min_sal = st.number_input("Tối thiểu: ", min_value = 1.0 , step = 0.1)
# with max_sal_col:
#     max_sal = st.number_input("Tối đa: ", min_value = min_sal, step = 0.1)
col1, col2 = st.columns([0.6,0.4], gap = 'large')
with col1:
    st.markdown("<div style='font-weight: bold;'>Input description:</div>", unsafe_allow_html=True)
    input_desc = st.radio("Mô tả công việc:", ('Tải file', 'Nhập từ bàn phím'), horizontal = True, label_visibility = "collapsed")
    if input_desc == 'Tải file':
        uploaded_desc_file = st.file_uploader("Input your text", type=['txt', 'pdf','docs'], key = 'input_desc', label_visibility = "collapsed")
    if input_desc == 'Nhập từ bàn phím':
        raw_desc_text = st.text_area("Input your text", label_visibility = "collapsed") 

    st.markdown("<div style='font-weight: bold;'>Choose method: </div>", unsafe_allow_html=True)
    input_method = st.radio("Chọn phương thức:", ('Machine Learning', 'Deep Learning'), label_visibility = "collapsed")



# st.write("Yêu cầu công việc:")
# input_request = st.radio("Yêu cầu công việc:", ('Tải file', 'Nhập từ bàn phím'), index=1, horizontal = True, label_visibility = "collapsed")

# if input_request == 'Tải file':
#     uploaded_rq_file = st.file_uploader("Input your text", type=['txt', 'pdf','docs'],key='input_request', label_visibility = "collapsed")
# if input_request == 'Nhập từ bàn phím':
#     raw_rq_text = st.text_area("Input your text", label_visibility = "collapsed")
with col2:
#Chọn model trainning
    st.markdown("<div style='font-weight: bold;'>Choose model:</div>", unsafe_allow_html=True)
    if input_method == 'Machine Learning':
        method ='ML'    
        input_model_ML = st.radio("Model:", ('LN', 'SGD','SVC'), horizontal = True, label_visibility = "collapsed")
        if input_model_ML == 'LN':
            path = path_ML1
        if input_model_ML == 'SGD':
            path = path_ML2
        if input_model_ML == 'SVC':
            path = path_ML3
        

    if input_method == 'Deep Learning':
        method ='DL'
        input_model_DL = st.radio("Model:", ('bGRU', 'bLSTM','customized','TextCNN'), horizontal = True, label_visibility = "collapsed")
        if input_model_DL == 'bGRU':
            path = path_DL1
        if input_model_DL == 'bLSTM':
            path = path_DL2
        if input_model_DL == 'customized':
            path = path_DL3
        if input_model_DL == 'TextCNN':
            path = path_DL4
    
    btn = st.button('Predict', type="primary")

    if btn:
        if method == 'ML':
            with open(path, 'rb') as file:
                model = pickle.load(file)
        else:
            model = load_model(path)

        #get value predict
        if input_desc == 'Tải file':
            if uploaded_desc_file:
                data = [uploaded_desc_file.read().decode("utf-8")]
        
        if input_desc == 'Nhập từ bàn phím':
                data = [raw_desc_text]

        if data:
            if method == 'DL':
                features = P.extract_feature_for_DL(data, tokenizer, max_length=MAX_SEQUENCE_LENGTH, encoder=bert)
                features = np.array(features)
            else:
                features = P.extract_feature_for_ML(data)

            preds = predicting(model, features, method)
            st.write(preds)
        btn = False
        




    

# features = P.extract_feature_for_ML(data)

# tags = [dict_job[key] for key in tags_index]
# st.write("Lĩnh vực: ")
# industry = st.multiselect("Lĩnh vực: ", industries_lst, default=tags, label_visibility = "collapsed")

# finish_button = st.columns(4)
# with finish_button[3]:
#     st.button('Hoàn thành', type="primary")
	
