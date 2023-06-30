# from setup import *
import preprocessing as P

import streamlit as st

st.set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>DỰ ĐOÁN NGÀNH NGHỀ TỪ MÔ TẢ CÔNG VIỆC</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Job description
st.markdown("<h3 style='text-align: center;'>MÔ TẢ CÔNG VIỆC:</h3>", unsafe_allow_html=True)
input_method = st.radio("Mô tả công việc:", ('Tải file', 'Nhập từ bàn phím'), horizontal=True, label_visibility="collapsed")
if input_method == 'Tải file':
    inputs = st.file_uploader("Nhập các mô tả công việc", type=['txt'], key='input_method', label_visibility="collapsed")
elif input_method == 'Nhập từ bàn phím':
    inputs = st.text_area("Nhập các mô tả công việc", label_visibility="collapsed")
st.markdown("<hr>", unsafe_allow_html=True)

# Choose method
st.markdown("<h3 style='text-align: center; font-weight: bold;'>PHƯƠNG THỨC DỰ ĐOÁN: </h3>", unsafe_allow_html=True)
st.markdown("""
    <style>
    .stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)
method = st.radio('Phương thức dự đoán', ('Machine Learning', 'Deep Learning'), horizontal=True,  label_visibility="collapsed")
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
# Choose model
with col1:
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>MÔ HÌNH: </h3>", unsafe_allow_html=True)
    if method == 'Machine Learning':
        model_options = ('Linear Regression', 'Stochastic Gradient Descent','Support Vector Machine')
        model = st.selectbox("Mô hình:", model_options)
    elif method == 'Deep Learning':
        model_options = ('Neural Network', 'TextCNN','Bi-LSTM', 'Bi-GRU')
        model = st.selectbox("Mô hình:", model_options)


# Choose feature extractor or pretrained model
with col2:
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>TRÍCH XUẤT ĐẶC TRƯNG / PRETRAINED-MODEL: </h3>", unsafe_allow_html=True)
    if method == 'Machine Learning':
        fe_options = ('TF-IDF', 'fastText','GloVe')
        fe = st.selectbox("Phương thức:", fe_options)
    elif method == 'Deep Learning':
        fe_options = ('phoBERT', 'XLMBert','DistilBert')
        fe = st.selectbox("Mô hình pretrained:", fe_options)

st.markdown("<hr>", unsafe_allow_html=True)

# Get predictions
# predictions = predicting.get_prediction(inputs, model, fe)


# if st.button('Dự đoán'):
    # st.write(setup.ALL_LABELS)