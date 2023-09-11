#!/usr/bin/env python
# coding: utf-8

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string
import re
import spacy
import time
import pickle
from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import streamlit as st

st.set_page_config(page_title="MNB Classifier", page_icon="📄")

sidebar = st.sidebar

st.markdown("# Review Classification")
sidebar.success("Klasifikasi Review Menggunakan Multinomial Naive Bayes")
sidebar.code(
    """
    Creator: Muhammad Ikhlasul Amal
    Universitas Brawijaya
    """
)
st.info(
    """ ##### Sistem ini akan mengklasifikasi atau mengelompokkan teks review kedalam karakteristik kualitas **satisfaction** atau **freedom from risk**. Selain itu, dilakukan juga penggalian opini atau sentimen pengguna apakah bernilai sentimen **positif** atau **negatif**. Algortima yang digunakan pada sistem ini adalah Multinomial Naive Bayes dan TF-IDF.
    """
)

#Preprocessing
def preprocessing_text(text):
    #Casefolding
    text = text.lower() # lowercase text

    #Remove puncutuation and number
    clean_spcl = re.compile('[/(){}\[\]\|@,;]')
    clean_symbol = re.compile('[^a-zA-Z]')
    #Remove single char
    clean_singl_char = re.compile(r"\b[a-zA-Z]\b")
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub(' ', text)
    text = clean_singl_char.sub('', text)

    #clean stopwords
    stopword = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stopword) # hapus stopword dari kolom review

    #Lemmatization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = nlp(text)
    text = ' '.join(token.lemma_ for token in doc if token in doc) # Extract the lemma for each token and join

    #Tokenizing
    token = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    return token

# Load Model
model_c = pickle.load(open('mnb_model_c.pkl', 'rb'))
model_s = pickle.load(open('mnb_model_s.pkl', 'rb'))

#TF IDF
tfidf_vect = pickle.load(open('tfidf_model.pkl', 'rb'))

def predict_model(model, data):
    predictions = model.predict(data)
    return predictions

st.write('Upload teks review (batch) dengan format **CSV**')
new_data = st.file_uploader('Masukkan file csv', type='csv')
if new_data is not None:
    new_data = pd.read_csv(new_data, names=['Review'])

    new_data['Clean'] = new_data['Review'].apply(preprocessing_text)
    new_data = new_data.astype({"Clean" : 'string'})
    new_data_tfidf = tfidf_vect.transform(new_data['Clean'])

    class_predict_c = predict_model(model_c, new_data_tfidf)
    class_predict_s = predict_model(model_s, new_data_tfidf)
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)
        my_bar.progress(percent_complete + 1)

    temp_c = []
    for value in class_predict_c:
        if value == 1:
            temp_c.append('Satisfaction')
        elif value == 0:
            temp_c.append('Freedom From Risk')
        else:
            temp_c.append('None')

    temp_s = []
    for value in class_predict_s:
        if value == 1:
            temp_s.append('Positif')
        elif value == 0:
            temp_s.append('Negatif')
        else:
            temp_s.append('None')

    time.sleep(0.01)
    st.success("**Done predicting**")
    st.balloons()
    time.sleep(0.01)

    review_temp = new_data['Review'].values.tolist()
    result = pd.DataFrame(np.column_stack([review_temp, temp_c, temp_s]), columns=['Review', 'Characteristic', 'Sentiment'])
    st.write(result)
