# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:40:51 2024

@author: Ajay
"""

import pandas as pd
import pickle 
import streamlit as st
import re
import string

def clean_word(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

loaded_vectorization = pickle.load(open('C:/Users/Ajay/Desktop/Coaaps-Project/vector.sav', 'rb'))
loaded_model = pickle.load(open('C:/Users/Ajay/Desktop/Coaaps-Project/Rumor_mill.sav', 'rb'))

def output_label(n):
    if n == 0:
        return "Fake Rumor"
    elif n == 1:
        return "True Rumor"
    else:
        return "Unknown"  # Add a default case for unknown predictions

def manual_testing(news):
    testing_news = {"text": [news]}  # Wrap the news text inside a list
    new_def_test = pd.DataFrame.from_dict(testing_news)  # Create DataFrame from dictionary
    new_def_test['text'] = new_def_test['text'].apply(clean_word)
    new_x_test = new_def_test['text']
    new_xv_test = loaded_vectorization.transform(new_x_test)
    pred_DT = loaded_model.predict(new_xv_test)
    
    return output_label(pred_DT[0])


def main():
    st.title('Rumor Tracking System')
    text = st.text_input('Enter your news text here: ')
    rummors = ''
    if st.button('Predicting News Result'):
        if text.strip() != '':
            rummors = manual_testing(text)
        else:
            st.error("Please enter some text!")  # Error handling for empty input
    st.success("Prediction: " + rummors)

if __name__ == '__main__':
    main()
