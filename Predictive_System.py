# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:33:05 2024

@author: Ajay
"""

import pandas as pd
import pickle
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
    
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(clean_word)
    new_x_test = new_def_test['text']
    new_xv_test = loaded_vectorization.transform(new_x_test)
    pred_DT = loaded_model.predict(new_xv_test)
    
    return print("\nPrediction: {}".format(output_label(pred_DT[0])))

news = str(input("Enter news text: "))
manual_testing(news)