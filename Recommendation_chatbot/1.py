# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:42:53 2020

@author: Mounica Jegurupati
"""
import re
import numpy as np
import pandas as pd
import pickle

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
    
print(pickle_model)

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def model_predict(new_review):
    cv = CountVectorizer()
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower() 
    print(new_review)
    new_review = new_review.split()
    print(new_review)
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    print(new_review)
    new_review = ' '.join(new_review)
    print(new_review)
    new_corpus = [new_review]
    print(new_corpus)
    new_X_test = cv.fit_transform(new_corpus).toarray()
    print(new_X_test)
    new_y_pred = pickle_model.predict(new_X_test)
    print(new_y_pred)
    
model_predict('Hello World')