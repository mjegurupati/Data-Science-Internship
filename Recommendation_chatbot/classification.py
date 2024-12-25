# Importing the libraries

import pickle
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('queries.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(40):
  #  review = re.sub('[^a-zA-Z]', ' ', dataset['Queries'][i])
    review = dataset['Queries'][i].lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#print(corpus)

# Creating the Bag of Words model
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
a1=accuracy_score(y_test, y_pred)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred=neigh.predict(X_test)
a2=accuracy_score(y_test, y_pred)



if(a1>a2):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y) 
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
else:
    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(neigh, file)


    

#print(model_predict('hospitals fdbgrbvdvhubfv ehr'))
#print(model_predict('Suggest a laptop dfebhmku,cdcdhbsdvyvhnfbi'))
    
    
  
        