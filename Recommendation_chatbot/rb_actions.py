import random
import json
import os
import re
import pickle

from rb_generate_ngrams import ngrammatch
from rb_intents import Intent, IntentComplete




pkl_filename = "pickle_model.pkl"


# Load from file
with open(pkl_filename, 'rb') as file:
    print(file)
    pickle_model = pickle.load(file)

def check_actions(current_intent, attributes, context):
    '''This function performs the action for the intent
    as mentioned in the intent config file'''
    '''Performs actions pertaining to current intent
    for action in current_intent.actions:
        if action.contexts_satisfied(active_contexts):
            return perform_action()
    '''

    context = IntentComplete()
    # return 'action: ' + current_intent.action, context
    return current_intent.action, context

'''Collects attributes pertaining to the current intent'''
def check_required_params(current_intent, attributes, context):

    for para in current_intent.params:
        if para.required == 'True':
            if para.name not in attributes:
                # Example of where the context is born
                # if para.name=='RegNo':
                    # context = GetRegNo()
                # returning a random prompt from available choices.
                return random.choice(para.prompts), context

    return None, context

'''Spellcheck and entity extraction functions go here'''
def input_processor(user_input, context, attributes, intent):

    # uinput = TextBlob(user_input).correct().string

    # update the attributes, abstract over the entities in user input
    attributes, cleaned_input = getattributes(user_input, context, attributes)

    return attributes, cleaned_input

def loadIntent(path, intent):
    import json 
  
# Opening JSON file 
'''    f = open(path,'r') 
  
    # returns JSON object as  
    # a dictionary 
    dat = json.loads(f.read()) 
    intent = dat[intent]
    return Intent(intent['intentname'], intent['parameters'], intent['actions'])  '''
    
def loadIntent(path, intent):
    with open(path) as file_intent:
        dat = json.load(file_intent)
        intent = dat[intent]
        return Intent(intent['intentname'], intent['parameters'], intent['actions'])

'''This function is used to classify the intent'''
def intentIdentifier(clean_input, context, current_intent):
    clean_input = clean_input.lower()
    
    # print("intentIdentifier - clean_input ", clean_input)
  #  print(clean_input)
    clean_input = model_predict(clean_input)
  #  print(clean_input)

    '''TODO : YOUR CODE HERE TO CLASSIFY THE INTENT'''
    #clean_input = 1
     
    
    # Scoring Algorithm, can be changed.
 #   scores = ngrammatch(clean_input)

    # choosing here the intent with the highest score
#    scores = sorted_by_second = sorted(scores, key=lambda tup: tup[1])
    # print('intentIdentifier - scores ', scores)
    # clean_input = "search"

    if current_intent is None:
        if clean_input == 1:
            current_intent = loadIntent('params/newparams1.cfg', 'hospital')
        elif clean_input == 0:
            current_intent = loadIntent('params/newparams1.cfg', 'OrderGame')
      #  print("intentIdentifier - current_intent ", current_intent.name)
        return current_intent
    else:
        # If current intent is not none, stick with the ongoing intent
        return current_intent

'''This function masks the entities in user input, and updates the attributes dictionary'''
def getattributes(uinput, context, attributes):

     # Can use context to context specific attribute fetching
    # print("getattributes context ", context)
    if context.name.startswith('IntentComplete'):
        return attributes, uinput
    else:
        # Code can be optimised here, loading the same files each time suboptimal
        files = os.listdir('./entities/')
        # Filtering dat files and extracting entity values inside the entities folder
        entities = {}
        for fil in files:
            if fil == ".ipynb_checkpoints":
                continue
            lines = open('./entities/'+fil).readlines()
            
            for i, line in enumerate(lines):
                lines[i] = line[:-1] 
            entities[fil[:-4]] = '|'.join(lines)

       
        # Extract entity and update it in attributes dict
        for entity in entities:
            for i in entities[entity].split('|'):
                if i.lower() in uinput.lower():
                    attributes[entity] = i
    #    print(attributes)
    #   print(entities)
        # Masking the entity values $ sign
        for entity in entities:
            uinput = re.sub(entities[entity], r'$'+entity, uinput, flags=re.IGNORECASE)
            
        
      #  print(uinput)

        return attributes, uinput


def model_predict(new_review):
    
    import pandas as pd
    import numpy as np
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
  #  from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
   # print(X.shape)
    new_review = new_review.lower() 
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
   # print(new_corpus)
    new_X_test = cv.transform(new_corpus).toarray()
  #  print(new_X_test)
    new_y_pred = pickle_model.predict(new_X_test)
  #S  print(new_y_pred)
   # print(corpus[-2])   
    return new_y_pred[0]
    
   
