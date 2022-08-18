import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import os
from nltk.corpus import wordnet
os.chdir('G:\MetaVerseCompany')
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import  hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Downloading the required sets such as stopwords or wordnet 
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

train=pd.read_csv('train.csv')
train.head()



print(train.isna().sum())
# Handling missing values (I defined it in a class form so that can be used later on in the pipeline)
class Missing_Value(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Instance is Created')                
    def fit(self,Data):
        self.Data=Data
        self.Data['title'].fillna(value='Not Speciefied',inplace=True)
        self.Data['text'].fillna(value='Empty',inplace=True)
        self.Data['author'].fillna(value='unknown',inplace=True)
        return self
    def transform(self,Data,y=None):
        
        Data['title'].fillna(value='Not Speciefied',inplace=True)
        Data['text'].fillna(value='Empty',inplace=True)
        Data['author'].fillna(value='unknown',inplace=True)
        return Data
        
    def fit_transform(self,Data,y=None):
        self.Data=Data
        self.Data['title'].fillna(value='Not Speciefied',inplace=True)
        self.Data['text'].fillna(value='Empty',inplace=True)
        self.Data['author'].fillna(value='unknown',inplace=True)
        return self.Data


class preprocessing(BaseEstimator,TransformerMixin):
    def __init__(self):
        print('Instance is Created')                
    def fit(self,Data):
        self.length=Data.shape[0]
        self.Data=Data
        for i in range(self.length):
            cleaned_txt=self.clean_text(self.Data['text'].iloc[i])
            cleaned_title=self.clean_text(self.Data['title'].iloc[i])
            cleaned_text_splitted=cleaned_txt.split(' ')
            cleaned_title_splitted=cleaned_title.split(' ')

            words_and_tags_text=nltk.pos_tag(cleaned_text_splitted)
            words_and_tags_title=nltk.pos_tag(cleaned_title_splitted)
            text_parts=[]
            for word,tag in words_and_tags_text:
                lemmatizer = WordNetLemmatizer()
                word_=lemmatizer.lemmatize(word,pos=self.get_wordnet_part_of_speech(tag))
                text_parts.append(word_)

            title_parts=[]
            for word,tag in words_and_tags_title:
                lemmatizer = WordNetLemmatizer()
                word_=lemmatizer.lemmatize(word,pos=self.get_wordnet_part_of_speech(tag))
                title_parts.append(word_)


            self.Data['text'].iloc[i]=' '.join(text_parts)
            self.Data['title'].iloc[i]=' '.join(title_parts)

        return self
    def transform(self,Data,y=None):
        length=Data.shape[0]
       
        for i in range(length):
            cleaned_txt=self.clean_text(Data['text'].iloc[i])
            cleaned_title=self.clean_text(Data['title'].iloc[i])
            cleaned_text_splitted=cleaned_txt.split(' ')
            cleaned_title_splitted=cleaned_title.split(' ')

            words_and_tags_text=nltk.pos_tag(cleaned_text_splitted)
            words_and_tags_title=nltk.pos_tag(cleaned_title_splitted)
            text_parts=[]
            for word,tag in words_and_tags_text:
                lemmatizer = WordNetLemmatizer()
                word_=lemmatizer.lemmatize(word,pos=self.get_wordnet_part_of_speech(tag))
                text_parts.append(word_)

            title_parts=[]
            for word,tag in words_and_tags_title:
                lemmatizer = WordNetLemmatizer()
                word_=lemmatizer.lemmatize(word,pos=self.get_wordnet_part_of_speech(tag))
                title_parts.append(word_)


            Data['text'].iloc[i]=' '.join(text_parts)
            Data['title'].iloc[i]=' '.join(title_parts)
        return Data
    def fit_transform(self,Data,y=None):
        self.fit(Data)
        return self.Data
    def clean_text(self,text):
        text=text.lower()
        text=re.sub(r"i\'m",'i am',text)
        text=re.sub(r"i’m",'i am',text)

        text=re.sub(r"she\'s",'she is',text)
        text=re.sub(r"she’s",'she is',text)

        text=re.sub(r"that\'s",'that is',text)
        text=re.sub(r"that’s",'that is',text)

        text=re.sub(r"they\'re",'they are',text)
        text=re.sub(r"they’re",'they are',text)

        text=re.sub(r"don\'t",'do not',text)
        text=re.sub(r"don’t",'do not',text)


        text=re.sub(r"doesn\'t",'does not',text)
        text=re.sub(r"doesn’t",'does not',text)

        text=re.sub(r"hasn\'t",'has not',text)
        text=re.sub(r"hasn’t",'has not',text)

        text=re.sub(r"hasn\''",'has not',text)

        text=re.sub(r"hadn\'t",'had not',text)
        text=re.sub(r"hadn’t",'had not',text)


        text=re.sub(r"haven\'t",'have not',text)
        text=re.sub(r"haven’t",'have not',text)

        text=re.sub(r"haven\'",'have not',text)
        text=re.sub(r"haven’",'have not',text)

        text=re.sub(r"isn\'t",'is not',text)
        text=re.sub(r"isn’t",'is not',text)

        text=re.sub(r"isn\'",'is not',text)
        text=re.sub(r"isn’",'is not',text)


        text=re.sub(r"didn\'t",'did not',text)
        text=re.sub(r"didn’t",'did not',text)
        text=re.sub(r"didn’",'did not',text)

        text=re.sub(r"wasn\'t",'was not',text)
        text=re.sub(r"wasn’t",'was not',text)

        text=re.sub(r"wasn\'",'was not',text)
        text=re.sub(r"wasn’",'was not',text)

        text=re.sub(r"weren\'t",'were not',text)
        text=re.sub(r"weren’t",'were not',text)

        text=re.sub(r"weren\'",'were not',text)
        text=re.sub(r"weren’",'were not',text)

        text=re.sub(r"he\'s",'he is',text)
        text=re.sub(r"he’s",'he is',text)


        text=re.sub(r"\'ll",'will',text)
        text=re.sub(r"’ll",'will',text)

        text=re.sub(r"[-()\"/@/#;:<>{}+=~|?, ]"," ",text)
        text=re.sub(r'[^a-zA_Z]',' ',text)

        return text
    def get_wordnet_part_of_speech(self,word_tag):
        if word_tag.startswith('J'):
            return wordnet.ADJ
        elif word_tag.startswith('V'):
            return wordnet.VERB
        elif word_tag.startswith('N'):
            return wordnet.NOUN
        elif word_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN




# Vectorizing the text and stacking them (because I chose both of title and text columns for inputs of the model)

class TfidfVectorizer_two_columns(BaseEstimator,TransformerMixin):
    def __init__(self,max_features=300):
        self.max_features=max_features
    def fit(self,Data):
        self.Data=Data
        self.vectorizer_title=TfidfVectorizer(max_features=self.max_features,analyzer='word')
        Xtrain_title_v=self.vectorizer_title.fit_transform(self.Data.loc[:,'title'])
        self.vectorizer_text=TfidfVectorizer(max_features=self.max_features,analyzer='word')
        Xtrain_text_v=self.vectorizer_text.fit_transform(self.Data.loc[:,'text'])
        print(Xtrain_title_v)
        print(Xtrain_text_v)
        self.X_train_all=hstack([Xtrain_text_v,Xtrain_title_v])
        return self
    def fit_transform(self,Data,y=None):
        self.fit(Data)
        return self.X_train_all
    def transform(self,Data,y=None):
        Xtrain_title_v=self.vectorizer_title.transform(Data.loc[:,'title'])
        Xtrain_text_v=self.vectorizer_text.transform(Data.loc[:,'text'])
        X_train_all=hstack([Xtrain_text_v,Xtrain_title_v])        
        return X_train_all




#{'classifier':[xgboost.XGBClassifier()],
#         "classifier__learning_rate"    : [0.01, 0.10, 0.20, 0.30 ],
#         'classifier__min_child_weight': [1, 5, 10],
#         'classifier__gamma': [0.5, 1, 2, 5],
#         'classifier__subsample': [0.6, 0.8, 1.0],
#         'classifier__colsample_bytree': [0.6, 0.8, 1.0],
#         'classifier__max_depth': [3, 4, 5]
#         },
# fitted_model.best_score_
# fitted_model.cv_results_

# I did make another pipeline because I know at this step the appropriate hyperparames based on the hyperparameter tuning results

pipeline = [('missing_value', Missing_Value()), ('preprocess', preprocessing()),('tfidf_two_vectors',TfidfVectorizer_two_columns()),('classifier',SVC(C=100, gamma=1))] 

pipe = Pipeline(pipeline) 
pipe.fit(train.iloc[:,:-1],train.iloc[:,-1])

# The inputs for pipe should be a dataframe, because we used a pipeline so that the transformations can be done automatically

test_set=pd.read_csv('test.csv')
y_predict_test=pipe.predict(test_set)



from sklearn.metrics import accuracy_score
test_label=pd.read_csv('labels.csv')
y_test=test_label['label'].values
f1_score(y_test,y_predict_test)


