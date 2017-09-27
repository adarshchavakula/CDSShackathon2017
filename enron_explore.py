#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:35:18 2017

@author: Vinay
"""

import os
os.chdir('/Users/Vinay/Projects/CDSS hackathon/enron/enron/CDSS')
import pandas as pd
import numpy as np
import json


import os, sys, email
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set_style('whitegrid')
import wordcloud


import networkx as nx
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer
from subprocess import check_output

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
import re

                                # Reading the data and pre-processing data

emails_df = pd.read_csv('enron_emails.csv')
personal_data = pd.read_csv('enron_employee_list_partial.csv')
sents = json.loads(open('enron-sentences.json').read())

with open("enron-sentences.json") as json_data:
    sents = json.loads(json_data)



## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs



# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails_df['user'] = emails_df['file'].map(lambda x:x.split('/')[0])
del messages

emails_df = emails_df.set_index('Message-ID')\
    .drop(['file', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding'], axis=1)
# Parse datetime
emails_df['Date'] = pd.to_datetime(emails_df['Date'], infer_datetime_format=True)
emails_df.dtypes

emails_df.to_csv('processed_enron.csv')
emails_df = pd.read_csv('processed_enron.csv')

def clean(text):
    stop = set(stopwords.words('english'))
    stop.update(("to","cc","subject","http","from","sent","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    porter= PorterStemmer()

    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    #stem = " ".join(porter.stem(token) for token in normalized.split())

    return normalized


analysis_df=emails_df[['From', 'To', 'Date','content']].dropna().copy()
analysis_df = analysis_df.loc[analysis_df['To'].map(len) == 1]
sub_df=analysis_df.copy()


                            #Get all pre-processed emails in a list of list

text_clean=[]
for text in sub_df['content']:
    text_clean.append(clean(text).split())

                        #Build Word2Vec mdel incrementally trained on the google one

modelwv = gensim.models.Word2Vec(size=300,window=8,min_count = 5)
modelwv.build_vocab(text_clean)
modelwv.intersect_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
modelwv.train(text_clean,total_examples=80012327,epochs = 1)

vocab_words=[]
for v in modelwv.wv.vocab:
    vocab_words.append(v)
word_vectors = []
for w in vocab_words:
    word_vectors.append(modelwv[w])

                                # Save clean_text and word2vec model

import pickle

file_Name = "cleantext"
fileObject = open(file_Name,'wb')
pickle.dump(text_clean,fileObject)
fileObject.close()

modelwv.save('word2vec_model')


                                            # Topic Modelling
                # Refer https://www.kaggle.com/jaykrishna/topic-modeling-enron-email-dataset


dictionary = corpora.Dictionary(text_clean)
text_term_matrix = [dictionary.doc2bow(text) for text in text_clean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(text_term_matrix, num_topics=4, id2word = dictionary, passes=1)

---------------------------------------------------------------

text_clean2 = text_clean[1:1000]

dictionary = corpora.Dictionary(text_clean2)
text_term_matrix = [dictionary.doc2bow(text) for text in text_clean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(text_term_matrix, num_topics=4, id2word = dictionary, passes=1)


analysis_df.to_csv('analysis_df.csv')

model = gensim.models.Word2Vec.load('word2vec_model')




                                            # Experiments to plot TSNE

import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

vocab_words=[]
for v in model.wv.vocab:
    vocab_words.append(v)
word_vectors = []
for w in vocab_words:
    word_vectors.append(model[w])

import pandas as pd

dfnew = pd.DataFrame({"words":vocab_words , "vectors":word_vectors })
dfnew.to_csv('w2v.csv')

X = model[model.wv.vocab]



tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)


plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()

 import numpy as np
 from sklearn.manifold import TSNE


 X = np.array(word_vectors[1000:10000])
 X_embedded = TSNE(n_components=2).fit_transform(X)


plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.show()



y = X_embedded[:, 0]
z = X_embedded[:, 1]
n =  [x for x in map(str,np.arange(9000))]
n = vocab_words[1:250]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i],y[i]))
