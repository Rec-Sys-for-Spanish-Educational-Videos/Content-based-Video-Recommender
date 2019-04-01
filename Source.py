# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:32:52 2019
@author: Mickey
"""

# python -m nltk.downloader -> download all

#go through the object vector instead of counting with nr

import json
from sklearn.cluster import KMeans
import gensim
import pandas as pd
import re
import tensorflow_hub as hub
import tensorflow as tf

def ldaResultsForCluster(cluster):
    for idx, topic in lda_models[cluster].print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

def showDictionaryWords(dictionaryNumber,nrOfWordsToShow):
    count = 0
    for k, v in dictionary[dictionaryNumber].iteritems():
        print(k, v)
        count += 1
        if count > nrOfWordsToShow:
            break

def preprocess(text):
    result = ""
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result=result+' '+token
    return result

#Opening the file with all the data about the videos.
with open('videos_upv_cleaned.json') as f:
	data = json.load(f)

documents = []
preprocessedDocumentsList = []
nrOfTranscriptsToProcess = 45000

for i in range(1,nrOfTranscriptsToProcess):
    if data[i]["transcription"] is not "" and len(data[i]["transcription"])>6000:
        preprocessedTranscript = preprocess(data[i]["transcription"])
        documents.append(preprocessedTranscript)
        wordList = re.sub("[^\w]", " ",  preprocessedTranscript).split()
        words = []
        for word in wordList:
            words.append(word)
        preprocessedDocumentsList.append(words)

#Using word embedding on all the transcripts
embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")
  
X = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    X = session.run(embed(documents))        

#Clustering
true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=3,verbose=True)
model.fit(X)

#Separating the transcript variables according to the cluster they belong into 

preprocessedListForDictionary = []

for i in range(true_k):
    preprocessedListForDictionary.append([])

nr=0
for i in range(1,nrOfTranscriptsToProcess):
    if data[i]["transcription"] is not "":
        prediction = model.predict(X[nr].reshape(1,-1))
        preprocessedListForDictionary[prediction[0]].append(preprocessedDocumentsList[nr])
        nr+=1


clusterWords = [None] * true_k
dictionary = []
for i in range(true_k):
    dictionary.append(gensim.corpora.Dictionary(preprocessedListForDictionary[i]))
    dictionary[i].filter_extremes(no_below=15, no_above=0.3, keep_n=100000)
    clusterWords[i]=[]
    
lda_models = [None] * true_k
bow_corpus = [None] * true_k  
corpus_tfidf = [None] * true_k
tfidf_models = [None] * true_k
  
for i in range(true_k):
    bow_corpus[i] = [dictionary[i].doc2bow(doc) for doc in preprocessedListForDictionary[i]]
    lda_models[i] = gensim.models.LdaMulticore(bow_corpus[i], num_topics=5, id2word=dictionary[i], passes=10, workers=10)
    print("Cluster ",i)

querryDataFrame = pd.DataFrame(columns=['VideoID','Assigned Cluster','LDA Scores'])

nr=0
for i in range(1,nrOfTranscriptsToProcess):
    if data[i]["transcription"] is not "":
        prediction = model.predict(X[nr].reshape(1,-1))
        nr+=1
        wordList = re.sub("[^\w]", " ",  data[i]["transcription"]).split()
        words = []
        for word in wordList:
            words.append(word)
        bow_corpus_predict = dictionary[prediction[0]].doc2bow(words)
        newLine = pd.DataFrame([[i,prediction[0],lda_models[prediction[0]][bow_corpus_predict]]],
                               columns=['VideoID','Assigned Cluster','CorrLDA Scores'])
        querryDataFrame = querryDataFrame.append(newLine,ignore_index = True)


querry = 'permanente parala  proteccion  delos  animales  encria instituido'
Y=[]
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    Y = session.run(embed([querry]))
querryCluster = model.predict(Y)[0]
wordList = re.sub("[^\w]", " ", querry).split()
words = []
for word in wordList:
    words.append(word)
bow_corpus_querry = dictionary[querryCluster].doc2bow(words)
querryScoresList = lda_models[querryCluster][bow_corpus_querry]
querryScores = {}
for score in querryScoresList:
    querryScores[score[0]]=score[1]
    
    
scoreDifferences = {}
for index,entry in querryDataFrame.iterrows():
    if entry['Assigned Cluster'] == querryCluster:
        totalScore=0
        for score in querryDataFrame.at[index,'CorrLDA Scores']:
            if score[0] in querryScores:
                totalScore += abs(score[1]-querryScores[score[0]])
            else:
                totalScore +=0.1
        scoreDifferences[entry['VideoID']]=totalScore

import operator    
sorted_x = sorted(scoreDifferences.items(), key=operator.itemgetter(1))
