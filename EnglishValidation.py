# -*- coding: utf-8 -*-
"""
Created on 4th of April
@author: Mickey
"""

# python -m nltk.downloader -> download all
# $ pip install langdetect


# TODO go through the object vector instead of counting with nr

import json
from sklearn.cluster import KMeans
import gensim
import pandas as pd
import re
import tensorflow_hub as hub
import tensorflow as tf
import urllib.request  
from bs4 import BeautifulSoup
import urllib

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

def getEnglishTranscriptForId(videoId):
    transcriptLink = "https://media.upv.es/rest/plugins/admin-plugin-translectures/dfxp/"+data[videoId]["_id"]+"/en"
    try:
        uf = urllib.request.urlopen(transcriptLink)
    except:
        print("Transcript ", i," didn't work.")
    html = uf.read()
    rawText = BeautifulSoup(html).get_text()
    result = ""
    for token in gensim.utils.simple_preprocess(rawText):
        result=result+' '+token
    return result


#Opening the file with all the data about the videos.
with open('videos_upv_cleaned.json') as f:
	data = json.load(f)

documents = []
preprocessedDocumentsList = []
videoIdDictionary = {}
nrOfTranscriptsToProcess = 45000

nr=0
for i in range(1,nrOfTranscriptsToProcess):
    if data[i]["transcription"] is not "" and len(data[i]["transcription"])>6000:
        transcriptLink = "https://media.upv.es/rest/plugins/admin-plugin-translectures/dfxp/"+data[i]["_id"]+"/en"
        try:
            uf = urllib.request.urlopen(transcriptLink)
        except:
            print("Transcript ", i," didn't work.")
        html = uf.read()
        rawText = BeautifulSoup(html).get_text()
        preprocessedTranscript = preprocess(rawText)
        videoIdDictionary[nr]=i
        nr+=1
        documents.append(preprocessedTranscript)
        wordList = re.sub("[^\w]", " ",  preprocessedTranscript).split()
        words = []
        for word in wordList:
            words.append(word)
        preprocessedDocumentsList.append(words)

#Using word embedding on all the transcripts
embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
  
embeddedDocuments = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddedDocuments = session.run(embed(documents))        

#Clustering
numberOfClusters = 4
model = KMeans(n_clusters=numberOfClusters, init='k-means++', max_iter=300, n_init=3,verbose=True)
model.fit(embeddedDocuments)

#Separating the transcript variables according to the cluster they belong into 

preprocessedListForDictionary = []

for i in range(numberOfClusters):
    preprocessedListForDictionary.append([])

for index, document in enumerate(embeddedDocuments):
    prediction = model.predict(document.reshape(1,-1))
    preprocessedListForDictionary[prediction[0]].append(preprocessedDocumentsList[index])

# Creating the dictionaries for the LDA models
        
dictionary = []
for i in range(numberOfClusters):
    dictionary.append(gensim.corpora.Dictionary(preprocessedListForDictionary[i]))
    dictionary[i].filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    
lda_models = [None] * numberOfClusters
bow_corpus = [None] * numberOfClusters  
corpus_tfidf = [None] * numberOfClusters
tfidf_models = [None] * numberOfClusters

# Creating the LDA models
  
for i in range(numberOfClusters):
    bow_corpus[i] = [dictionary[i].doc2bow(doc) for doc in preprocessedListForDictionary[i]]
    lda_models[i] = gensim.models.LdaMulticore(bow_corpus[i], num_topics=5, id2word=dictionary[i], passes=10, workers=10)

querryDataFrame = pd.DataFrame(columns=['VideoID','Assigned Cluster','LDA Scores'])

for index, document in enumerate(embeddedDocuments):
        prediction = model.predict(document.reshape(1,-1))
        wordList = re.sub("[^\w]", " ",  documents[index]).split()
        words = []
        for word in wordList:
            words.append(word)
        bow_corpus_predict = dictionary[prediction[0]].doc2bow(words)
        newLine = pd.DataFrame([[videoIdDictionary[index],prediction[0],lda_models[prediction[0]][bow_corpus_predict]]],
                               columns=['VideoID','Assigned Cluster','LDA Scores'])
        querryDataFrame = querryDataFrame.append(newLine,ignore_index = True)


querry = 'computer science and algorithms, data structures'
embeddedQuery=[]
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddedQuery = session.run(embed([querry]))
querryCluster = model.predict(embeddedQuery)[0]
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
        for score in querryDataFrame.at[index,'LDA Scores']:
            if score[0] in querryScores:
                totalScore += abs(score[1]-querryScores[score[0]])
            else:
                totalScore +=0.1
        scoreDifferences[entry['VideoID']]=totalScore

import operator    
sortedScores = sorted(scoreDifferences.items(), key=operator.itemgetter(1))

# Priting the top 5 transcripts with their scores.
for score in sortedScores[:3]:
    print(score)
    print(getEnglishTranscriptForId(score[0]))
