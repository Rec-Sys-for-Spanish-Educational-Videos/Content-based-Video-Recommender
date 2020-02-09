# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:32:52 2019
@author: Mickey
"""

# python -m nltk.downloader -> download all
# $ pip install langdetect
# $ pip install tensorflow_hub
# $ pip install tensorflow


import json
from sklearn.cluster import KMeans
import gensim
import pandas as pd
import re
import tensorflow_hub as hub
import tensorflow as tf

#Function that returns the Topics of an LDA model and the words with the scores that form the topic.

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

#Function for tokenization
def preprocess(text):
    result = ""
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result=result+' '+token
    return result

#Function that returns the result for a query
def resultForQuery(query):
    embeddedQuery=[]
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddedQuery = session.run(embed([query]))
    querryCluster = model.predict(embeddedQuery)[0]
    wordList = re.sub("[^\w]", " ", query).split()
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
    
    result = ''
    # Creating a string with the top 5 transcripts with their scores.
    for score in sortedScores[:5]:
        result = result + str(score) +'\n'   
    return result

#Opening the file with all the data about the videos.
with open('videos_upv_cleaned.json') as f:
	data = json.load(f)

documents = []
preprocessedDocumentsList = []
videoIdDictionary = {}
nrOfTranscriptsToProcess = 45000

#Creating the bag of words and the input for NNLM
nr=0
for i in range(1,nrOfTranscriptsToProcess):
    if data[i]["transcription"] is not "" and len(data[i]["transcription"])>6000:
        preprocessedTranscript = preprocess(data[i]["transcription"])
        videoIdDictionary[nr]=i
        nr+=1
        documents.append(preprocessedTranscript)
        wordList = re.sub("[^\w]", " ",  preprocessedTranscript).split()
        words = []
        for word in wordList:
            words.append(word)
        preprocessedDocumentsList.append(words)

#Using word embedding algorithm on all the transcripts and creating the matrix
embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")
  
embeddedDocuments = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddedDocuments = session.run(embed(documents))        

#Creating the cluster model to cluster the transcripts into 3 clusters(the domains)
numberOfClusters = 3
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

