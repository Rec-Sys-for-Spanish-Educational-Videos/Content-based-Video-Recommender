# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:58:22 2019

@author: Diana
"""



import gensim
import pandas as pd
import re
import pickle
import tensorflow as tf
import tensorflow_hub as hub


import wiki
import preprocessData as pData


#extract data bases of videos

dataframe = pd.read_excel("LabeledTranscripts.xlsx", usecols = [0])

videosName = dataframe.values

dataframe = pd.read_excel("LabeledTranscripts.xlsx", usecols = [1])

transcripts = dataframe.values

dataframe = pd.read_excel("LabeledTranscripts.xlsx", usecols = [2])

labels = dataframe.values


documents = []
documentsWithLabels = [[],[],[]]
preprocessedDocumentsList = []
videoIdDictionary = {}


#Process all transcripts
nrOfTranscriptsToProcess = len(videosName)



for i in range(0,nrOfTranscriptsToProcess):
    transcript = transcripts[i][0]
    videoName = videosName[i][0]
    label = labels[i][0]
    if transcript !=  "" and len(transcript)>6000:      #keep only valid transcripts for preprocessing
        preprocessedTranscript = pData.preprocess(transcript)   

        #keep all preprocessed transcripts in a container for each label
        documentsWithLabels[label].append((preprocessedTranscript,videoName))
        
        #keep all preprocessed transcripts together
        documents.append([preprocessedTranscript, videoName, label])
        
        
        wordList = re.sub("[^\w]", " ",  preprocessedTranscript).split()
        words = []
        for word in wordList:
            words.append(word)

        #keep all words from transcripts
        preprocessedDocumentsList.append(words)


noOfClusters = 3
preprocessedListForDictionary = []


for i in range(noOfClusters):
    preprocessedListForDictionary.append([])


#append transcripts words in a list
index = 0
for transcript, videoName, label in documents:
    preprocessedListForDictionary[label].append(preprocessedDocumentsList[index])
    index += 1



# Creating the dictionaries for the LSI models

dictionary = []
for i in range(noOfClusters):
    dictionary.append(gensim.corpora.Dictionary(preprocessedListForDictionary[i]))

bow_corpus = [None] * noOfClusters  
lsi = [None] * noOfClusters
indexList = [None] * noOfClusters


from gensim import models
from gensim import similarities

# Creating the LSI models

for i in range(noOfClusters):
    if( len(dictionary[i]) != 0  ):
        #we have to create a bag of words( BoW )
        bow_corpus[i] = [dictionary[i].doc2bow(doc) for doc in preprocessedListForDictionary[i]]
        
        #we will transform it in a tf-idf vector
        tfidf = models.TfidfModel(bow_corpus[i]) 
        corpus_tfidf = tfidf[bow_corpus[i]]

        """
        words = []
        for cp in corpus_tfidf[2:3]:
            for id, freq in cp:
                words.append(str(dictionary[1][id]) )
                
        print(words)
        """
        
        
        lsi[i] = models.LsiModel(corpus = corpus_tfidf, id2word=dictionary[i], num_topics=5)
        #we will compute a similarity matrix, which it will help us later, for query
        indexList[i] = similarities.MatrixSimilarity(lsi[i][corpus_tfidf])

        
        #print(indexList[1])
        print(lsi[i].print_topics(num_topics= 5 , num_words=10))
        

#load the model
embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")
clf = pickle.load(open("alex_pretrained_model.sav", 'rb'))

print("..................")
#Function that returns the result for a query

def resultForQuery(query):
    ans = wiki.wikipedia_search(query)     #search a sequence on wiki pages
    if(len(ans["itemList"]) != 0 ):             #if we have a result
            queryWiki =  (ans["itemList"][0]["description"])       #assign as query this sequence
    else:   
            queryWiki = query                  #else assign just the query
                
    with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddedQueryWiki = session.run(embed([queryWiki]))

    #make a predict for the query for its cluster
    queryClusterWiki = clf.predict(embeddedQueryWiki)[0]
        
    #search in assigned container
    cluster = queryClusterWiki
    
    print("cluster")
    print(cluster)
    
    #transform in a bow corpus the query
    vec_bow = dictionary[cluster].doc2bow(pData.singularizeQuery(query))
    # convert the query to LSI space
    vec_lsi = lsi[cluster][vec_bow]  
    
    # perform a similarity query against the corpus
    sims = indexList[cluster][vec_lsi]  
    
    
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    
    #print top 10 results
    for i, s in enumerate(sims[:10]):
        print(s, documentsWithLabels[cluster][s[0]][1])


'''
while(True):
    query = input()
    resultForQuery(query)
'''




