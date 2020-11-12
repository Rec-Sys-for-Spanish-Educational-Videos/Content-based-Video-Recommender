# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:22:41 2019

@author: Diana
"""
import inflector

#preprocess the query
def singularizeQuery(query):
    listOfWords = []
    for word in query.lower().split():
        word = singularizator.singularize(word)
        listOfWords.append(word)
        
    return listOfWords

#get all stopwords from an input file and put their in a list
def buildStopWords():
    fileInput = open('stopwords.txt', 'r', encoding='latin-1')
    for word in fileInput:
        word = word.replace("\n", "")
        spanish_stopwords.append(word)


def ifIsFromStopWords(word):
    if word in spanish_stopwords:
        return True
    
    return False

#need to run a java server
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost', port=9000)


#keep from transcripts just the nouns in singular form
def preprocess(text):
    result = " "
    
    for word, pos in nlp.pos_tag(text):
        if pos == 'NOUN' :
            if( ifIsFromStopWords(word) == False ):
                result += singularizator.singularize(word) + " "
                
    
    return result

spanish_stopwords = []
buildStopWords()
singularizator = inflector.Spanish()