# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:51:33 2019

@author: Diana
"""
import wikipedia

#return a sentence from a wiki page
def wikipedia_search(query, depth = 0, lang="es", max_result=1):
   
    items = []
    ret = { }
    ret["query"] = query
    ret["itemList"] = items
    wikipedia.set_lang(lang)
    wikiterm = wikipedia.search(query)

    #try to get a summary
    #in same cases, after a search, this algorithm returns a list of wiki pages. We have to keep informations from first accesible page
    for idx, term in enumerate(wikiterm[0:max_result]):
        try:
            item = {
                "description": wikipedia.summary(term, sentences=1),
            }
            items.append(item)
        except wikipedia.exceptions.DisambiguationError as e:
            if( depth == 1 ):
                return False
        
            for page_name in e.options:
                trySearch = wikipedia_search(page_name, 1)
                if( trySearch != False ):
                    ret = trySearch
                    break
                
    return ret


print(wikipedia_search("leyes del estado republicano"))
