# Text-Document-Similarity-Detection-Without-Using-Python-ML-Libraries
Building similarity detection ML codes from scratch using linear algebra.


Creating a python program that will compute the text document similarity between different documents. The implementation will take a list of documents as an input text corpus, and it will compute a dictionary of words for the given corpus. Later, when a new document (i.e, search document) is provided, the implementation should provide a list of documents that are similar to the given search document, in descending order of their similarity with the search document. For computing similarity between any two documents in our question, we are using the following distance measures (optionally, you are using  other set measures as well).

1. dot product between the two vectors

2. distance norm (or Euclidean distance) between two vectors .e.g. || u − v ||

As part of answering the question, we can also compare and comment on which of the two methods (or any other measure if you have used some other measure) will perform better and what are the reasons for it.





#Answer:

The code is devided in two segments, one for finding matching words list, the other for finding similarity percentage
Part 1: matching words list
###### First We Define a corpus which is a list of documents
corpus=['When you are old and grey and full of sleep,',
'And nodding by the fire, take down this book,',
'And slowly read, and dream of the soft look',
'Your eyes had once, and of their shadows deep;',
'',
'How many loved your moments of glad grace,',
'And loved your beauty with love false or true,',
'But one man loved the pilgrim soul in you,',
'And loved the sorrows of your changing face;',
'',
'And bending down beside the glowing bars,',
'Murmur, a little sadly, how Love fled',
'And paced upon the mountains overhead',
'And hid his face amid a crowd of stars.',
'-When You Are Old ',
'-Yeats']
###### Data Cleaning function (removing all punctuations, whitespaces and transforming everything into lowercase)
import re
    
def cleaner(x):   
    x=" ".join(map(str,x))
    x=re.sub(r"\W"," ",x) 
    x=re.sub(' +', ' ',x) 
    x=x.lower()
    return x.split()
###### Then we make a list of elements with all the words to build our dictionary
final_list=cleaner(corpus)
​
###### We make a dictionaty with values as number of word occurences in the corpus  
corp_dict = {}
for m in final_list:
    if( m in corp_dict.keys()):
        corp_dict[m] += 1
    else:
        corp_dict[m]=1
        
print(corp_dict)
{'when': 2, 'you': 3, 'are': 2, 'old': 2, 'and': 11, 'grey': 1, 'full': 1, 'of': 6, 'sleep': 1, 'nodding': 1, 'by': 1, 'the': 6, 'fire': 1, 'take': 1, 'down': 2, 'this': 1, 'book': 1, 'slowly': 1, 'read': 1, 'dream': 1, 'soft': 1, 'look': 1, 'your': 4, 'eyes': 1, 'had': 1, 'once': 1, 'their': 1, 'shadows': 1, 'deep': 1, 'how': 2, 'many': 1, 'loved': 4, 'moments': 1, 'glad': 1, 'grace': 1, 'beauty': 1, 'with': 1, 'love': 2, 'false': 1, 'or': 1, 'true': 1, 'but': 1, 'one': 1, 'man': 1, 'pilgrim': 1, 'soul': 1, 'in': 1, 'sorrows': 1, 'changing': 1, 'face': 2, 'bending': 1, 'beside': 1, 'glowing': 1, 'bars': 1, 'murmur': 1, 'a': 2, 'little': 1, 'sadly': 1, 'fled': 1, 'paced': 1, 'upon': 1, 'mountains': 1, 'overhead': 1, 'hid': 1, 'his': 1, 'amid': 1, 'crowd': 1, 'stars': 1, 'yeats': 1}
###### We transform the dictionaty with decending order of values so that we can find the matches in decending order 
final_dict=dict(sorted(corp_dict.items(),reverse=True, key=lambda item: item[1]))
print(final_dict)
{'and': 11, 'of': 6, 'the': 6, 'your': 4, 'loved': 4, 'you': 3, 'when': 2, 'are': 2, 'old': 2, 'down': 2, 'how': 2, 'love': 2, 'face': 2, 'a': 2, 'grey': 1, 'full': 1, 'sleep': 1, 'nodding': 1, 'by': 1, 'fire': 1, 'take': 1, 'this': 1, 'book': 1, 'slowly': 1, 'read': 1, 'dream': 1, 'soft': 1, 'look': 1, 'eyes': 1, 'had': 1, 'once': 1, 'their': 1, 'shadows': 1, 'deep': 1, 'many': 1, 'moments': 1, 'glad': 1, 'grace': 1, 'beauty': 1, 'with': 1, 'false': 1, 'or': 1, 'true': 1, 'but': 1, 'one': 1, 'man': 1, 'pilgrim': 1, 'soul': 1, 'in': 1, 'sorrows': 1, 'changing': 1, 'bending': 1, 'beside': 1, 'glowing': 1, 'bars': 1, 'murmur': 1, 'little': 1, 'sadly': 1, 'fled': 1, 'paced': 1, 'upon': 1, 'mountains': 1, 'overhead': 1, 'hid': 1, 'his': 1, 'amid': 1, 'crowd': 1, 'stars': 1, 'yeats': 1}
#Finally we define the function that returns the list of matches in a given search document
​
def match_list(input_dict,input_doc):
    l2=input_doc
    l1=list(input_dict.keys())
    mat1=[]
    for j in l1:
        for i in range (len(l2)):
            if l2[i]==j:
                mat1.append(j)
                break
​
    return mat1
#We provide our search doc/docs in a list 
​
search_doc=["This piece is usually considered to be about Yeats’ personal life.\
It discusses the unrequited love that existed between Yeats and someone he used to be involved with.\
The poem is structured as a dramatic monologue in which the speaker is addressing his once lover. \
Through the image of a book, Yeats is able to remind the listener that she has been loved by many, \
but by none like she is by one man in particular. This is a reference to the speaker, of course.\
He loved her completely, and not just for her beauty as others have. \
It is the speaker’s hope that after being reminded of these facts that she feels regret for leaving him."]
​
#We clean our search doc as well with the predefined function.
​
final_SD=cleaner(search_doc)
#We use our function to find the matching keywords in the search doc and dictionary in decending order
match_list(final_dict,final_SD)
['and',
 'of',
 'the',
 'loved',
 'love',
 'a',
 'by',
 'this',
 'book',
 'once',
 'many',
 'beauty',
 'with',
 'but',
 'one',
 'man',
 'in',
 'his',
 'yeats']
Part 2: Here we use vectorization to make similarity estimates.
Mathematical explanation of 3 ways of doing this can be found in here :
https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python

#we define a function to vectorize documents based on our corpus
def vectorizer(input_dict,input_doc):
    l2=input_doc
    l1=list(input_dict.keys())
    mat1=[]
    for j in l1:
        for i in range (len(l2)):
            if l2[i]==j:
                mat1.append(1)
                break
        else:
            mat1.append(0)
    return mat1
doc1= ["When you are old and grey and full of sleep"]
​
doc2=["But one man loved the pilgrim soul in you"]
 
cln_doc1=cleaner(doc1)
​
cln_doc2=cleaner(doc2)
    
v1= vectorizer(final_dict,cln_doc1)
​
v2=vectorizer(final_dict,cln_doc2)
 #we have successfully changed these words into vectors :) where 1 means the dictionary word is available, 0 means not available
print(v1, v2)
[1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1. Cosine similarity using dot products
from math import sqrt, pow, exp
 
def squared_sum(x):
    """ return 3 rounded square rooted value """
    return round(sqrt(sum([a*a for a in x])),3)
#now for calculation of cosine similarity we have the funcion:
​
​
def cos_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = squared_sum(x)*squared_sum(y)
    return round(numerator/float(denominator),3)
​
cos_similarity(v1, v2)
0.111
2. Euclidian Distance using norm
def euclidean_distance(x,y):
    """ return euclidean distance between two lists """
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
​
def distance_to_similarity(distance):
    return (1/exp(distance))
#now for calculation of euclidean_distance we have the funcion:
dist=euclidean_distance(v1,v2)
​
distance_to_similarity(dist)
0.018315638888734182
Jaccard Similarity
def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
​
jaccard_similarity(doc1[0], doc2[0])
​
0.6666666666666666
