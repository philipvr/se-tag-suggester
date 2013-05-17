#!/usr/bin/env python
import sys
import re
import string
import numpy
from scipy import sparse
from bs4 import BeautifulSoup
from lxml import etree
from stemming import porter2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def tokenize(text):
    """
    Take a string and split it into tokens on word boundaries.

    A token is defined to be one or more alphanumeric characters,
    underscores, or apostrophes.  Remove all other punctuation, whitespace, and
    empty tokens.  Do case-folding to make everything lowercase. This function
    returns a list of the tokens in the input string.
    """
    tokens = re.findall("[\w']+", text.lower())
    return [porter2.stem(token) for token in tokens]

class IDMap(object):
    def __init__(self):
        self.mapIdsToText = {}
        self.mapTextToIds = {}
        self.nextId = 0
        
    def add_tokens(self, text):
        for tag in text:
            if tag not in self.mapTextToIds:
                self.mapTextToIds[tag] = self.nextId
                self.mapIdsToText[self.nextId] = tag
                self.nextId = self.nextId + 1
        return [self.mapTextToIds[tag] for tag in text]
    
    def get_id(self, text):
        if text in self.mapTextToIds:
            return self.mapTextToIds[text]
        else:
            return None
    
    def get_token(self, id):
        if id in self.mapIdsToText:
            return self.mapIdsToText[id]
        else:
            return None
    
    def get_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.mapTextToIds:
                ids.append(self.mapTextToIds[tok])
        return ids
            
    def size(self):
        return len(self.mapIdsToText)

class Labeler(object):
    def __init__(self, path):
        self.xml_path = path
        self.textIdsMap = IDMap()
        self.tagIdsMap = IDMap()
        self.classif = OneVsRestClassifier(LinearSVC())
        
    def posts(self, path):
        for event, element in etree.iterparse(path, tag="row"):
            post = {}
            post["Id"] = int(element.get("Id"))
            post["Body"] = element.get("Body")
            post["Tags"] = element.get("Tags")
            post["Title"] = element.get("Title")
            post["PostTypeId"] = element.get("PostTypeId")
            yield post

            
    def get_questions(self):
        questions = []
        for post in self.posts(self.xml_path):
            if post['PostTypeId'] == '1':
                id = int(post['Id'])
                title = post['Title']
                
                body = BeautifulSoup(post['Body']).get_text()
                
                textTokens = tokenize(title)
                textTokens.extend(tokenize(body))
                textTokensIds = self.textIdsMap.add_tokens(textTokens)
                
                tags = post['Tags']
                tagTokens = string.split(tags[1:len(tags)-1],"><")
                tagIds = self.tagIdsMap.add_tokens(tagTokens)
                
                q = { 'Id': id,
                    'Title': title,
                    'Body': body,
                    'TextTokenIds': textTokensIds,
                    'TagIds': tagIds
                }
                questions.append(q)
        print "number of tags: \t", self.tagIdsMap.size()
        print "number of words: \t", self.textIdsMap.size()
        print "number of questions: \t", len(questions), "\n"
        return questions
        
    def create_classifier(self):
        questions = self.get_questions()
        X = sparse.lil_matrix((len(questions),self.textIdsMap.size()))
        y = []
        
        for i,q in enumerate(questions):
            textTokIds = list(set(q['TextTokenIds']))
            for k in textTokIds:
                X[i,k] = q['TextTokenIds'].count(k)
            y.append(tuple(q['TagIds']))

        self.classif.fit(X, y)
            
    def label_question(self, new_question):
        title = new_question['Title']
        body = BeautifulSoup(new_question['Body']).get_text()
        
        textTokens = tokenize(title)
        textTokens.extend(tokenize(body))
        textTokensIds = self.textIdsMap.get_ids(textTokens)
        
        X = sparse.lil_matrix((1,self.textIdsMap.size()))
        textTokIds = list(set(textTokensIds))
        for k in textTokIds:
            X[0,k] = textTokensIds.count(k)
            
        labels = self.classif.predict(X)
        return [self.tagIdsMap.get_token(label) for label in labels[0]]
    
def main():
    if len(sys.argv) == 1:
        labeler = Labeler('photo.stackexchange.com\\posts.xml')
        labeler.create_classifier()
        TEST_POST1 = {"Body": "<p>What is focal-length?  Are focal-length and zoom synonymous?  How does the focal length of a photo affect it?</p>\n", "ViewCount": "287", "LastEditorDisplayName": "", "Title": "What is focal length and how does it affect my photos?", "LastEditorUserId": "1943", "LastActivityDate": "2011-08-16T11:52:11.027", "LastEditDate": "2011-08-16T11:52:11.027", "AnswerCount": "4", "CommentCount": "1", "AcceptedAnswerId": "113", "Score": "11", "PostTypeId": "1", "OwnerUserId": "54", "Tags": "<terminology><focal-length>", "CreationDate": "2010-07-15T20:03:35.343", "FavoriteCount": "2", "Id": "103"}
        TEST_POST2 = {"Body": "<p>How does one cancel out extra light while clicking a photo? Is decreasing the aperture a better option, or using a faster shutter speed? What are other options we could try?</p>\n", "ViewCount": "150", "LastEditorDisplayName": "", "Title": "How does one deal with too much light when taking a photo?", "LastEditorUserId": "1943", "LastActivityDate": "2011-09-01T20:52:02.160", "LastEditDate": "2011-09-01T02:13:05.537", "AnswerCount": "4", "CommentCount": "2", "Score": "4", "PostTypeId": "1", "OwnerUserId": "6476", "Tags": "<exposure><aperture><shutter-speed><light>", "CreationDate": "2011-08-31T22:33:41.450", "Id": "15341"}
        TEST_POST3 = {"Body": "I just purchased a brand new Canon Rebel T3i kit from Costco.. I used it for a couple of days, but now when I have the 18-55mm lens attached and the camera on auto-focus, it will not take a picture. Still will take a picture in manual focus. And will take a picture in either focus with the other lens on (55-250mm).I've been told by a friend that the lens may not be seating right. Are there any other settings I can check? I am very new to the camera and I probably messed with some settings or accidentally bumped something I shouldn't have. Or I can take it back to Coscto and exchange it.\n", "Title": "Why won't my new Canon Rebel T3i take a picture with the 18-55mm lens in autofocus mode?", "Id": "24910"}
        print 'Title: ', TEST_POST1['Title'], '\nBody: ', BeautifulSoup(TEST_POST1['Body']).get_text()[:-1]
        print 'Tags: ', labeler.label_question(TEST_POST1), '\n'
        print 'Title: ', TEST_POST2['Title'], '\nBody: ', BeautifulSoup(TEST_POST2['Body']).get_text()[:-1]
        print 'Tags: ', labeler.label_question(TEST_POST2), '\n'
        print 'Title: ', TEST_POST3['Title'], '\nBody: ', BeautifulSoup(TEST_POST3['Body']).get_text()[:-1]
        print 'Tags: ', labeler.label_question(TEST_POST3), '\n'
    else:
        labeler = Labeler(sys.argv[1] + '\\posts.xml')
        labeler.create_classifier()

    print 'New questions', '\n============='
    while (True):
        title = raw_input("Title: ")
        body = raw_input("Body: ")
        print 'Tags: ', labeler.label_question({'Title': title, 'Body': body}), '\n'

if __name__ == "__main__":
    main()
