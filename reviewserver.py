# Arthur: Andrew Lewis
# NLP server for Plectr

import spacy
from enum import Enum
import pyrebase
import re
from spacy.tokens import Token
from math import isclose
from monkeylearn import MonkeyLearn
# Imports the Google Cloud client library

from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request
nlp_en = spacy.load('en')
ml = MonkeyLearn('3b2ae96a397cf0e07a62561b935d2b3baee6f4ba')
model_id = 'cl_pi3C7JiL'
app = Flask('reviewserver')

config = {
  "apiKey": "AIzaSyBj1p5goV6yxL73du1Bl0lI-ihdstZLFHM",
  "authDomain": "plectr-8a45b.firebaseapp.com",
  "databaseURL": "https://plectr-8a45b.firebaseio.com",
  "storageBucket": "plectr-8a45b.appspot.com"
}

firebase = pyrebase.initialize_app(config)

auth = firebase.auth()

user = auth.sign_in_with_email_and_password('admin@admin.com', 'testserver')


class Topic(Enum):
    KNOWLEDGE = 1
    SERVICE = 2
    EXPLANATION = 3
    # TIME = 4

class Rating(Enum):
    VERY_BAD = -3
    BAD = -2
    SOMEWHAT_BAD = -1
    SOMEWHAT_GOOD = 1
    GOOD = 2
    VERY_GOOD = 3

# Class holding data for token
class LexiconEntry:
    _IS_REGEX_REGEX = re.compile(r'.*[.+*\[$^\\]')

    def __init__(self, lemma: str, topic: Topic, rating: Rating):
        assert lemma is not None
        self.lemma = lemma
        self._lower_lemma = lemma.lower()
        self.topic = topic
        self.rating = rating
        self.is_regex = bool(LexiconEntry._IS_REGEX_REGEX.match(self.lemma))
        self._regex = re.compile(lemma, re.IGNORECASE) if self.is_regex else None
    # Returns the score of the token being processed for a match
    # Cutoof for lowest vakue is 0.65. (Best performing over trails)
    def matching(self, token: Token) -> float:
        """
        A weight between 0.0 and 1.0 on how much ``token`` matches this entry.
        """
        assert token is not None
        result = 0.0
        if self.is_regex:
            if self._regex.match(token.text):
                result = 0.65
                return result
            elif self._regex.match(token.lemma_):
                result = 0.65
                return result
        else:
            if token.text == self.lemma:
                result = 1.0
                return result
            elif token.text.lower() == self.lemma:
                result = 0.9
                return result
            elif token.lemma_ == self.lemma:
                result = 0.85
                return result
            elif token.lemma_.lower() == self.lemma:
                result = 0.75
                return result
            else:
                testToken = nlp_en(self.lemma)
                result = token.similarity(testToken)    
        return result

    def __str__(self) -> str:
        result = 'LexiconEntry(%s' % self.lemma
        if self.topic is not None:
            result += ', topic=%s' % self.topic.name
        if self.rating is not None:
            result += ', rating=%s' % self.rating.name
        if self.is_regex:
            result += ', is_regex=%s' % self.is_regex
        result += ')'
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def setRating(self, rating: Rating):
        self.rating = Rating    

# This class defines a lexicon based filter using spacy's word vectoring 
# similarity
#
class Lexicon:
    def __init__(self):
        self.entries: List[LexiconEntry] = []

    
    def append(self, lemma: str, topic: Topic):
        lexicon_entry = LexiconEntry(lemma, topic, None)
        self.entries.append(lexicon_entry)

    # Returns best matching lexicon entry "Topic" for token
    def lexicon_entry_for(self, token: Token) -> LexiconEntry:
        """
        Entry in lexicon that best matches ``token``.
        """
        result = None
        lexicon_size = len(self.entries)
        lexicon_entry_index = 0
        best_matching = 0.0
        while lexicon_entry_index < lexicon_size and not isclose(best_matching, 1.0):
            lexicon_entry = self.entries[lexicon_entry_index]
            matching = lexicon_entry.matching(token)
            if matching > best_matching:
                result = lexicon_entry
                best_matching = matching
            lexicon_entry_index += 1
        if best_matching >= 0.65:
            return result
        else:
            return None    


# Initialize Lexicon Table
lexicon = Lexicon()

# Possible token matches for Service field
lexicon.append('teach'     , Topic.SERVICE)
lexicon.append('tutor', Topic.SERVICE)
lexicon.append('help'      , Topic.SERVICE)
lexicon.append('polite'     , Topic.SERVICE)
lexicon.append('attitude', Topic.SERVICE)
lexicon.append('friendly', Topic.SERVICE)

# Possible token matches for Knowledge field
lexicon.append('knowledge'  , Topic.KNOWLEDGE)
lexicon.append('subject', Topic.KNOWLEDGE)
lexicon.append('smart', Topic.KNOWLEDGE)
lexicon.append('understand', Topic.KNOWLEDGE)

# Possible token matches for Explanation 
lexicon.append('explain', Topic.EXPLANATION)
lexicon.append('patient', Topic.EXPLANATION)
lexicon.append('speak', Topic.EXPLANATION)
lexicon.append('explanation', Topic.EXPLANATION)
lexicon.append('concise', Topic.EXPLANATION)

# Debugging
@app.route('/base')
def showStatus():
    return 'Ok'

# Flask POST handler
# receives reciew from app and processes score
# writes new scores to database
@app.route('/receive-review', methods=['POST'])
def receiveReview():

    requestData = request.get_json()
    review = requestData['review']
    ratingInt = int(requestData['rating'])
    tutorID = requestData['tutorID']
    feedback = nlp_en(review)
    sections = {
        'SERVICE': 0,
        'KNOWLEDGE': 0,
        'EXPLANATION': 0
    }
    for sent in feedback.sents:
        data = [sent.text]
        result = ml.classifiers.classify(model_id, data)
        resultTag = result.body[0]['classifications'][0]['tag_name']
        confidence = result.body[0]['classifications'][0]['confidence']
        for token in sent:
            lexicon_entry = lexicon.lexicon_entry_for(token)
            if lexicon_entry is not None:
                lexicon_entry_topic = lexicon_entry.topic.name
                if ratingInt > 0:
                    # Positive cases
                    if resultTag == "Positive" and ratingInt >= 3:
                        print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)
                        sections[lexicon_entry_topic]+=1
                    # false-positive cases
                    elif resultTag == "Positive" and ratingInt <= 2:
                        if confidence < 0.8:
                            sections[lexicon_entry_topic]-=1
                            print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)
                        else:
                            print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)
                            sections[lexicon_entry_topic]+=1
                    elif resultTag == "Negative" and ratingInt >= 4:
                        # false-negative
                        if confidence < 0.8:
                            sections[lexicon_entry_topic]+=1
                            print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)
                        else:
                            print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)
                            sections[lexicon_entry_topic]-=1
                else:
                    if resultTag == "Positive" and confidence >= 0.6:
                        print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)
                        sections[lexicon_entry_topic]+=1
                    # false-positive cases
                    elif resultTag == "Negative" and confidence >= 0.6:
                        sections[lexicon_entry_topic]-=1
                        print(token, '    ', lexicon_entry_topic, '   ', lexicon_entry)


    return updateDatabase(sections, tutorID, resultTag, ratingInt)

# Updates scores
def updateDatabase(section, tutorID, resultTag, ratingInt):
    db = firebase.database()
    score = db.child("Users/students/"+tutorID+"/rating").get()
    print(score.val())
    scores = {'SERVICE' : 0,
    'KNOWLEDGE': 0,
    'EXPLANATION': 0,
    'overall': 0,
    'count': 0}
    for rating in score.each():
        print(rating.val())
        print(rating.key())
        update = {rating.key(): rating.val()}
        scores.update(update)

    for x,y in section.items():
        if y > 0:
            newScore = scores[x]+1
            db.child("Users/students/"+tutorID+"/rating/"+x).set(newScore)
        elif y < 0:
            newScore = scores[x]-1
            db.child("Users/students/"+tutorID+"/rating/"+x).set(newScore)
    overall = scores['overall']
    numberOfReviews = scores['count']
    if ratingInt > 0:
        newRating = ((overall * numberOfReviews) + ratingInt)/(numberOfReviews+1)
        db.child("Users/students/"+tutorID+"/rating/overall").set(newRating)

    db.child("Users/students/"+tutorID+"/rating/count").set(numberOfReviews+1)
    
    return '''
           The sections values is: {}
           Sentiment: {}
           Total Rating: {}
           The tutorID is: {}'''.format(section.items(), resultTag, newRating, tutorID)

@app.route('/json-example')
def jsonexample():
    return 'Todo...'
