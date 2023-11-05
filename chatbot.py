
import random
import json
import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
##from spellchecker import SpellChecker
import tensorflow as tf
from tensorflow import keras

lemmatizer = WordNetLemmatizer()
##test lemmatizer
print(lemmatizer.lemmatize('dogs'))

##read json file
intents = json.loads(open('intents.json').read())
print(intents["intents"][1])

from keras.models import load_model

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')



##chop sentence and lemmatize it, option to ass a spell checker
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    #spell = SpellChecker()
    #corrected_sentence = [spell.correction(word)for word in sentence_words]
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

##predict what class of response tag it should give
def predict_class (sentence):
    if sentence.strip() == "":
        # If the input is empty, return a response from the noanswer tag
        return [{'intent': 'noanswer', 'probability': '1.0'}]
    ##run bag of words fn
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

#get response from predicted class and randomise which response to give
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

##print to promt the user to ask a question
print("Ask Mr. PiggyBank Buddy a question!")

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)
    