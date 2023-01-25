import numpy as np
import torch
import nltk
from nltk.stem.porter import PorterStemmer
import random
import openai
import json

from neuralnet import NeuralNet


class ChatBot:
    def __init__(self, data=torch.load("../models/model.pth")):        
        ### SET API KEY HERE FOR GPT3 ###
        self.api_key = None
        ### IF GPT3 DESIRED THEN SET TO 1 ###
        self.use_gpt3 = 0
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.api_key = None
        self.activate_gpt(self.api_key)
        self.load_intents()
        self.tags = data['tags']
        self.words = data['all_words']
        self.context = ""
        self.bot_name = "Chazbot"
        self.start_token = f"\n{self.bot_name}:"
        self.restart_token = "\nYou:"
        
        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(data['model_state'])
        
        
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        stemmer = PorterStemmer()
        return stemmer.stem(word=word.lower())

    def bag_of_words(self, tokenized_sentence):
        tokenized_sentence = [self.stem(w) for w in tokenized_sentence]

        bag = np.zeros(len(self.words), dtype=np.float32)
        for idx, w in enumerate(self.words):
            if w in tokenized_sentence:
                bag[idx] = 1.0

        return bag

    def load_intents(self):
        with open('../data/processed/intents.json', 'r') as f:
            self.intents = json.load(f)

    def activate_gpt(self, api_key=None):

        if api_key:
            self.api_key = api_key
            print("GPT active with API key: " + self.api_key)
        elif self.api_key:
            print("GPT active with API key: " + self.api_key)
        else:
            self.use_gpt3 = 0
            print("Error, GPT disabled, need API key.")
    
    def get_response(self, sentence, conversation):
        sentence = self.tokenize(sentence)
        X = self.bag_of_words(sentence)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]


        for intent in self.intents['intents']:
            if tag == intent['tag']:
                if self.use_gpt3:
                    if prob.item() > 0.5:
                        self.context = intent['context']
                    openai.api_key = self.api_key
                    recent_conversation = "".join(conversation[-6:])
                    prompt=f"Context: {self.context}\n{recent_conversation}{self.start_token}"
                    response = openai.Completion.create(
                        model='text-davinci-002',
                        prompt=prompt,
                        temperature=0.5,
                        max_tokens=256,
                        top_p=1,
                        best_of=1,
                        frequency_penalty=1,
                        presence_penalty=0.2,
                        stop=[self.restart_token]
                    )
                    output = f"{response['choices'][0]['text'].strip()}"
                else:
                    if prob.item() > 0.7: 
                        output = f"{random.choice(intent['responses'])}"
                    else:
                        output = f"I do not understand the question, please rephrase."

        return output