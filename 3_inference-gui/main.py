import numpy as np
import torch
import json
import nltk
from torch import nn
from nltk.stem.porter import PorterStemmer
from tkinter import Tk, Label, Text, Scrollbar, Entry, Button

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

use_gpt3 = 1
if use_gpt3:
    import openai
    api_key = None
else:
    import random
    
with open('intents.json', 'r') as f:
    intents = json.load(f)
    

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
    
class ChatBot:
    def __init__(self, data=torch.load("model.pth"), use_gpt3=0):
        self.use_gpt3 = use_gpt3
        self.tags = data['tags']
        self.words = data['all_words']
        self.context = ""
        self.bot_name = "Chazbot"
        self.start_token = f"\n{self.bot_name}:"
        self.restart_token = "\nYou:"
        
        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(device)
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
    
    def get_response(self, sentence, conversation):
        sentence = self.tokenize(sentence)
        X = self.bag_of_words(sentence)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]


        for intent in intents['intents']:
            if tag == intent['tag']:
                if self.use_gpt3:
                    if prob.item() > 0.5:
                        self.context = intent['context']
                    openai.api_key = api_key
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

BG_RED = "#F54242"
BG_COLOR = "#F5F5F5"
TEXT_COLOR = "#000"

FONT = "Arial 10"
FONT_BOLD = "Arial 13 bold"


class ChatApp:
    def __init__(self):
        self.window = Tk() # init Tk widget
        self.chatbot = ChatBot(use_gpt3=use_gpt3)
        self.conversation = []
        self.greeting = f"{self.chatbot.start_token} Hello there. I am {self.chatbot.bot_name}, designed to answers general questions you may have about Charlie!"
        self._init_params()
        self._setup_main_window()
        self._greeting()

    def run(self):
        self.window.mainloop()
        
    def _greeting(self):
        msg_bot = f"{self.greeting}\n"
        self.conversation.append(f"{self.greeting}")
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", msg_bot)
        self.text_widget.configure(state="disabled")   
        
    def _init_params(self):
        self.width = 400
        self.height = 500
        self.line_width = int(self.width * 0.95)
        
    def _setup_main_window(self):
        self.window.title(self.chatbot.bot_name)
        self.window.resizable(width=False, height=False)
        self.window.configure(width=self.width, height=self.height, bg=BG_COLOR)
        
        # Head Label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Now chatting with Chazbot", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        
        # Separator line
        line = Label(self.window, width=self.line_width, bg=BG_RED)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # Text widget
        self.text_widget = Text(self.window, width=20, height=2, wrap='word',
                                bg=BG_COLOR, fg=TEXT_COLOR, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(state="disabled")
        
        # Scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # Bottom Label
        bottom_label = Label(self.window, bg=BG_COLOR, height=80)
        bottom_label.place(relwidth=1, rely=0.82)
        
        # Message Entry Box
        self.msg_entry = Entry(bottom_label, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.75, relheight=0.05, relx=0.05, rely=0.01)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # Send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, bd=0,
                             width=20, bg=BG_RED, fg=BG_COLOR, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.8, relheight=0.05, relwidth=0.15, rely=0.01)
        
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg)
        
    def _insert_message(self, msg):
        if not msg:
            return
        
        if len(msg) > 80:
            res = "Messages can be no longer than 80 characters!"
            self.conversation.append(f"{self.chatbot.start_token} {res}")
            msg_bot = f"{self.chatbot.start_token} {res}\n"
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", msg_bot)
            self.text_widget.configure(state="disabled") 
            return
        
        self.msg_entry.delete(0, "end")
        self.conversation.append(f"{self.chatbot.restart_token} {msg}")
        msg_user = f"{self.chatbot.restart_token} {msg}\n"
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", msg_user)
        self.text_widget.configure(state="disabled")
        
        res = self.chatbot.get_response(msg, self.conversation)
        self.conversation.append(f"{self.chatbot.start_token} {res}")
        msg_bot = f"{self.chatbot.start_token} {res}\n"
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", msg_bot)
        self.text_widget.configure(state="disabled") 
        
        self.text_widget.see("end")
        

if __name__ == '__main__':
    app = ChatApp()
    app.run()