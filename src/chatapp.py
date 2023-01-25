from tkinter import Tk, Label, Text, Scrollbar, Entry, Button
from chatbot import ChatBot

BG_RED = "#F54242"
BG_COLOR = "#F5F5F5"
TEXT_COLOR = "#000"

FONT = "Arial 10"
FONT_BOLD = "Arial 13 bold"


class ChatApp:
    def __init__(self):
        self.window = Tk() # init Tk widget
        self.chatbot = ChatBot()
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
        self.head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Now chatting with Chazbot", font=FONT_BOLD, pady=10)
        self.head_label.place(relwidth=1)
        
        # Separator line
        self.line = Label(self.window, width=self.line_width, bg=BG_RED)
        self.line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # Text widget
        self.text_widget = Text(self.window, width=20, height=2, wrap='word',
                                bg=BG_COLOR, fg=TEXT_COLOR, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(state="disabled")
        
        # Scrollbar
        self.scrollbar = Scrollbar(self.text_widget)
        self.scrollbar.place(relheight=1, relx=0.974)
        self.scrollbar.configure(command=self.text_widget.yview)
        
        # Bottom Label
        self.bottom_label = Label(self.window, bg=BG_COLOR, height=80)
        self.bottom_label.place(relwidth=1, rely=0.82)
        
        # Message Entry Box
        self.msg_entry = Entry(self.bottom_label, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.75, relheight=0.05, relx=0.05, rely=0.02)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # Send button
        self.send_button = Button(self.bottom_label, text="Send", font=FONT_BOLD, bd=0,
                             width=20, bg=BG_RED, fg=BG_COLOR, command=lambda: self._on_enter_pressed(None))
        self.send_button.place(relx=0.8, relheight=0.05, relwidth=0.15, rely=0.02)
                
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