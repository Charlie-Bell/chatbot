# This script runs the chat app so that the chatbot can be used.
# To use GPT-3, enable and set the API key in src/chatbot.py.

import sys
sys.path.append("../src/")
from chatapp import ChatApp

if __name__ == '__main__':
    app = ChatApp()
    app.run()