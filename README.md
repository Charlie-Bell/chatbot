<h2><center>Chazbot</h2></center>

Chazbot is a GPT-3-based chatbot created to answer questions about Charlie's personal and professional life, allowing employers to gather information which may not be available on a resume.

Generative Pre-trained Transformer 3, or GPT-3, is a 175-billion parameter transformer network trained on 45TB of data, making it the largest pretrained transformer in the world. It is capable of few-shot learning making it ideal for working with small datasets, hence why it was picked for this project. The model is not yet public and can only be accessed through OpenAI's development API.

This [video demo](https://drive.google.com/file/d/151Omr83xMpLAuhMKR7VuaUct09z9GlPk/view?usp=sharing) illustrates the capabilities of Chazbot in question answering with state-of-the-art natural language understanding and generation. A free API key is required from openai.com should the user want to talk to Chazbot, as the bot uses GPT-3's API.

<img src = "https://drive.google.com/uc?export=view&id=1YMNwq3sM4q0WbJ0pElrtwHjO6-KDnbKj">

To run the chatbot only the folder '3_inference-gui' is required, simply run the 'chatbot' notebook or 'main.py' with a valid API key from openai.com and ensure all the packages are installed.

There were 3 main stages in the development of Chazbot: dataset generation, model training and inference.

To generate the dataset, 70 prompt and completion pairs, i.e. questions and answers, were written manually. These pairs were then categorised into 40 intents, which were then further categorised into 6 contexts.
A sample of one is my 70 pairs would be:
-prompt: how old is Charlie?
-completion: he is 28 years old.
-intent: age.
-context: <paragraph about personal information>

These 70 pairs were put into GPT-3, which was tasked in rephrasing each 5x to get a dataset of about 420 samples.
Then another transformer model, Pegasus, had the job or rewording further to generate a total 4620 samples.
New intents and contexts were then automatically assigned to the new samples.

A small neural network was then trained to perform intent classification based on the generated prompts, so that when given an unseen prompt, it can accurately classify the intent.

From those intent classifications, the associated context was fed into GPT-3 along with the prompt to get a new completion which is generated using both the prompt and it's associated context.

A total of 3 notebooks were created for this project:
1. 'dataset-generator-gpt3-pegasus' which generates the 4620 sample dataset from the 70 manual samples.
2. 'training' which trains the model for intent classification with a small neural network.
3. 'chatbot' which defines a GUI and sets up the chatbot for inference.
