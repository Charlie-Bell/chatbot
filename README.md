# Chazbot

## Objective
Chazbot is a GPT-3-based chatbot created to answer questions about Charlie's personal and professional life, allowing companies and recruiters to gather information which may not fit on a resume.

## Background
One big problem with modern chatbots is the lack of human-like generated responses. Early state of the art large language models (LLM), e.g. GPT-2, BERT, are excellent at typical prediction, translation and classification tasks, but still lack in human-like text generation tasks. The question answering capabilities of these early LLMs were not much better than extractive question answering, providing only the bare minimum of text to answer a question.

Generative Pre-trained Transformer 3, or GPT-3, is a 175-billion parameter transformer network trained on 45TB of data, making it one of the largest pretrained transformers in the world. It is capable of few-shot learning making it ideal for working with small datasets and generating sophisticated responses, hence why it was picked for this project.

## Demo
This [video demo](https://drive.google.com/file/d/151Omr83xMpLAuhMKR7VuaUct09z9GlPk/view?usp=sharing) illustrates the capabilities of Chazbot in question answering with state-of-the-art natural language understanding and generation. A free API key is required from openai.com should the user want to talk to Chazbot, as the bot uses GPT-3's API.

<p align="center">
  <img src = "https://drive.google.com/uc?export=view&id=1YMNwq3sM4q0WbJ0pElrtwHjO6-KDnbKj">
</p>

## Methodology
### Cost limitations
GPT-3 is not yet public and can only be accessed through OpenAI's development API. The API has a token limit and is priced around the number of tokens used. Fine-tuning the model on a small dataset is still too expensive for an individual, hence this project aims to reduce the number of tokens used by applying intent classification to the users input in order to pick a context. The context is then fed into GPT-3 with the input - this means only the associated information is given to GPT-3 to generate a response, minimizing token usage. GPT-3 requires no fine-tuning using this method and conversations can be relatively cheap.

### Development
There were 3 main stages in the development of Chazbot: dataset generation, model training and the chatbot application.

#### Dataset generation
To generate the dataset, 70 prompt and completion pairs, i.e. questions and answers, were written manually. These pairs were then categorised into 40 intents, which were then further categorised into 6 contexts.
A sample of one is my 70 pairs would be:
-prompt: how old is Charlie?
-completion: he is 28 years old.
-intent: age.
-context: <paragraph about personal information>

These 70 pairs were put into GPT-3, which was tasked in rephrasing each 5x to get a dataset of about 420 samples.
Then another transformer model, Pegasus, had the job or rewording further to generate a total 4620 samples.
New intents and contexts were then automatically assigned to the new samples.

#### Model training
A small neural network was trained to perform intent classification based on the generated prompts, so that when given an unseen prompt, it can accurately classify the intent.

From those intent classifications, the associated context was fed into GPT-3 along with the prompt to get a new completion which is generated using both the prompt and it's associated context.

## Chatbot application - How to use:
<i>Bear in mind OpenAI do not provide OAuth in their API, therefore use of your own API key is done at your own risk.</i>
To run the chatbot, simply run the 'chatbot' notebook or 'main.py' with a valid API key from openai.com and ensure all the packages are installed.

Two notebooks manage data generation and encoder training:
1. 'dataset-generator-gpt3-pegasus' which generates the 4620 sample dataset from the 70 manual samples.
2. 'training' which trains the model for intent classification with a small neural network.

<b>main.py is used to run the chatbot application.</b> The use needs to supply the API key in order to get GPT-3 generated responses.

## Conclusion
In future, to scale the project, the contexts and intents can be broken down into smaller chunks. Were the pricing of GPT-3 to change, it could even be affordable for an individual. (UPDATE) Now GPT-4 and ChatGPT have been released, perhaps one day these are even accessible. With the new plugins introduced, GPT-4 could even be setup to summarise company databases and websites for FAQs or chatbot purposes, and eventually at an affordable price.

