<h2><center>Chazbot</h2></center>

Chazbot is

A demo was created... An API key is required...

[demo](https://drive.google.com/file/d/151Omr83xMpLAuhMKR7VuaUct09z9GlPk/view?usp=sharing)


<img src = "https://drive.google.com/uc?export=view&id=1YMNwq3sM4q0WbJ0pElrtwHjO6-KDnbKj">

So what I did was manually write 70 prompt and completion pairs, i.e. questions and answers. I then categorised these 70 pairs into 40 intents, which can be further categorised into 6 contexts.
A sample of one is my 70 pairs would be:
-prompt: how old is Charlie?
-completion: he is 28 years old.
-intent: age.
-context: personal.

With these 70 samples I put them into a GPT-3 without the context or intent, just the prompt. I tasked GPT-3 to rephrase each 5x to get a dataset of about 420 samples.
I then had another transformer model, Pegasus, reword further to generate a total 4620 samples.
I automatically had the new intents and contexts assigned to the new samples.

I then trained a small neural network to do intent classification based on the generated prompts, so that when I give it an unseen prompt, it can still classify the intent.

From those intent classifications, I feed the associated context into GPT-3 along with the unseen prompt to get a new unseen completion which is generated using both the prompt and it's associated context.

A total of 3 notebooks were created for this project:
1. 1_Landslide_prediction_CNN uses a CNN approach to the problem.
2. 2_Landslide_prediction_Boosting uses a boosting and stacking approach to the problem.
3. 3_Landslide_prediction_Combination takes the predictions from the above networks and uses them as features in the a fully connected NN.
