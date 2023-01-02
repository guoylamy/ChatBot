# ChatBot -- A Conversational Based Agent

1) Introduction
Set up the problem: 
Our conversational based agent – ChatChat, is an end-to-end conversational agent, which takes a string and outputs a response in string format.  
Inputs: A set of strings representing a sentence
Outputs: A set of strings representing responses to the input sentence.
We will use conversations from movie scripts collected from https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html as our dataset. 
After extracting and preprocessing the conversations, we use 75% of dialogues to train our RNN model, and use the rest 25% dialogues to do tests.
To evaluate the chatbot performance, we are going to use 2 metrics:
1.	The similarity between the response from chatbot and the response from the movie script. The “similarity” is measured by atomic match (ratio of matched words to the original sentence length) and most significant match (ratio of matched nonstop words to the original sentence length). 
2.	The reasonableness of chatbot response. With libraries such as “language-check”, we can get the number of grammar errors in a sentence. The lower the ratio of the number of grammar errors to the original sentence length, the higher the quality of a sentence.

Motivation: 
Towards solving this problem, we can develop a chatbot to simulate conversations with users in English. This can help users find “someone” to chat with when they are bored or feel lonely. What’s more, since we are training this chatbot with a dataset from movies, some responses would be dramatic and bring more happiness to users. 
2) How We Have Addressed Feedback From the Proposal Evaluations
From the proposal feedback, we realized that an appropriate conversation dataset is crucial for our project. Since RNNs take a fair amount of time to train, each piece of training data should be precise and tidy, and the dataset should consist of ample amounts of data for effective training. We could not find pre-trained language models specifically for COVID-19, nor could we find suitable COVID-19 Q&A dialogue dataset from Kaggle. We also intended to crawl the text from briefings of CDC and whitehouse related to COVID-19. However, we found that the briefings usually come with hundreds of words for each answer, which would significantly increase the amount of time for training and also raise the difficulty of accurate response, thus we need to narrow down the scope of this project. Moreover, we find that the information related to COVID-19 is constantly updating, from the data about infections, vaccination, to the regulations and treatments. A one-time training will absolutely become outdated as the time passes, which will harm the effectiveness of our model. Since this is a serious topic and users need time sensitive professional answers, the response about that topic should be as accurate as possible. If we just use a small and outdated dataset to train, the model is imprecise and could be misleading. Therefore, we decided to change the conversation topic from COVID-19 to movie since it is more entertaining and has certain fault tolerance. 

We previously thought we could use user’s satisfaction as a metric, just like some chatbots do on shopping websites. Maybe we could give a rate to the generated response by ourselves manually.  But after carefully thinking about the mentor's suggestions, we decided not to use this metric since this might be a heavy workload and too subjective. We think the remaining two metrics – dialogue efficiency (similarity) and quality (reasonableness) metric – are enough to test the model performance.

3) Prior Work We are Closely Building From
1.	Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." arXiv preprint arXiv:1406.1078 (2014).
Link: https://arxiv.org/abs/1406.1078
Our project is divided into two parts: understand and answer, which both deal with the sequence of linguistic phrases.  The RNN Encoder-Decoder model proposed by this article encodes a sequence of symbols to a fixed-length vector representation, and decodes the representation into another sequence of symbols, which are jointly trained together to maximize the conditional probability of a target sequence given a source sequence. This model will be used in our project to give a semantically and syntactically meaningful response.
There is a tutorial about the implementation of this article. It will not be directly integrated into our code, but will help us understand the details of the workflow of this model.
Link:https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb
2.	Cho, Kyunghyun, et al. "On the properties of neural machine translation: Encoder-decoder approaches." arXiv preprint arXiv:1409.1259 (2014).
Link: https://arxiv.org/abs/1409.1259
GRU proposed by this paper simplifies the LSTM while maintaining its ability to remember long term dependencies. Its performance is similar to LSTM but trains much faster and requires less memory. It will be a good reference for us to implement our project with the limited computation resources and time.

4) What We are Contributing
1.	Contribution(s) in Code: N/A
2.	Contribution(s) in Application: 
a.	With the development of NLP, the chatbot has been intensively adapted by companies to answer customers’ questions in a fast and efficient way while reducing human cost.
b.	Build a chatbot that is capable of answering questions in terms of specific topics such as Covid19, etc. 
3.	Contribution(s) in Data: N/A
4.	Contribution(s) in Algorithm: N/A
5.	Contribution(s) in Analysis: N/A
5) Detailed Description of Each Proposed Contribution, Progress Towards It, and Any Difficulties Encountered So Far
5.1 Project Basics
The chatbot project mainly consists of three parts:
1.	Preprocess the data collected into a format of dialogue with one question corresponding to one or more answers, apply text tokenization to dialogues and encode those tokenized texts(not sure now, but we will figure it out after learning RNN and LSTM this week).
2.	Train a classifier based on Long Short-Term memory with recurrent neural network architecture as well as learning the hyperparameters such as learning rate, batch size, etc.
3.	Inference the input based on our trained model.
5.2 Project Progress
Currently, all group members are focusing on finding appropriate datasets to train our RNN model. Since we want to build a chatbot to answer covid19 specific questions, we find no open datasets on Kaggle.com that have data in format of dialogue. We even try to scrape the data from some authoritative websites such as white House, however, the dialogue data is far from enough and most questions and answers are so long that we may not get a good model from it. 
We decided to first train a chatbot based on simple dialogue such as dialogues in a movie and the dataset we found is Cornell Movie Dialogues Corpus described in section 1. The format of the dialogue is a little bit complicated and cannot be directly used to feed our training model. We’ve written a data cleaning script to convert the raw data into question-answer pairs. 
Since we haven’t learned the LSTM and RNN, we cannot build our model right now, but we watched some LSTM introductory videos and RNN papers we mentioned above to learn it by ourselves. 

5.3 Difficulties
1.	Datasets: there is no dialogue-format data in Kaggle about Covid19, if we still want to build a chatbot capable of answering covid19 related questions, we need to build a dataset by ourselves and we are not quite sure where to find enough data and what the standard of a good dataset is.
2.	We haven’t implemented and trained our model but one potential problem is what the test results would look like. Is it possible that the test results will give a sentence that is semantically meaningless? If possible, how can we avoid such problems from happening? Increasing the volume of datasets ? 

6)  Risk Mitigation Plan
●	Q: How will we modularize the implementation of this project for better workload distribution, faster development, and debugging?
A: Currently we intend to decouple this project into three parts: data preprocessing, RNN model with LSTM, inference. Each of our three group members will mainly take charge of one module. The data preprocessing will transfer the complicated dialogue to Q&A format, and then convert them to the representation of the vector or matrix for model training. The model building will mainly refer to the ideas of the two reference papers. The inference module takes the output weight matrix of model and user inputs and generates an accurate and meaningful response. Each of this module will conduct unit testing before integration testing for more efficient debugging.
●	Q: How will we avoid the case that after spending quite a lot of time developing and training, the performance and results turn out to be disappointing, but there is not enough time left for us to rebuild?
A: We will firstly train and test on a simpler “toy” dataset to get some early results and bug-fixing before we train on the completed dataset. Training and testing on the “toy” dataset will consume much less time and give us a direct feedback of whether the result is grammatically correct and semantically meaningful. We do not intend to get a very satisfying result from the toy dataset. It will help us build a runnable and meaningful model that has no fatal flaws before we train on the time-consuming and completed dataset.
●	Q: How will we scale our objectives based on time and progress to avoid ending up with nothing?
A: The most basic and feasible objective for this project is to build a workable chatbot that makes sense. For this objective, we will test with questions that have limited length, e.g. 10 words, and the questions similar to the ones appeared in training. The answers would not need to be very precise, but should be semantically correct and syntactically meaningful. As for the improvement, we intend to test with longer and complexer questions, including those of the topic not appearing in the dataset. The answers should be precise and match people’s normal dialogue habits.
●	Q: What if we need too many computation resources?
A: We will firstly train with a limited-sized toy dataset to get a brief view of computation resources that we need. We will limit the size of the completed dataset for formal training within our AWS credits. We can also limit the number of epochs for training to a reasonable scale to reduce cost.
