# Sentiment-Analysis-on-Hindi-Reviews
 In this project, sentiment analysis is performed on Hindi  movie reviews. Sentiment analysis is a text classification problem in which text is assigned as positive, negative or neutral depending on sentiment which it strongly forces. In this two different classification approaches: Resource based classification using HindiSentiWordNet (H-SWN) and In-language classification are discussed.
 
# Requirements
1. python3
2. Scikit-learn
3. Numpy, pandas
4. NLTK
5. pickle
6. codecs

# Methods
We have used two approaches to classify the sentiment of hindi reviews as positive or negative.
1. RESOURCE BASED CLASSIFICATION using H-SWN: In this approach we used Hindi-SentiWordNet to classify the sentiment of hindi movie reviews.
2. IN LANGUAGE CLASSIFICATION using various classifiers: This approach is semantic analysis based on training the classifiers on the same language as text.

# Dataset
We have used a total of 1000 Hindi movie reviews for Sentiment analysis. We have taken 250 labelled reviews from IIT-Bombay which contai 125 positive reviews and 125 negative reviews. In addition, we have manually collected 750 reviews from a Hindi movie review website(jagaran.com) and labelled them as positive or negative manually. Out of 750 reviews collected 375 reviews are positive and 375 reviews are negative.

# Files
classifiers.py : This module is used to do In-language classification. It applies different types of classifiers on the featureset generated using Bag of word model with feature value as TermFrequency or Term-Frequency-Inverse-Document_Frequency(TFIDF).

ResourceBasedSentimentClassification.py: This module is used to do Resource based sentiment classification of hindi reviews using HindiSentiWordnet as a resource.

UnigramTfFeatureGeneration.py: This module is used to generate Unigram+Tf Featureset of reviews. 

UnigramTfidfFeaturesetGeneration.py: This module is used to generate Unigram+Tfidf Featureset of reviews.

pos_hindi.txt: This contains positive hindi reviews of dataset. Reviews are seperated by $. 

neg_hindi.txt: This contains negative Hindi reviews of dataset. 

# How to Run
Run on terminal 'python ResourceBasedSentimentClassification.py' to do the sentiment classification through HindiSentiwordnet. It is called Resource Based Semantic analysis.

Run on terminal 'python classifiers.py' to do In language Semantic Analysis.

# Result
Resource Based classification is a simple binary classification which resulted with an accuracy of 54%. The unigram model of In-language classification resulted with an accuracy of 70% using voting classifier and the TF-IDF model of In language classification resulted with best accuracy of 90.85% using decision tree classifier which is higher than the unigram model.

