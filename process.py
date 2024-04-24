
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import nltk
import pickle
import re
import string
  
# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


def process(file_path):
    data = pd.read_csv(file_path)

    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Comments"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Comments"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Comments"]]
    data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["Comments"]]
    score = data["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05 :
            sentiment.append('Positive')
        elif i <= -0.05 :
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')
    data["Sentiment"] = sentiment
    data.head()

    final_data=data.drop(['Positive','Negative','Neutral','Compound'],axis=1)



    stop_words = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemer = SnowballStemmer(language="english")
    lzr = WordNetLemmatizer()

    def text_processing(text):
        try:
            # convert text into lowercase
            text = text.lower()

            # remove new line characters in text
            text = re.sub(r'\n',' ', text)

            # remove punctuations from text
            text = re.sub('[%s]' % re.escape(punctuation), "", text)

            # remove references and hashtags from text
            text = re.sub("^a-zA-Z0-9$,.", "", text)

            # remove multiple spaces from text
            text = re.sub(r'\s+', ' ', text, flags=re.I)

            # remove special characters from text
            text = re.sub(r'\W', ' ', text)

            # tokenize the text
            words = word_tokenize(text)

            # remove stop words and lemmatize words
            words = [lzr.lemmatize(word) for word in words if word not in stop_words]

            # join the words back into a single string
            processed_text = ' '.join(words)

            return processed_text
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

    data_copy = final_data.copy()
    data_copy.Comments = data_copy.Comments.apply(lambda text: text_processing(text))

    le = LabelEncoder()
    data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

    processed_data = {
        'Sentence':data_copy.Comments,
        'Sentiment':data_copy['Sentiment']
    }

    processed_data = pd.DataFrame(processed_data)

    df_neutral = processed_data[(processed_data['Sentiment']==1)]
    df_negative = processed_data[(processed_data['Sentiment']==0)]
    df_positive = processed_data[(processed_data['Sentiment']==2)]


    # Concatenate the upsampled dataframes with the neutral dataframe
    final_data = pd.concat([df_negative,df_neutral,df_positive])

    corpus = []
    for sentence in final_data['Sentence']:
        corpus.append(sentence)
    corpus[0:5]

   

    # Load the trained CountVectorizer and GaussianNB model
    with open('count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    try:
        with open('naive_bayes_model.pkl', 'rb') as f:
            classifier = pickle.load(f)
    except EOFError:
        print('Error loading the Naive Bayes model. The file may be corrupted.')

    # Assuming 'new_data' is your new input data
    # Preprocess the new data
    new_data_processed = [text_processing(corpus[i]) for i in range(len(corpus))]

    # Vectorize the preprocessed data using the same CountVectorizer
    new_data_vectorized = cv.transform(new_data_processed).toarray()

    # Make predictions using the trained classifier
    predictions = classifier.predict(new_data_vectorized)

    # Count the number of positive and negative comments
    positive_comments = sum(predictions == 2)  # Assuming '2' represents 'Positive' class
    negative_comments = sum(predictions == 0)  # Assuming '0' represents 'Negative' class
    neutral_comments = sum(predictions == 1)  # Assuming '1' represents 'Negative' class

    # Print the number of positive and negative comments
    return positive_comments, negative_comments, neutral_comments
    
 

