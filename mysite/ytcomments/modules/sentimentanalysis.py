# %%
import csv
import string
import re
import pandas as pd
import numpy as np
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import os
from django.conf import settings
BASE_DIR = settings.BASE_DIR

# %%
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# %%
import nltk
nltk.download("stopwords")

# %%
tokenizer = RegexpTokenizer(r'\w+')

# %%
en_stopwords = set(stopwords.words("english"))

# %%
ps = PorterStemmer()

# %%
def getCleanData(text):
    # Converting sting into lower case
    text = str.lower(text)
    # Removing urls from the text
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text)  
    # Generating Patterns for Emoji
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)  # removing emoticons
    # Removing Emoji from the string
    text = emoji_pattern.sub(r'', text)
    # removes the digits from the string
    text = re.sub(" \d+", " ", text)
    # Removing the white spaces from the beginning and end of string
    text = text.strip()
    # Tokenizing the text
    tokens = tokenizer.tokenize(text)
    # Removing the stop words
    new_tokens = [token for token in tokens if token not in en_stopwords]
    # Stemming
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text

# %%
# Reading Comments from the csv
comments_csv_path = os.path.join(BASE_DIR, 'ytcomments/modules/comments.csv')
comments = pd.read_csv(comments_csv_path, sep='\t', names=['comment'])
# Loading data from the csv to dataframe and then cleaning the data
df = comments['comment'].apply(getCleanData)
df.replace('\n', 'NaN')
df.replace(' ', 'NaN')

# %%
df = df.to_numpy()

# %%
i=0
# Creating a new csv file with cleaned data
cleandedComments_csv_path = os.path.join(BASE_DIR, 'ytcomments/modules/cleanedComments.csv')
with open(cleandedComments_csv_path, "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i in range(0, 450):
        str1 = ''.join(str(e) for e in df[i])
        if (str1 != '\n' and str1 != ''):
            csv_file.write(str1 + "\n")
        else:
            continue

# %%
# Getting the training and testing data
dataset_csv_path = os.path.join(BASE_DIR, 'ytcomments/modules/DataSet.csv')
cleandedComments_csv_path = os.path.join(BASE_DIR, 'ytcomments/modules/cleanedComments.csv')
data_train = pd.read_csv(dataset_csv_path, encoding="latin-1")
data_testing = pd.read_csv(cleandedComments_csv_path, encoding="latin-1", names=["Comment"])
labels = data_train.Sentiment

# %%
X = data_train.SentimentText.apply(getCleanData)
y = data_train.Sentiment

# %%
y

# %%
stopset = set(stopwords.words("english"))

# %%
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents="ascii", stop_words=stopset)

# %%
X = vectorizer.fit_transform(X)

# %%
print(y.shape)
print(X.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.60, random_state=20)

# %%
clf = MultinomialNB()

# %%
clf.fit(X_train, y_train)

# %%
from sklearn.metrics import roc_auc_score

# %%
accuracy = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]) * 100
accuracy

# %%
from ytcomments.modules.comments import commentExtract
def analysizeComments(videoID, commentsCount):
    comments_list = commentExtract(videoID, commentsCount)
    cleaned_comments = [getCleanData(text) for text in comments_list]
    comments_vector = vectorizer.transform(cleaned_comments)
    clf.predict(comments_vector)
    result = {}
    for i in range(len(comments_list)):
        result[i] = {
            comments_list[i]: clf.predict(comments_vector)[i]
        }
    return result
