import random
import csv
import math
import codecs
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

sentiment_labels = ["positive", "negative"]
usefulness_labels = ["useful", "useless"]


class Review(object):
    def __init__(self, review_content, sentiment, usefulness):
        self.content = review_content
        self.token_list = word_tokenize(self.content)
        self.sentiment = sentiment
        self.usefulness = usefulness

    def __getitem__(self, index):
        return self.token_list[index]

    def idx(self, token):
        return self.token_list.index(token)

    def __str__(self):
        return ' '.join(self.token_list)

    def __repr__(self):
        return self.__str__()


def read_csv(path, mode):
    data = {}
    if mode in ["text", 't']:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                review_id, summary, text, sentiment, usefulness = row
                assert sentiment in sentiment_labels
                assert usefulness in usefulness_labels
                assert review_id not in data.keys()
                if isinstance(text, str):
                    data[review_id] = Review(text, sentiment, usefulness)
                else:
                    data[review_id] = Review(str(text), sentiment, usefulness)
        data = data.values()
    elif mode in ["summary", 's']:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                review_id, summary, text, sentiment, usefulness = row
                assert sentiment in sentiment_labels
                assert usefulness in usefulness_labels
                assert review_id not in data.keys()
                if isinstance(summary, str):
                    data[review_id] = Review(summary, sentiment, usefulness)
                else:
                    data[review_id] = Review(
                        str(summary), sentiment, usefulness)
        data = data.values()
    else:
        assert mode in ["combined", 'c']
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                review_id, summary, text, sentiment, usefulness = row
                assert sentiment in sentiment_labels
                assert usefulness in usefulness_labels
                assert review_id not in data.keys()
                if isinstance(summary, str) and isinstance(text, str):
                    content = summary + ' ' + text
                elif isinstance(summary, str):
                    content = summary + ' ' + str(text)
                elif isinstance(text, str):
                    content = str(summary) + ' ' + text
                else:
                    content = str(summary) + ' ' + str(text)
                data[review_id] = Review(content, sentiment, usefulness)
        data = data.values()
    return data

def read_data(train_path="data/reviews-after-2010-train.csv",
              test_path="data/reviews-after-2010-test.csv", mode="text"):
    """
    Returns a dict including two keys:
        tfidf and data.
    tfidf -> tfidf_vectorizer
    data -> four lists: x_train, x_test, y_train, y_test.
    The possible values for mode are "text", "summary" and "combined".
    Valid initials can be accepted.
        "text": classify based on Text
        "summary": classify based on Summary
        "combined": classify based on Summary + Text
    """
    train_reviews = read_csv(train_path, mode)
    test_reviews = read_csv(test_path, mode)
    
    train_content = [review.content for review in train_reviews]
    test_content = [review.content for review in test_reviews]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(train_content)  # use "df" of "TF-IDF" from train_content
    
    # x_train and x_test have the same feature set
    x_train = tfidf_vectorizer.transform(train_content)
    x_test = tfidf_vectorizer.transform(test_content)
    
    y_train = [review.sentiment for review in train_reviews]
    y_test = [review.sentiment for review in test_reviews]
    
    return {"tfidf": tfidf_vectorizer, 
            "data": (x_train, x_test, y_train, y_test)}
