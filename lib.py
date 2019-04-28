import random
import csv
import math
import codecs
import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from IPython.display import HTML, display

SMOOTH_CONST = 1e-8
TRAIN_SPLIT = 0.8

sentiment_labels = ["positive", "negative"]
usefulness_labels = ["useful", "useless"]


class Review(object):
    def __init__(self, review_content, sentiment, usefulness):
        self.token_list = word_tokenize(review_content)
        self.token_list = [t.lower() for t in self.token_list]  # lowercase
        self.token_set = set(self.token_list)
        self._bigram_list = [(self.token_list[idx], self.token_list[idx+1])
                             for idx in range(len(self.token_list) - 1)]
        self._feature_set = set(self._bigram_list).union(self.token_set)
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
    Returns two lists of reviews: the train set and the test set.
    The possible values for mode are "text", "summary" and "combined".
    Valid initials can be accepted.
        "text": classify based on Text
        "summary": classify based on Summary
        "combined": classify based on Summary + Text
    """
    train_reviews = read_csv(train_path, mode)
    test_reviews = read_csv(test_path, mode)
    return train_reviews, test_reviews


def show_confusion_matrix(predictions):
    """
    Display a confusion matrix as an HTML table.
    rows are true label; columns are predicted labels.
    predictions is a list of (review, predicted_sentiment) pairs
    """
    num_sen = len(sentiment_labels)
    conf_mat = np.zeros((num_sen, num_sen), dtype=np.int32)
    for review, predicted_sentiment in predictions:
        gold_idx = sentiment_labels.index(review.sentiment)
        predicted_idx = sentiment_labels.index(predicted_sentiment)
        conf_mat[gold_idx, predicted_idx] += 1
    df = pd.DataFrame(data=conf_mat, columns=sentiment_labels,
                      index=sentiment_labels)
    display(HTML(df.to_html()))


def class2color_style(s):
    class2color = {
        'positive': 'red',
        'neutral': 'pink',
        'negative': 'green',
        'useful': 'purple',
        'useless': 'blue',
    }
    try:
        return "color: %s" % class2color[s]
    except KeyError:
        return "color: black"


def show_reviews(reviews, search_term=None, head=True):
    """
    Displays an HTML table of reviews alongside labels
    Only displays the head of the table by default
    """
    if search_term is not None:
        reviews = [t for t in reviews if search_term in str(t).lower()]
    columns = ['Content', 'Sentiment', 'Usefulness']
    data = [[str(t), t.sentiment, t.usefulness] for t in reviews]
    pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame(data, columns=columns)
    if head:
        df = df.head()
    s = df.stype.applymap(class2color_style)\
                .set_properties(**{'text-align': 'left'})
    display(HTML(s.render()))


def show_predictions(predictions, show_mistakes_only=False, head=True):
    """
    Displays an HTML table comparing true sentiment labels to predicted
    sentiment labels. predictions is a list of (review, 
    predicted_sentiment) pairs
    Only displays the head of the table by default
    """
    if show_mistakes_only:
        predictions = [(t, p) for (t, p) in predictions if t.sentiment != p]
    columns = ['Content', 'True sentiment', 'Predicted sentiment']
    data = [[str(t), t.sentiment, predicted_sentiment]
            for t, predicted_sentiment in predictions]
    pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame(data, columns=columns)
    if head:
        df = df.head()
    s = df.style.applymap(class2color_style)\
                .set_properties(**{'text-align': 'left'})
    display(HTML(s.render()))


def most_discriminative(reviews, token_probs, prior_probs):
    """
    Prints, for each sentiment, which tokens are most discriminative
    i.e. maximize P(sentiment|token), including normalization
    by P(token)
    """
    all_tokens = set(
        [token for review in reviews for token in review.token_set])

    # maps token to a probability distribution over sentiment labels
    # for a review containing just this token

    token2dist = {}

    for token in all_tokens:
        single_token_review = Review(token, "", "")
        log_dist = {c: get_log_posterior_prob(
            single_token_review, prior_probs[c], token_probs[c]) for c in sentiment_labels}
        min_log_dist = min(log_dist.values())
        # shift so the smallest value is 0 before taking exp
        # "+" corrected as "-" compared with the original version
        log_dist = {c: l - min_log_dist for c, l in log_dist.items()}
        dist = {c: math.exp(l) for c, l in log_dist.items()}
        s = sum(dist.values())
        dist = {c: dist[c] / s for c in sentiment_labels}
        token2dist[token] = dist

    # for each sentiment print the tokens that maximize P(c|token)
    # (normalized by P(token))
    print("MOST DISCRIMINATIVE TOKEN: \n")
    for c in sentiment_labels:
        probs = [(token, dist[c]) for token, dist in token2dist.items()]
        probs = sorted(probs, key=lambda x: x[1], reverse=True)
        print("{0:20} {1:10}".format("TOKEN", "P(%s|token)" % c))
        for token, p in probs[:10]:
            print("{0:20} {1:.4f}".format(str(token), p))
        print()


def get_sentiment_f1(predictions, c):
    """
    Inputs:
        predictions: a list of (review, predicted_sentiment) pairs
        c: a sentiment
    Calculate the precision, recall and F1 for a single sentiment
    c (e.g., positive)
    """

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for review, predicted_sentiment in predictions:
        true_sentiment = review.sentiment
        if true_sentiment == c and predicted_sentiment == c:
            true_positives += 1
        elif true_sentiment == c and predicted_sentiment != c:
            false_negatives += 1
        elif true_sentiment != c and predicted_sentiment == c:
            false_positives += 1

    if true_positives == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = true_positives * 100 / (true_positives + false_positives)
        recall = true_positives * 100 / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)

    print(c)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print()
    # print("Class %s: precision %.2f, recall %.2f, F1 %.2f" % (c, precision, recall, f1))

    return f1


def evaluate(predictions):
    """
    Calculate average F1
    """
    average_f1 = 0
    for c in sentiment_labels:
        f1 = get_sentiment_f1(predictions, c)
        average_f1 += f1

    average_f1 /= len(sentiment_labels)
    print("Average F1: ", average_f1)


def calc_probs(reviews, c):
    """
    Input:
        reviews: a list of reviews
        c: a string representing a sentiment
    Returns:
        prob_c: the prior probability of sentiment c
        feature_probs: a Counter mapping each feature to P(feature|c)
    """
    num_reviews = len(reviews)
    num_reviews_about_c = len([t for t in reviews if t.sentiment == c])
    prob_c = num_reviews_about_c / num_reviews
    feature_counts = Counter()  # maps token -> count and bigram -> count
    for review in reviews:
        if review.sentiment == c:
            for feature in review._feature_set:
                feature_counts[feature] += 1
    feature_probs = Counter(
        {feature: count / num_reviews_about_c for feature, count in feature_counts.items()})
    return prob_c, feature_probs


def learn_nb(reviews):
    feature_probs = {}
    prior_probs = {}
    for c in sentiment_labels:
        prior_c, feature_probs_c = calc_probs(reviews, c)
        feature_probs[c] = feature_probs_c
        prior_probs[c] = prior_c
    return prior_probs, feature_probs


def get_log_posterior_prob(review, prob_c, feature_probs_c):
    """
    Calculate the posterior P(c|review)
    (Actually, calculate something proportional to it).

    Inputs:
        review: a review
        prob_c: the prior probability of sentiment c
        feature_probs_c: a Counter mapping each feature to P(feature|c)
    Return:
        The posterior P(c|review)
    """
    log_posterior = math.log(prob_c)
    for feature in review._feature_set:
        if feature_probs_c[feature] == 0:
            log_posterior += math.log(SMOOTH_CONST)
        else:
            log_posterior += math.log(feature_probs_c[feature])
    return log_posterior


def classify_nb(review, prior_probs, token_probs):
    """
    Classifies a review. Calculates the posterior P(c|review)
    for each sentiment label c, and returns the sentiment with
    the largest posterior.
    Input:
        review
    Output:
        string equal to most-likely sentiment for this reviews
    """
    log_posteriors = {c: get_log_posterior_prob(
        review, prior_probs[c], token_probs[c]) for c in sentiment_labels}
    return max(log_posteriors.keys(), key=lambda c: log_posteriors[c])


def visualize_review(review, prior_probs, token_probs):
    """
        Visualizes a review and its probabilities in an IPython notebook.
        Input:
            review: a review as a string
            prior_probs: priors for each sentiment
            token_probs: a dictionary of Counters that contain the unigram
                probabilities for each sentiment
    """

    # boileplate HTML part 1
    html = """
    <div id="viz-overlay" style="display:none;position:absolute;width:250px;height:110px;border: 1px solid #000; padding:8px;  background: #eee;">
	<p>
       <span style="color:red;">P(<span class="viz-token-placeholder"></span> | positive) = <span id="viz-p-positive"></span></span><br>
	   <span style="color:green;">P(<span class="viz-token-placeholder"></span> | negative) = <span id="viz-p-negative"></span><br>      
    </p>
    </div>

    <div id="viz-review" style="padding: 190px 0 0;">
    """

    tokens = review.token_list
    for token in tokens:
        probs = [token_probs['positive'][token],
                 token_probs['negative'][token]]
        idx = np.argmax(probs) if sum(probs) > 0 else 0
        max_class = sentiment_labels[idx]

        html += '<span style="%s" class="viz-token" data-positive="%f" data-negative="%f">%s</span> ' \
            % (class2color_style(max_class), token_probs['positive'][token], token_probs['negative'][token], token)

    # Predicted sentiment
    predicted_sentiment = classify_nb(review, prior_probs, token_probs)
    html += '<p><strong>Predicted sentiment: </strong> <span style="%s"> %s</span><br>' \
        % (class2color_style(predicted_sentiment), predicted_sentiment)
    html += '<strong>True sentiment: </strong> <span style="%s"> %s</span></p>' \
        % (class2color_style(review.sentiment), review.sentiment)

    # Javascript
    html += """
    </div>
     <script type="text/javascript">
	$(document).ready(function() {
		$("span.viz-token").mouseover(function() {
			$("span.viz-token").css({"font-weight": "normal"});
			$(this).css({"font-weight": "bold"});
			$("span.viz-token-placeholder").text($(this).text());
			$("#viz-p-positive").text($(this).data("positive"));
			$("#viz-p-negative").text($(this).data("negative"));
			$("#viz-overlay").show();
			$("#viz-overlay").offset({left:$(this).offset().left-110+$(this).width()/2, top:$(this).offset().top - 140});
		});
	});
    </script>

    """

    display(HTML(html))
