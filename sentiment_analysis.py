# Takes about ~10 sec to run

import sys
import csv
import re
import string
from collections import defaultdict
from sklearn import linear_model

args = sys.argv

if len(args) != 3:
    print "ERROR: incorrect number of parameters. To run this program you must provide two parameters (csv files corresponding to training data and testing data, respectively)."

with open(args[1], 'rb') as csvfile:
    train = list(list(row) for row in csv.reader(csvfile, skipinitialspace=True))
with open(args[2], 'rb') as csvfile:
    test = list(list(row) for row in csv.reader(csvfile, skipinitialspace=True))

def build_features(tweet):
    d = defaultdict(int)
    for word in tweet:
        d[word.lower()] += 1
    features = dict((k, float(d[k])) for k in most_common).values()
    return features

test_sentiments = []
test_tweets = []
train_sentiments = []
train_tweets = []
feature_list = []
for i in range(len(test)):
    test_sentiments.append(test[i][0])
    test_tweets.append(re.findall(r"[\w']+|[" + string.punctuation + "]", test[i][1]))
for i in range(len(train)):
    train_sentiments.append(train[i][0])
    train_tweets.append(re.findall(r"[\w']+|[" + string.punctuation + "]", train[i][1]))
    feature_list.append(re.findall(r"[\w']+|[" + string.punctuation + "]", train[i][1]))
feature_list = [item for sublist in feature_list for item in sublist]
d = defaultdict(int)
for word in feature_list:
    d[word.lower()] += 1
most_common = sorted(d, key = d.get, reverse = True)[:1000]
test_features = []
train_features = []
for i in range(len(test_tweets)):
    test_features.append(build_features(test_tweets[i]))
for i in range(len(train_tweets)):
    train_features.append(build_features(train_tweets[i]))

algo = linear_model.LogisticRegression()
algo.fit(train_features, train_sentiments)
hypotheses = algo.predict(test_features)
count = 0
for i in range(len(hypotheses)):
    if hypotheses[i] != test_sentiments[i]:
        count +=1
print "Misclassification Rate =", float(count)/len(hypotheses)
