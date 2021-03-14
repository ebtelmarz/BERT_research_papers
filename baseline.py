import numpy as np
import pandas as pd
import csv

# Import packages for pre-processing
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Packages for pretrained glove and tfidf
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sb

import warnings

warnings.filterwarnings('ignore')

# Data Modelling
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogisticRegression': LogisticRegression(n_jobs=-1),
    'kNN': KNeighborsClassifier(n_jobs=-1),
    'SVC': SVC(),
    'RandomForestClassifier': RandomForestClassifier(n_jobs=-1)
}

params = {
    'LogisticRegression': {'solver': ['newton-cg', 'saga', 'sag'],
                           'C': np.logspace(-10, 0, 10)
                           },
    'kNN': {'n_neighbors': [3, 5, 10],
            'weights': ["uniform", "distance"]
            },
    'SVC': {'C': [0.1, 1, 10],
            'gamma': ['auto', 'scale']
            },
    'RandomForestClassifier': {'n_estimators': [10, 50, 100],
                               'criterion': ['gini', 'entropy']
                               }
}

# Load the SNLI dataset
path = '/Users/sergiogarcia/Documents/Thesis/Project/snli_1.0/'
train_data = pd.read_csv(path + 'snli_1.0_train.txt', sep="\t")
train_data = pd.DataFrame(train_data, columns=['sentence1', 'sentence2', 'gold_label'])

# Remove all the rows labeled as '-' and NaN values
train_data = train_data.loc[train_data['gold_label'] != '-']
train_data = train_data.dropna()

# Create a subset of the data in order to run de model faster
train_data = train_data.iloc[:10000, :]
train_data.head()

# Test set
test_data = pd.read_csv(path + 'snli_1.0_test.txt', sep="\t")
test_data = pd.DataFrame(test_data, columns=['sentence1', 'sentence2', 'gold_label'])

# Remove all the rows labeled as '-' and NaN values
test_data = test_data.loc[test_data['gold_label'] != '-']
test_data = test_data.dropna()

# Create a subset of the data in order to run de model faster
test_data = test_data.iloc[:2500, :]
test_data.head()

train_data = train_data.sample(frac=1, random_state=203)

# Size of the datasets
print('Train Data size:', train_data.shape)
print('Test Data size:', test_data.shape)

# Labels distributions
print('==Labels train set:==\n{}'.format(train_data['gold_label'].value_counts()))
print('\n==Labels test set:==\n{}'.format(test_data['gold_label'].value_counts()))

lemmat = WordNetLemmatizer()

def clean_text(sentence, lemma=False):
    # Step 1: Transform text to lower case, remove url, some punctuations and long repeated characters
    sent = sentence.lower()
    sent = re.sub(r'(http:)\S+', r'', sent)
    sent = re.sub(r'[\.,`=_/#]', r' ', sent)
    sent = re.sub(r'(\w)\1{2,}', r'\1\1', sent)

    # Step 2: Find the tokens of each text and apply lemmatization
    tokens = word_tokenize(sent)
    if lemma: tokens = [lemmat.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


# Data Cleaning in both sets
train_data['clean_sent1'] = train_data['sentence1'].map(lambda x: clean_text(x, lemma=True))
train_data['clean_sent2'] = train_data['sentence2'].map(lambda x: clean_text(x, lemma=True))

test_data['clean_sent1'] = test_data['sentence1'].map(lambda x: clean_text(x, lemma=True))
test_data['clean_sent2'] = test_data['sentence2'].map(lambda x: clean_text(x, lemma=True))

# Column with integer labels
dict_label = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
train_data['int_label'] = train_data['gold_label'].map(lambda x: dict_label[x])
test_data['int_label'] = test_data['gold_label'].map(lambda x: dict_label[x])

# Preview results after pre-processing
train_data.iloc[:3]

# Loading the 300-dimension vectors
path = "/Users/sergiogarcia/Documents/Thesis/Data/glove.6B/glove.6B.300d.txt"

glove_file = datapath(path)
glove_vec_file = get_tmpfile("glove.6B.300d.txt")
glove2word2vec(glove_file, glove_vec_file)
word_vectors = KeyedVectors.load_word2vec_format(glove_vec_file)

# Size of the data vocabulary
all_sent = np.concatenate([train_data['clean_sent1'].values, train_data['clean_sent2'].values], axis=0)
words = []
for i in all_sent:
    tokens = word_tokenize(i.lower())
    words.extend(tokens)
vocab = list(set(words))

print('Size of the data vocab:', len(vocab))

# Size of the GloVe vocabulary
print("Size of GloVe's vocab:", len(list(word_vectors.vocab)))

# Number of words out of the GloVe vocabulary (OOV)
out_vocab = []
for i in vocab:
    if i in word_vectors.vocab: continue
    out_vocab.append(i)
print('Number of OOV:', round(len(out_vocab) / len(vocab), 2))


# Create a Dictionary of labels with its cosine values
def dic_labels(x, y):
    dic = {
        'neutral': [],
        'entailment': [],
        'contradiction': []
    }

    for i, j in zip(x, y):
        if j == 'entailment':
            dic[j].append(i)
        elif j == 'neutral':
            dic[j].append(i)
        else:
            dic[j].append(i)

    return dic


# Compute the embedding of a sentence by computing the average embedding of contained words
def vectorize_sent(word_vectors, sent):
    word_vecs = []
    for token in sent.split():
        if token not in word_vectors:
            continue
        else:
            word_vecs.append(word_vectors[token])

    return np.mean(np.array(word_vecs), axis=0)


# Vectorize the sentences
X_train1 = np.array([vectorize_sent(word_vectors, sent) for sent in train_data['clean_sent1'].values])
X_train2 = np.array([vectorize_sent(word_vectors, sent) for sent in train_data['clean_sent2'].values])

X_test1 = np.array([vectorize_sent(word_vectors, sent) for sent in test_data['clean_sent1'].values])
X_test2 = np.array([vectorize_sent(word_vectors, sent) for sent in test_data['clean_sent2'].values])

# Compute distance metrics
cosine = lambda x, y: cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).flatten()[0]
euclidean = lambda x, y: euclidean_distances(x.reshape(1, -1), y.reshape(1, -1)).flatten()[0]

train_data['dist_cos'] = list(map(cosine, X_train1, X_train2))
train_data['dist_euc'] = list(map(euclidean, X_train1, X_train2))

dist_cos = dic_labels(train_data['dist_cos'].values, train_data['gold_label'].values)
dist_euc = dic_labels(train_data['dist_euc'].values, train_data['gold_label'].values)

train_data.iloc[:5]

# Cosine scores distribution by labels
metric = ['Cosine', 'Euclidean']
results = [dist_cos, dist_euc]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

for i in range(0, 2):
    sb.distplot(results[i]['neutral'], hist=False, ax=ax[i])
    sb.distplot(results[i]['entailment'], hist=False, color='green', ax=ax[i])
    sb.distplot(results[i]['contradiction'], hist=False, color='red', ax=ax[i])
    ax[i].set_title(metric[i] + ' Distribution by labels')

plt.show()

print(X_train1.shape, X_train2.shape, X_test1.shape, X_test2.shape)

# The approach used to represent the input is going to be subtraction (A sent_vector - B sent_vector)
X_train = []
for i in range(0, len(X_train1)):
    subs = X_train1[i] - X_train2[i]
    X_train.append(subs)

X_test = []
for i in range(0, len(X_test1)):
    subs = X_test1[i] - X_test2[i]
    X_test.append(subs)

y_train = np.array(train_data['int_label'])
y_test = np.array(test_data['int_label'])

print('X_train Shape:', len(X_train))
print('y_train Shape:', y_train.shape)


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()  # list of models' names
        self.grid_searches = {}  ## empty dictionary for the Grid.fit of each model

    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs, cv=3)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search  ## fitting output from gird_search
        print('Done.')

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)  # The results for every combination of param
            frame = frame.filter(regex='^(?!.*param_).*$')  # remove columns about GRID parameters
            frame['estimator'] = len(frame) * [name]  # add the name of the model for every combo
            frames.append(frame)
        df = pd.concat(frames)  # final dict of all the training that the grid model has done

        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        # Reorder the columns so estimator is the first one
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns
        df = df[columns]
        return df
