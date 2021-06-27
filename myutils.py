import contractions
import gensim.downloader as api
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import fasttext
import fasttext.util
import gensim
import pandas as pd

tk = RegexpTokenizer(r'\w+')
schemas = ["vulnerable", "angry", "impulsive", "happy", "detached", "punishing", "healthy"]
num_of_schemas = 7
max_words = 2000
max_epochs = 30
vec_size = 500

def read_corpus(input_list):
    i = 0
    result = []
    for sentence in input_list:
        result.append(gensim.models.doc2vec.TaggedDocument(sentence, [i]))
        i += 1
    return result


def tokenizer(texts: list) -> list:
    tokenized_texts = []
    for i in range(len(texts)):
        words = tk.tokenize(texts[i])
        tokenized_texts.append(words)
    return tokenized_texts


def remove_stopwords(s: str) -> str:
    new_str = ""
    for word in s.split():
        if word not in stopwords.words('english'):
            new_str += word + " "
    return new_str


def split_data(input_x, label_y, percent: float) -> (list, list, list, list):
    # Before
    # num_of_train = math.floor(len(data) * percent)
    # train_data = []
    # test_data = []
    # for i in range(0, num_of_train):
    #     train_data.append(data[i])
    #
    # for i in range(num_of_train, len(data)):
    #     test_data.append(data[i])

    # After
    x_train, x_test, y_train, y_test = train_test_split(input_x, label_y, test_size=percent, random_state=42)
    # print("y_train: ", np.sum(y_train, axis=0))
    # print("y_test: ", np.sum(y_test, axis=0))
    # x_train, y_train, x_test, y_test = iterative_train_test_split(input_x, label_y, test_size=percent)
    # print("y_train: ", np.sum(y_train, axis=0))
    # print("y_test: ", np.sum(y_test, axis=0))
    return x_train, y_train, x_test, y_test, percent


# Return list of tokenized strings through pre-processing(lowercase, noise removal, stop-word removal)
def pre_process_data(texts: list) -> (list, list):
    # Convert all to lowercase
    processed_texts = list(map(lambda s: s.lower(), texts))

    # Noise removal
    processed_texts = list(map(lambda s: contractions.fix(s), processed_texts))

    # TODO: Spelling correction
    # TODO: remove numbers
    # TODO: remove punctuation

    # Stop word-removal
    processed_texts = list(map(lambda s: remove_stopwords(s), processed_texts))

    # Tokenizer of strings
    tokenized_texts = tokenizer(processed_texts)

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokenized_texts = list(map(lambda s: (list(map(lambda y: lemmatizer.lemmatize(y), s))), tokenized_texts))
    processed_texts = list(map(lambda s: ' '.join(list(map(lambda y: lemmatizer.lemmatize(y), s))), tokenized_texts))
    # print("no lemmatization")
    return processed_texts, tokenized_texts


def get_average_for_each_label(dataframe):
    rows, cols = (dataframe.shape[0], dataframe.shape[1])
    text_list = []

    texts = dataframe['Text']
    for txt in texts:
        text_list.append(txt)

    average_label_list = np.zeros((rows, len(schemas)))
    for i in range(dataframe.shape[0]):
        j = 0
        average_label_list[i][j] = avg_helper(dataframe,i, 5, 15)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 16, 26)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 27, 35)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 36, 46)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 47, 56)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 57, 67)
        j += 1
        average_label_list[i][j] = avg_helper(dataframe, i, 68, 78)

    return text_list, average_label_list


def avg_helper(dataframe, i, begin, end):
    mean = dataframe.iloc[i, begin:end].mean()
    for j in dataframe.iloc[i, begin:end]:
        if (j is 5 or j is 6) and mean < 3.5:
            mean = 3.5
    return get_label(mean)


def get_label(mean) -> int:
    mean = round(mean)
    if mean <= 3:
        return 0
    elif 3 < mean <= 4:
        return 1
    elif 4 < mean <= 5:
        return 2
    elif 5 < mean <= 6:
        return 3
    else:
        return 0


def get_range_of_label(dataframe):
    rows, cols = (dataframe.shape[0], dataframe.shape[1])
    text_list = []

    texts = dataframe['Text']
    index_list = []
    for txt in texts:
        text_list.append(txt)
    
    range_of_label = np.zeros((rows, 67))
    for i in range(dataframe.shape[0]):
        range_of_label[i][:10] = dataframe.iloc[i, 5: 15]
        range_of_label[i][10:20] = dataframe.iloc[i, 16:26]
        range_of_label[i][20:28] = dataframe.iloc[i, 27:35]
        range_of_label[i][28:38] = dataframe.iloc[i, 36:46]
        range_of_label[i][38:47] = dataframe.iloc[i, 47:56]
        range_of_label[i][47:57] = dataframe.iloc[i, 57:67]
        # print("dataframe.iloc[i, 68:78]", dataframe.iloc[i].shape)
        range_of_label[i][57:67] = dataframe.iloc[i, 68:78]
        

    return text_list, range_of_label 

def rank_the_labels_without_avg(multilabels):
    boolean_list = []
    for i in range(len(multilabels)):
        label_bool_arr = []
        each_label_set = multilabels[i]

        vulnerable = each_label_set[:10]
        more_than_5_vulnerable = np.argwhere(vulnerable >= 5)
        avg_vulnerable = np.mean(vulnerable)

        if len(more_than_5_vulnerable) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        angry = each_label_set[10:20]
        more_than_5_angry = np.argwhere(angry >= 5)
        avg_angry = np.mean(angry)

        if len(more_than_5_angry) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        impulsive = each_label_set[20:28]
        more_than_5_impulsive = np.argwhere(impulsive >= 5)
        avg_impulsive = np.mean(impulsive)

        if len(more_than_5_impulsive) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        happy = each_label_set[28:38]
        more_than_5_happy = np.argwhere(happy >= 5)
        avg_happy = np.mean(happy)

        if len(more_than_5_happy) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        detached = each_label_set[38:47]
        more_than_5_detached = np.argwhere(impulsive >= 5)
        avg_detached = np.mean(detached)

        if len(more_than_5_detached) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        punishing = each_label_set[47:57]
        more_than_5_punishing = np.argwhere(punishing >= 5)
        avg_punishing = np.mean(punishing)

        if len(more_than_5_punishing) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        healthy = each_label_set[57:67]
        more_than_5_healthy = np.argwhere(healthy >= 5)
        avg_healthy = np.mean(healthy)

        if len(more_than_5_healthy) > 0:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        boolean_list.append(label_bool_arr)
    return boolean_list   

def rank_the_labels(multilabels):
    boolean_list = []
    for i in range(len(multilabels)):
        label_bool_arr = []
        each_label_set = multilabels[i]

        vulnerable = each_label_set[:10]
        more_than_5_vulnerable = np.argwhere(vulnerable >= 5)
        avg_vulnerable = np.mean(vulnerable)

        if len(more_than_5_vulnerable) > 0 or avg_vulnerable >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        angry = each_label_set[10:20]
        more_than_5_angry = np.argwhere(angry >= 5)
        avg_angry = np.mean(angry)

        if len(more_than_5_angry) > 0 or avg_angry >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        impulsive = each_label_set[20:28]
        more_than_5_impulsive = np.argwhere(impulsive >= 5)
        avg_impulsive = np.mean(impulsive)

        if len(more_than_5_impulsive) > 0 or avg_impulsive >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        happy = each_label_set[28:38]
        more_than_5_happy = np.argwhere(happy >= 5)
        avg_happy = np.mean(happy)

        if len(more_than_5_happy) > 0 or avg_happy >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        detached = each_label_set[38:47]
        more_than_5_detached = np.argwhere(impulsive >= 5)
        avg_detached = np.mean(detached)

        if len(more_than_5_detached) > 0 or avg_detached >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        punishing = each_label_set[47:57]
        more_than_5_punishing = np.argwhere(punishing >= 5)
        avg_punishing = np.mean(punishing)

        if len(more_than_5_punishing) > 0 or avg_punishing >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        healthy = each_label_set[57:67]
        more_than_5_healthy = np.argwhere(healthy >= 5)
        avg_healthy = np.mean(healthy)

        if len(more_than_5_healthy) > 0 or avg_healthy >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        boolean_list.append(label_bool_arr)
    return boolean_list

def rank_the_labels_without_5(multilabels):
    boolean_list = []
    for i in range(len(multilabels)):
        label_bool_arr = []
        each_label_set = multilabels[i]

        vulnerable = each_label_set[:10]
        more_than_5_vulnerable = np.argwhere(vulnerable >= 5)
        avg_vulnerable = np.mean(vulnerable)

        if avg_vulnerable >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        angry = each_label_set[10:20]
        more_than_5_angry = np.argwhere(angry >= 5)
        avg_angry = np.mean(angry)

        if avg_angry >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        impulsive = each_label_set[20:28]
        more_than_5_impulsive = np.argwhere(impulsive >= 5)
        avg_impulsive = np.mean(impulsive)

        if avg_impulsive >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        happy = each_label_set[28:38]
        more_than_5_happy = np.argwhere(happy >= 5)
        avg_happy = np.mean(happy)

        if avg_happy >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        detached = each_label_set[38:47]
        more_than_5_detached = np.argwhere(impulsive >= 5)
        avg_detached = np.mean(detached)

        if avg_detached >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        punishing = each_label_set[47:57]
        more_than_5_punishing = np.argwhere(punishing >= 5)
        avg_punishing = np.mean(punishing)

        if avg_punishing >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        healthy = each_label_set[57:67]
        more_than_5_healthy = np.argwhere(healthy >= 5)
        avg_healthy = np.mean(healthy)

        if avg_healthy >= 3.5:
            label_bool_arr.append(1)
        else:
            label_bool_arr.append(0)

        boolean_list.append(label_bool_arr)
    return boolean_list

def get_average_for_each_label_index(dataframe):
    rows, cols = (dataframe.shape[0], dataframe.shape[1])
    text_list = []

    texts = dataframe['Text']
    index_list = []
    for txt in texts:
        text_list.append(txt)
    
    average_label_list = np.zeros((rows, len(schemas)))
    for i in range(dataframe.shape[0]):
        index_list.append(i)
        j = 0
        average_label_list[i][j] = dataframe.iloc[i, 5: 15].mean()
        j += 1
        average_label_list[i][j] = dataframe.iloc[i, 16:26].mean()
        j += 1
        average_label_list[i][j] = dataframe.iloc[i, 27:35].mean()
        j += 1
        average_label_list[i][j] = dataframe.iloc[i, 36:46].mean()
        j += 1
        average_label_list[i][j] = dataframe.iloc[i, 47:56].mean()
        j += 1
        average_label_list[i][j] = dataframe.iloc[i, 57:67].mean()
        j += 1
        average_label_list[i][j] = dataframe.iloc[i, 68:78].mean()

    return text_list, index_list, average_label_list

# takes in dataframe, returns list of 'Texts' and list of 'Labels'
def get_text_labels(dataframe):
    rows, cols = (dataframe.shape[0], dataframe.shape[1])

    text_list = []
    label_list = np.zeros((rows, len(schemas)))

    texts = dataframe['Text']
    for txt in texts:
        text_list.append(txt)

    is_vulnerable = dataframe['is_vulnerable']
    is_angry = dataframe['is_angry']
    is_impulsive = dataframe['is_impulsive']
    is_happy = dataframe['is_happy']
    is_detached = dataframe['is_detached']
    is_punishing = dataframe['is_punishing']
    is_healthy = dataframe['is_healthy']

    for i in range(dataframe.shape[0]):
        j = 0
        label_list[i][j] = 1 if bool(is_vulnerable[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_angry[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_impulsive[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_happy[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_detached[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_punishing[i]) == True else 0
        j += 1
        label_list[i][j] = 1 if bool(is_healthy[i]) == True else 0

    return text_list, label_list

# TODO: train models
def training_model_fast_text():
    # If model is obtained, no need to run this part of code\
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    return ft


# returns pre-trained word2vec model
def get_word2vec():
    print('LOAD GOOGLE WORD VECTORS')

    return api.load("word2vec-google-news-300")


def training_model_d2v(texts):
    tagged_docs = read_corpus(texts)
    print("TRAINING MODEL")

    model = gensim.models.Doc2Vec(documents=tagged_docs, vector_size=vec_size, window=10, epochs=max_epochs, min_count=1, workers=4, alpha=0.025, min_alpha=0.025)
    model.save("../models/schema-d2v-knn.model")
    return model

"""
Code to parse sklearn classification_report
Original: https://gist.github.com/julienr/6b9b9a03bd8224db7b4f
Modified to work with Python 3 and classification report averages
"""

import sys
import collections

def parse_classification_report(clfreport):
    """
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class
    """
    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[1] == 'avg'

    # We cannot simply use split because class names can have spaces. So instead
    # figure the width of the class field by looking at the indentation of the
    # precision header
    cls_field_width = len(header) - len(header.lstrip())
    # Now, collect all the class names and score in a dict
    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    # average
    data['avg'] = parse_line(avg_line)[1:]

    return data

def report_to_latex_table(data):
    avg_split = False
    out = ""
    out += "\\begin{table}\n"
    out += "\\caption{Latex Table from Classification Report}\n"
    out += "\\label{table:classification:report}\n"
    out += "\\centering\n"
    out += "\\begin{tabular}{c | c c c r}\n"
    out += "Class & Precision & Recall & F-score & Support\\\\\n"
    out += "\midrule\n"
    for cls, scores in data.items():
        if 'micro' in cls:
            out += "\\midrule\n"
        out += cls + " & " + " & ".join([str(s) for s in scores])
        out += "\\\\\n"
    out += "\\end{tabular}\n"
    out += "\\end{table}"
    return out
