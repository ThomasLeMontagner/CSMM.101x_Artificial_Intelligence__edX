import os
import csv
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

# paths
local_train_path = "./aclImdb/train/"
local_test_path = "imdb_te.csv"
train_path = "../resource/lib/publicdata/aclImdb/train/"  # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv"  # test data for grade evaluation


# return list of English stop words
def getstopwords():
    stopwords = []
    swfile = open("stopwords.en.txt", "r")
    for word in swfile.readlines():
        stopwords.append(word.strip())
    return stopwords


# Words list
stopwords = getstopwords()
train_words = set()


# Process Imdb comments by generating in *.csv file with columns row_number, text and polarity
def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
   and combine text files under train_path directory into
   imdb_tr.csv. Each text file in train_path should be stored
   as a row in imdb_tr.csv. And imdb_tr.csv should have two
   columns, "text" and label'''

    data = []
    read_pos_files(data, inpath)
    read_neg_files(data, inpath)

    if mix:
        np.random.shuffle(data)

    data.insert(0, ["row_number", "text", "polarity"])

    # create file
    with open(name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


# Read all the comment files, process them and add comment in a table (data)
def read_files(data, path, polarity):
    i = len(data)
    for filename in os.listdir(path):
        file = open(path + filename, "r")
        line = remove_stopwords(file.readline())
        #train_words.update(line.split(' '))
        data.append([i, line, polarity])
        #data.append([line, polarity])
        i += 1
        file.close()


# Read the negatives comments in a "neg" folder
def read_neg_files(data, path):
    read_files(data, path + "neg/", 0)


# Read the positives comments in a "pos" folder
def read_pos_files(data, path):
    read_files(data, path + "pos/", 1)


# Remove English stopwords from a line
def remove_stopwords(line):
    # lower case text
    newline = line.lower()
    # remove punctuations
    newline = re.sub("[^a-zA-Z0-9]", " ", newline)
    final_line = []

    for word in newline.split(' '):
        if not word in stopwords:
            final_line.append(word)
    return ' '.join(final_line)


# Create a unigram representation (table) of Imdb comments
# def create_ngram(name="imdb_tr.csv"):
#     tr_data = pd.read_csv(name, encoding = "ISO-8859-1")
#     print(tr_data.head())
#     unigram = set()
#     bigram = set()
#     for text in tr_data['text']:
#         w = word_tokenize(text)
#         unigram.update(word_tokenize(text))
#         bigram.update(bigrams(w))
#
#     generate_unigram_file(unigram)
#     generate_binagram_file(bigram)
#
#
# def generate_binagram_file(bigram):
#     new_binagram = [' '.join(l) for l in bigram]
#     output = pd.DataFrame(new_binagram)
#     output.to_csv('bigram.csv', index=False, header=['terms'])
#
#
# def generate_unigram_file(unigram):
#     output = pd.DataFrame(unigram)
#     output.to_csv('unigram.csv', index=False, header=['terms'])
#
#
# def create_unigram_rep(name="imdb_tr.csv"):
#     uni_terms = pd.read_csv('unigram.csv', encoding = "ISO-8859-1")
#     tr_texts = pd.read_csv(name, encoding = "ISO-8859-1")
#     print(uni_terms.head())
#     print(tr_texts.head())
#
#     #unigram_rep = pd.DataFrame(0, index=range(len(tr_texts.index)), columns=list(uni_terms['terms']))
#     unigram_rep=  pd.DataFrame(columns=uni_terms['terms'])
#     print(unigram_rep.head())

# Create a unigram model of the data
def unigram(data):
    vect = CountVectorizer()
    return vect.fit(data)


# Create a bigram model of the data
def bigram(data):
    vect = CountVectorizer(ngram_range=(1,2))
    return vect.fit(data)


# Transform model with tf-idf
def tfidf(data):
    trans = TfidfTransformer()
    return trans.fit(data)


# Return the training data (text and polarity) of a givzn csv file
def get_train_data(name="imdb_tr.csv",):
    data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
    return data['text'], data['polarity']


# Return the test data (text) of a given csv file
def get_test_data(name="imdb_tr.csv",):
    data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
    return data['text']


# Generate the prediction of a SGD for given train data (X_train and Y_train) and test data (X_test)
def stochastic_descent_prediction(X_train, Y_train, X_test):
    clf = SGDClassifier(loss="hinge", penalty = "l1")
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)


# Generate a file with given name filled out with the given data
def write_output(data, file_name):
    data = [str(d) for d in data]
    file = open(file_name, 'w')
    file.writelines('%s\n' % item for item in data)
    file.close()
    pass


# Vectorize the input data, run a stochastic gradient descent and generate result in output file
def analyze_data(transformer, Xtest_uni, Xtrain_uni, Y_train, file_name):
    print("Starting data ana!!lysis for " + file_name)
    X_train_vect = transformer.transform(Xtrain_uni)
    X_test_vect = transformer.transform(Xtest_uni)
    Y_test_v = stochastic_descent_prediction(X_train_vect, Y_train, X_test_vect)
    write_output(Y_test_v, file_name=file_name)
    print("File " + file_name + " completed.")


if __name__ == "__main__":
    print('Data processing...')
    #imdb_data_preprocess(local_train_path, mix=True)
    print('Data processing completed.')
    [X_train, Y_train] = get_train_data()
    X_test = get_test_data(name=local_test_path)

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    print("Unigram processing...")
    uni_vectorizer = unigram(X_train)
    analyze_data(uni_vectorizer, X_test, X_train, Y_train, "unigram.output.txt")

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
    print("Bigram processing...")
    bi_vectorizer = bigram(X_train)
    analyze_data(bi_vectorizer, X_test, X_train, Y_train, "bigram.output.txt")

    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigramtfidf.output.txt'''
    print("Unigram with tf-idf processing...")
    X_train_uni = uni_vectorizer.transform(X_train)
    uni_tfidf = tfidf(X_train_uni)
    X_test_uni = uni_vectorizer.transform(X_test)
    analyze_data(uni_tfidf, X_test_uni, X_train_uni, Y_train, "unigramtfidf.output.txt")

    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to bigramtfidf.output.txt'''
    print("Bigram with tf-idf processing...")
    X_train_bi = bi_vectorizer.transform(X_train)
    bi_tfidf = tfidf(X_train_bi)
    X_test_bi = bi_vectorizer.transform(X_test)
    analyze_data(bi_tfidf, X_test_bi, X_train_bi, Y_train, "bigramtfidf.output.txt")
    pass
