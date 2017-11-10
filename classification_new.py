import sys,os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing,svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def sentiment_labels(filename):
    s_labels = [int(l.split('\t')[-1]) for l in open(filename)]
    return s_labels

def new_line_tokenizer(s):
    toks = [l.strip() for l in s.split('\n')]
    return toks

def load_data_vectors(files_to_process):
    sorted_files = [None for _ in files_to_process]
    for f in files_to_process:
        f_index = int(f.split('.gexf')[0].split('_')[-1])
        sorted_files[f_index] = f
    vectorizer = CountVectorizer(input='filename', tokenizer=new_line_tokenizer, lowercase=False)
    vectors = vectorizer.fit_transform(sorted_files)
    print vectors.shape
    raw_input()
    return vectors

def get_files_to_process(dirname, extn):
    files_to_process = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                files_to_process.append(os.path.join(root, f))

    files_to_process = list(set(files_to_process))
    return files_to_process


def main():

    filename = 'imdb_labelled.txt'
    WL_dir = sys.argv[1]
    extn = '.WL2'
    WL_files_to_process = get_files_to_process(WL_dir, extn)

    sentiment = sentiment_labels(filename)
    vectors = load_data_vectors(WL_files_to_process)
    vectors_normalized = preprocessing.normalize(vectors, norm = 'l2')

    train_feature_vectors, test_feature_vectors, train_sentiment, test_sentiment = train_test_split(
        vectors_normalized, sentiment, test_size = 0.2, random_state = 42)


    print 'train and test set shapes', train_feature_vectors.shape, test_feature_vectors.shape

    # classifier = LinearSVC(C=2.5, class_weight='balanced')

    # parameters = {'kernel': ['linear','rbf'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'class_weight':['balanced', None]}
    # svr = svm.SVC()

    parameters = {'C': [2, 2.1, 2.2, 2.3, 2.4, 2.5, 3], 'class_weight':['balanced', None]}
    svr= LinearSVC()

    classifier = GridSearchCV(svr, parameters)
    classifier.fit(train_feature_vectors, train_sentiment)
    print 'i have trained my classifier to perform sentiment analysis'
    best_parameters = classifier.best_params_
    print "best parameter is :", best_parameters

    predicted_sentiment = classifier.predict(test_feature_vectors)
    acc = accuracy_score(test_sentiment, predicted_sentiment)
    print 'i have a test set accuracy of: ', acc


if __name__ == '__main__':
    main()
