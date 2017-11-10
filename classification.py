import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

'''
def load_labels(filename):
    lables = []
    f = open(filename, 'rU')
    for l in f.readlines():
        if l.replace('\n', '').split('\t')[-1]=='': continue
        lable = l.split('\t')[0]
        if lable == 'acq':
            lables.append(0)
        elif lable == 'crude':
            lables.append(1)
        elif lable == 'earn':
            lables.append(2)
        elif lable == 'grain':
            lables.append(3)
        elif lable == 'interest':
            lables.append(4)
        elif lable == 'money-fx':
            lables.append(5)
        elif lable == 'ship':
            lables.append(6)
        elif lable == 'trade':
            lables.append(7)
        else: print '***********ERROR****************'
    return lables


'''
def load_labels(filename):
    lables = []
    f = open(filename, 'rU')
    for l in f.readlines():
        if l.replace('\n', '').split('\t')[-1]=='': continue
        lable = l.split('\t')[0]
        if lable=='student': lables.append(0)
        elif lable=='course': lables.append(1)
        elif lable=='faculty': lables.append(2)
        elif lable=='project': lables.append(3)
        else: print '***********ERROR****************'
    return lables


split_str = 'v 0 '
def get_vocabulary(filename):
    f = open(filename, 'rU').readlines()
    vocabulary = []
    m = 1
    for n, l in enumerate(f):
        #if split_str in l:
        if l.startswith(split_str):
            if (n-2)<0: continue
            vocab_string = ''.join(f[m:n-2])
            m = n
            vocabulary.append(vocab_string)
    vocab_string=''.join(f[m:])
    vocabulary.append(vocab_string)
    print 'the length of the vocabulary is: ', len(vocabulary)
    return vocabulary

graph_str = 't # '
def load_data_vectors(file, vocabulary):
    all_vects = []
    f = open(file, 'rU').readlines()
    m=0
    mmm=0
    for n, l in enumerate(f):
        vectors = [0] * len(vocabulary)
        #if graph_str in l:
        if l.startswith(graph_string):
            if n-1<m+1: continue
            graph_string = ''.join(f[m+1:n-1])
            #print f[m]
            m = n
            for ind, vocab in enumerate(vocabulary):
                mm = 0
                ######## bug here ###################################
                for nn, string in enumerate(vocab.split('\n')):
                    mmm=nn
                    if string.split(' ')[-1] not in graph_string: continue
                    else: mm+=1
                    #if string.split(' ')[-1] in graph_string: mm+=1
                if mm!=0 and mm==mmm:
                    vectors[ind]+= 1
            all_vects.append(vectors)
    #print all_vects[260]
    return np.array(all_vects)

'''
graph_str = 't # '
def get_vocabulary_length(filename):
    f = open(filename, 'rU').readlines()
    vocab_len = []
    for n, l in enumerate(f):
        if graph_str in l:
            vocab_len.append(int(l.split(graph_str)[-1]))
    length = max(vocab_len)
    print 'vocabulary length: ', length
    return length

where_str = 'where:'
def load_data_vectors(file, vocab_len):
    all_vects = []
    f = open(file, 'rU').readlines()
    for n, l in enumerate(f):
        if where_str in l:
            vectors = [0] * vocab_len
            l= l[8:-2]
            index = l.split(', ')
            for ind in index: vectors[int(ind)]+= 1
            all_vects.append(vectors)
    return np.array(all_vects)
'''


def main():


    train_filename = '../data/webkb/webkb-train-stemmed.txt'
    test_filename = '../data/webkb/webkb-test-stemmed.txt'
    vocab_file = '../../graph_of_words/WebKB/graph_train.subgraph.s20'
    train_data_file = '../../graph_of_words/WebKB/graph.train.data'
    test_data_file = '../../graph_of_words/WebKB/graph.test.data'
    '''

    train_filename = '../data/R8/r8-train-no-stop.txt'
    test_filename = '../data/R8/r8-test-no-stop.txt'
    vocab_file = '../../graph_of_words/R8/graph_train.subgraph.s100'
    train_data_file = '../../graph_of_words/R8/graph.train.data'
    test_data_file = '../../graph_of_words/R8/graph.test.data'
    '''

    train_labels = load_labels(train_filename)
    test_labels = load_labels(test_filename)
    vocab = get_vocabulary(vocab_file)
    print 'train and test labels shapes', len(train_labels), len(test_labels)
    print 'train set counter', Counter(train_labels)
    print 'test set counter', Counter(test_labels)

    train_feature_vectors = load_data_vectors(train_data_file, vocab)
    test_feature_vectors = load_data_vectors(test_data_file, vocab)
    print 'train and test set shapes', train_feature_vectors.shape, test_feature_vectors.shape

    classifier = LinearSVC(C=0.5, class_weight='balanced')
    classifier.fit(train_feature_vectors, train_labels)
    print 'i have trained my classifier to perform sentiment analysis'

    predicted_labels = classifier.predict(test_feature_vectors)
    acc = accuracy_score(test_labels, predicted_labels)
    print 'i have a test set accuracy of: ', acc
    print classification_report(test_labels, predicted_labels)


if __name__ == '__main__':
    main()