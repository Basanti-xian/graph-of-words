import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from collections import Counter
'''
def load_train_data(filename):
    lables = []
    data = []
    f = open(filename, 'rU')
    for l in f.readlines():
        sen = l.replace('\n', '').split('\t')[-1]
        print l
        print sen
        raw_input()
        if sen == '': continue
        data.append(sen)
        lable = l.split('\t')[0]
        if lable=='acq': lables.append(0)
        elif lable=='crude': lables.append(1)
        elif lable=='earn': lables.append(2)
        elif lable=='grain': lables.append(3)
        elif lable=='interest': lables.append(4)
        elif lable=='money-fx': lables.append(5)
        elif lable=='ship': lables.append(6)
        elif lable == 'trade':lables.append(7)
        else: print '***********ERROR****************'
    vectorizer = CountVectorizer(input=u'content', ngram_range=(1, 1), lowercase=True)
    vectors = vectorizer.fit_transform(data)
    return lables, vectors, vectorizer

def load_test_data(filename, vectorizer):
    lables = []
    data = []
    f = open(filename, 'rU')
    for l in f.readlines():
        sen = l.replace('\n', '').split('\t')[-1]
        if sen == '': continue
        data.append(sen)
        lable = l.split('\t')[0]
        if lable=='acq': lables.append(0)
        elif lable=='crude': lables.append(1)
        elif lable=='earn': lables.append(2)
        elif lable=='grain': lables.append(3)
        elif lable=='interest': lables.append(4)
        elif lable=='money-fx': lables.append(5)
        elif lable=='ship': lables.append(6)
        elif lable == 'trade':lables.append(7)
        else: print '***********ERROR****************'
    vectors = vectorizer.transform(data)
    return lables, vectors


'''
def load_train_data(filename):
    lables = []
    data = []
    f = open(filename, 'rU')
    for l in f.readlines():
        sen = l.replace('\n', '').split('\t')[-1]
        if sen == '': continue
        data.append(sen)
        lable = l.split('\t')[0]
        if lable=='student': lables.append(0)
        elif lable=='course': lables.append(1)
        elif lable=='faculty': lables.append(2)
        elif lable=='project': lables.append(3)
        else: print '***********ERROR****************'
    vectorizer = CountVectorizer(input=u'content', ngram_range=(1, 1), lowercase=True)
    vectors = vectorizer.fit_transform(data)
    return lables, vectors, vectorizer

def load_test_data(filename, vectorizer):
    lables = []
    data = []
    f = open(filename, 'rU')
    for l in f.readlines():
        sen = l.replace('\n', '').split('\t')[-1]
        if sen == '': continue
        data.append(sen)
        lable = l.split('\t')[0]
        if lable=='student': lables.append(0)
        elif lable=='course': lables.append(1)
        elif lable=='faculty': lables.append(2)
        elif lable=='project': lables.append(3)
        else: print '***********ERROR****************'
    vectors = vectorizer.transform(data)
    return lables, vectors


def main():

    train_filename = '../data/webkb/webkb-train-stemmed.txt'
    train_lables, train_vectors, vectorizer = load_train_data(train_filename)
    train_feature_vectors = preprocessing.normalize(train_vectors, norm = 'l2')

    test_filename = '../data/webkb/webkb-test-stemmed.txt'
    test_lables, train_vectors = load_test_data(test_filename, vectorizer)
    test_feature_vectors = preprocessing.normalize(train_vectors, norm = 'l2')

    print 'train and test set shapes', train_feature_vectors.shape, test_feature_vectors.shape
    print 'train set counter', Counter(train_lables)
    print 'test set counter', Counter(test_lables)

    #classifier = LinearSVC(C=2.5, class_weight='balanced')
    #classifier = LinearSVC()

    parameters = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'class_weight':['balanced']}
    svr = svm.SVC()

    # parameters = {'C': [1, 1.5, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 3], 'class_weight':['balanced', None]}
    # svr= LinearSVC()

    classifier = GridSearchCV(svr, parameters)

    classifier.fit(train_feature_vectors, train_lables)
    print 'i have trained my classifier to perform sentiment analysis'

    #best_parameters = classifier.best_params_
    #print "best parameter is :", best_parameters

    predicted_lables = classifier.predict(test_feature_vectors)
    print 'prediction counter', Counter(predicted_lables)
    acc = accuracy_score(test_lables, predicted_lables)
    print 'i have a test set accuracy of: ', acc
    print classification_report(test_lables, predicted_lables)


if __name__ == '__main__':
    main()