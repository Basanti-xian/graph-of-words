import string, sys, os
import networkx as nx
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

reload(sys)
sys.setdefaultencoding('utf8')

'''
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    a = TreebankWordTokenizer()
    tokens = a.tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
def stopwords_tokens(text):
    validwords = []
    for i in tokenize(text):
      if i not in stop:
          validwords.append(i)
    return validwords
'''

def rolling_window(seq, window_size):
    it = iter(seq)
    win = [it.next() for cnt in xrange(window_size)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win


if __name__=="__main__":

    #train_filename = '../data/webkb/webkb-train-stemmed.txt'
    train_filename = '../data/R8/r8-train-no-stop.txt'
    f = open(train_filename,'rU')
    train_sentences = f.readlines()
    ind = 0
    for sen in train_sentences:
        #sen = sen.replace('\n', '').split('\t')[-1].split(' ')
        sen = sen.replace('\n', '').split('\t')[-1].split(' ')[:-1]
        if sen==['']:
            print '***************skip*****************'
            continue
        G = nx.Graph()
        G.add_nodes_from(sen)

        for w in rolling_window(sen, 4):
            G.add_edges_from([(w[0], w[1]), (w[0], w[2]),(w[0], w[3])])

        #nx.draw(G)
        #op_fname ="../../graph_of_words/WebKB/graph_of_words_{}.train.gexf".format(ind)
        op_fname = "../../graph_of_words/R8/graph_of_words_{}.train.gexf".format(ind)
        print op_fname
        #nx.write_graphml (G,op_fname)
        nx.write_gexf(G,op_fname)
        ind +=1
        del G
    raw_input()
    test_filename = '../data/R8/r8-test-no-stop.txt'
    f = open(test_filename,'rU')
    test_sentences = f.readlines()
    ind= 0
    for sen in test_sentences:
        sen = sen.replace('\n', '').split('\t')[-1].split(' ')[:-1]
        if sen==['']:
            print '***************skip*****************'
            continue
        G = nx.Graph()
        G.add_nodes_from(sen)
        for w in rolling_window(sen, 4):
            G.add_edges_from([(w[0], w[1]), (w[0], w[2]),(w[0], w[3])])
        #op_fname ="../../graph_of_words/WebKB/graph_of_words_{}.test.gexf".format(ind)
        op_fname = "../../graph_of_words/R8/graph_of_words_{}.test.gexf".format(ind)
        print op_fname
        nx.write_gexf (G,op_fname)
        ind+= 1
        del G
