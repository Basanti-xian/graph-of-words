import networkx as nx
from graph_tool.topology import subgraph_isomorphism


split_str = 'v 0 '
def get_vocabulary(filename):
    f = open(filename, 'rU').readlines()
    vocabulary = []
    m = 0
    for n, l in enumerate(f):
        if split_str in l:
            vocab_string = ''.join(f[m:n-1])
            m = n
            vocabulary.append(vocab_string)
    print 'the length of the vocabulary is: ', len(vocabulary)
    return vocabulary[1:]


def subgraph_process(vocabulary):
    for ind, vocab_string in enumerate(vocabulary):
        G = nx.Graph()
        vocab_string = vocab_string.split('\n')
        for l in vocab_string[:-1]:
            if l[0]=='v':
                G.add_node(l.split(' ')[-1])
            elif l[0]=='e':
                w = l.split(' ')[-1].split('--')
                G.add_edge(w[0], w[1])

        op_fname ="vocab_subgraph_{}.gexf".format(ind)
        nx.write_gexf (G,op_fname)
        del G


def main():

    # vocab_file = 'vocabulary.subgraph'
    # subgraph_process(get_vocabulary(vocab_file))

    print subgraph_isomorphism(vocab_subgraph_0.gexf, graph_of_words_0.gexf)




if __name__ == '__main__':
    main()