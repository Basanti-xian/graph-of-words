import networkx as nx
import os,sys

reload(sys)
sys.setdefaultencoding('utf8')


def read_graph(fname=None):
    if not fname:
        print 'no valid path or file name'
        return None
    else:
        try:
            g = nx.read_gexf (path=fname)
        except:
            print "unable to load graph from file", fname
            return 0
    print 'loaded {} a graph with {} nodes and {} egdes'.format(fname,g.number_of_nodes(),g.number_of_edges())
    return g




#filename = '../../graph_of_words/R8/graph.train.data'
filename = '../../graph_of_words/R8/graph.test.data'
f = open(filename, 'w')
def preprocess_graph(num, g):
    nodes = []
    f.write('t # {}\n'.format(num))
    for n, v in enumerate(g.nodes_iter()):
        value = 'v {} {}\n'.format(n, v)
        nodes.append(v)
        f.write(str(value))
    for e in g.edges_iter():
        e = sorted(e)
        value = 'e {} {} {}\n'.format(nodes.index(e[0]), nodes.index(e[1]), str('{}--{}'.format(e[0], e[1])))
        f.write(str(value))



def get_files_to_process(dirname, extn):
    files_to_process = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                files_to_process.append(os.path.join(root, f))

    files_to_process = list(set(files_to_process))
    files_to_process = sorted(files_to_process, key=lambda x: int(x.split('_')[-1].split(extn)[0]))
    return files_to_process




def main():

    gexf_dir = '../../graph_of_words/R8/'
    train_extn = '.train.gexf'
    train_gexf_files_to_process = get_files_to_process(gexf_dir, train_extn)
    test_extn = '.test.gexf'
    test_gexf_files_to_process = get_files_to_process(gexf_dir, test_extn)

    #for num, graph in enumerate(train_gexf_files_to_process):
    #    preprocess_graph(num, read_graph(graph))

    for num, graph in enumerate(test_gexf_files_to_process):
        preprocess_graph(num, read_graph(graph))

    f.write('t # -1\n')


if __name__ == '__main__':
    main()