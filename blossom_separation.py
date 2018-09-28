'''
    input: networkx undirected graph with vertices capacity b, edge weight x
    output: blossom inequality that separates x
'''

import networkx as nx
from gomory_hu import gomory_hu_tree, min_odd_cut


def blossom_separation(Graph, b, x):
    '''
    1. add dummy slack vertex check
    2. add dummy edges with appropriate weights, x is a dict check
    3. lable all vertices, denote T the odd vertices set
    4. build gomory_hu_tree for the resulting graph with T as the terminal set
    5. check the |T|-1 edges of the cut-tree
    '''
    G = Graph.copy()
    G.add_node('v')
    nx.set_node_attributes(G, b, 'capacity')
    terminal_set = set()

    for i in G:
        if i!='v':
            G.add_edge(i, 'v')
            if b[i] % 2 == 1.0:
                terminal_set.add(i)

    if sum(b.values()) % 2 == 1.0:
        terminal_set.add('v')
    for e in G.edges('v'):
        i = e[0] if e[1] == 'v' else e[1]
        x[frozenset(e)] = b[i]
        for ee in Graph.edges(i):
            x[frozenset(e)] -= x[frozenset(ee)]

    nx.set_edge_attributes(G, x, 'weight')

    gh_tree = gomory_hu_tree(G, capacity='weight')

    moc, W = min_odd_cut(gh_tree, terminal_set)
    if moc < 1:
        return W
    else:
        return None

# if __name__ == '__main__':
#     G = nx.Graph()
#     G.add_node(1)
#     G.add_node(2)
#     G.add_node(3)
#     G.add_edge(1,2)
#     G.add_edge(1,3)
#     G.add_edge(2,3)
#
#     b = {1:1, 2:1, 3:1}
#     x = {}
#     for e in G.edges:
#         x[frozenset(e)] = 0
#     e = list(G.edges)[0]
#     x[frozenset(e)] = 1
#     print x
#
#     print blossom_separation(G, b, x)
