import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import build_residual_network
default_flow_func = edmonds_karp

def gomory_hu_tree(G, capacity='capacity', flow_func = None):
    '''
        Gusfield implementation
    '''
    if flow_func is None:
        flow_func = default_flow_func

    if len(G) == 0:  # empty graph
        msg = 'Empty Graph does not have a Gomory-Hu tree representation'
        raise nx.NetworkXError(msg)

    R = build_residual_network(G, capacity)
    p = {}
    fl = {}
    nodes = list(G.nodes)
    root = nodes[0]
    for i in nodes:
        p[i] = root

    for s in nodes[1:]:
        t = p[s]
        cut_value, partition = nx.minimum_cut(G, s, t, capacity=capacity, flow_func=flow_func, residual=R)
        X, Y = partition
        fl[s] = cut_value
        for i in nodes:
            if i != s and i in X and p[i] == t:
                p[i] = s
        if p[t] in X:
            p[s] = p[t]
            p[t] = s
            fl[s] = fl[t]
            fl[t] = cut_value
    T = nx.Graph()
    T.add_nodes_from(G)
    T.add_weighted_edges_from(((u, p[u], fl[u]) for u in nodes[1:]))
    return T

def min_odd_cut(T, V_odd):
    # print(nx.get_edge_attributes(T,'weight').values())
    moc = max(nx.get_edge_attributes(T,'weight').values())
    W = 0
    for e in T.edges():
        this_T = T.copy()
        this_T.remove_edge(*e)
        V1 = nx.node_connected_component(this_T,e[0])
        V2 = nx.node_connected_component(this_T,e[1])
        n_odd_node = 0
        for v in V_odd:
            n_odd_node += (1 if v in V1 else 0)
        if n_odd_node % 2 == 1:
            this_cut = T.get_edge_data(*e)
            if this_cut['weight'] < moc:
                moc = this_cut['weight']
                if 'v' in V1:
                    W = V2
                else:
                    W = V1
    return moc, W

# if __name__ == '__main__':
#     G = nx.Graph()
#     G.add_nodes_from([1,2,3,4,5,6])
#     G.add_edge(1,2, capacity=10)
#     G.add_edge(1,6, capacity=8)
#     G.add_edge(2,6, capacity=3)
#     G.add_edge(2,3, capacity=4)
#     G.add_edge(2,5, capacity=2)
#     G.add_edge(6,3, capacity=2)
#     G.add_edge(5,6, capacity=3)
#     G.add_edge(3,5, capacity=4)
#     G.add_edge(3,4, capacity=5)
#     G.add_edge(4,6, capacity=2)
#     G.add_edge(4,5, capacity=7)
#
#
#
#     T = gomory_hu_tree(G)
#     for e in T.edges:
#         print e
#         print G.get_edge_data(*e)
#
#
#
#     V_odd = [2,3,5,6]
#     print min_odd_cut(T, V_odd)
