import collections
import numpy as np

Node = collections.namedtuple('Node', ['id', 'name'])

def Get_Adjmatrix_from_celldag(cell_dag):
    Num_nodes = len(cell_dag)
    Adj = np.zeros((Num_nodes, Num_nodes))
    for i, dag_node in enumerate(cell_dag):
        # why dag_node is just a int value not a contruct?
        Adj[0:i, i] = cell_dag[dag_node].adj_node
        Adj[i, 0:i] = cell_dag[dag_node].adj_node
    return Adj