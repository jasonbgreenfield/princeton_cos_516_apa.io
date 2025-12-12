"""
File: kleene_dag.py
Name: Jason Greenfield
Purpose: implement Kleene's algorithm using a DAG for efficiency
Note: based on algorithm 1 in lecture notes 16 for COS 516, taught by Professor Zachary Kincaid at Princeton University
Note: use python 3.12 for pygraphviz dependency for DAG visualization
"""

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


###
# DAG for kleene path expressions
###
INTERN_TABLE = {}

def intern(node):
    """
    interns kleene obj node based on its key
    this enforces the dag structure so only one version of each object is created
    """
    key = node.key()
    if key in INTERN_TABLE:
        return INTERN_TABLE[key]
    INTERN_TABLE[key] = node
    return node


###
# base class for all Kleene objects
###
class KleeneObject:
    def key(self):
        # returns unique key for this object
        raise NotImplementedError

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return isinstance(other, KleeneObject) and self.key() == other.key()

    def __str__(self):
        return self.to_string()

    def to_string(self):
        raise NotImplementedError

###
# graph structures and leaves
###
class EmptySet(KleeneObject):
    def key(self):
        return ("empty",)

    def to_string(self):
        return "∅"

class Epsilon(KleeneObject):
    def key(self):
        return ("epsilon",)

    def to_string(self):
        return "{ϵ}"

class Edge(KleeneObject):
    # represents <u,v> edge
    def __init__(self, u, v):
        self.u = u
        self.v = v

    def key(self):
        return ("edge", self.u, self.v)

    def to_string(self):
        return f"<{self.u},{self.v}>"


###
# init vars
###
EMPTY = intern(EmptySet())
EPSILON = intern(Epsilon())


###
# kleene function classes
###
class KleeneUnion(KleeneObject):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def key(self):
        return ("union", self.left.key(), self.right.key())

    def to_string(self):
        return f"({self.left.to_string()} + {self.right.to_string()})"

class KleeneConcat(KleeneObject):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def key(self):
        return ("concat", self.left.key(), self.right.key())

    def to_string(self):
        return f"({self.left.to_string()}{self.right.to_string()})"

class KleeneStar(KleeneObject):
    def __init__(self, pe):
        self.pe = pe

    def key(self):
        return ("star", self.pe.key())

    def to_string(self):
        return f"({self.pe.to_string()})*"




###
# Kleene functions
###
def kleene_union(u, v):
    """
    Union returns the union of both sets of paths through either u or v
    Various simplifications are implemented
    Default is to return KleeneUnion(u_pe, v_pe)
    """
    # simplification: ∅ + p = p
    if u is EMPTY:
        return v
    elif v is EMPTY:
        return u
    # simplification: p + p = p
    elif u is v:
        return u
    # simplification: p + p* = p*
    elif isinstance(v, KleeneStar) and v.pe is u:
        return v
    # simplification: p* + p = p*
    elif isinstance(u, KleeneStar) and u.pe is v:
        return u
    # NOTE: (pq) + p =/= pq. this does not simplify!
    # simplification: (pq*) + p = (q*p) + p = (pq*)
    elif isinstance(u, KleeneConcat) and ((isinstance(u.right, KleeneStar) and u.left is v) or
                                          (isinstance(u.left, KleeneStar) and u.right is v)):
        return u
    # simplification: p + (pq*) = p + (q*p) = (pq*)
    elif isinstance(v, KleeneConcat) and ((isinstance(v.right, KleeneStar) and v.left is u) or
                                        (isinstance(v.left, KleeneStar) and v.right is u)):
        return v
    # otherwise, return naive u+v
    return intern(KleeneUnion(u, v))


def kleene_concatenation(u, v):
    """
    Concatenation returns the sequence of u then v
    Various simplifications are implemented
    Default is to return KleeneConcat(u_pe, v_pe)
    """
    # simplification: # ∅p = p∅ = ∅
    if u is EMPTY or v is EMPTY:
        return EMPTY
    # simplification: εx = xε =  x
    elif u is EPSILON:
        return v
    elif v is EPSILON:
        return u
    # simplification: pp* = p*
    elif isinstance(v, KleeneStar) and v.pe is u:
        return v
    # simplification: p*p = p*
    elif isinstance(u, KleeneStar) and u.pe is v:
        return u
    # simplification: p*p* = p*
    # NOTE: we cannot simplify pp, but we can simplify pp*, p*p, and p*p* all to just p*
    elif isinstance(u, KleeneStar) and isinstance(v, KleeneStar) and u is v:
        return u
    # simplification: (qp*)p* = (p*q)p* = qp*
    elif ((isinstance(u, KleeneConcat) and isinstance(v, KleeneStar)) and
          ((isinstance(u.right, KleeneStar) and u.right is v) or
           (isinstance(u.left, KleeneStar) and u.left is v))):
            return u
    # simplification: p*(qp*) = p*(p*q) = qp*
    elif ((isinstance(v, KleeneConcat) and isinstance(u, KleeneStar)) and
          ((isinstance(v.left, KleeneStar) and v.left is u) or
           (isinstance(v.right, KleeneStar) and v.right is u))):
            return v
    # otherwise, return naive concatenation of uv
    return intern(KleeneConcat(u, v))


def kleene_closure(u):
    """
    Closure returns the sequence of u then v
    Simplifications:
        Case 1: consider the empty language ∅
            if u is ∅, return ∅ since you can't loop around nothing
        Case 2: consider the empty set / singleton word  {ϵ}
            same as above, if u is {ϵ}, return {ϵ} since you can't loop around nothing
        Case 3: p** = p*
    Otherwise, return KleeneStar(u_pe)
    """
    # simplification: ∅* = ε* = ε
    if u is EMPTY or u is EPSILON:
        return EPSILON
    # simplification: p** = p*
    if isinstance(u, KleeneStar):
        return u
    # otherwise, return naive u-star
    return intern(KleeneStar(u))


def run_kleene_algorithm(nodes, edges, start):
    # make sure start is in our list of nodes
    if start not in nodes:
        print(f'Invalid {start}! Must be in {nodes}')
        return None

    # make sure neither of our helper values are in the list of node names
    if '∅' in nodes or '{ϵ}' in nodes:
        print('Invalid node names, ∅ and {ϵ} cannot be nodes!')

    # start node must be 0th index in list of nodes
    nodes.sort()
    nodes.remove(start)
    nodes.insert(0, start)

    # init mapping from node name to num
    node_name_to_index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    # init matrix keeping track of all path expressions between nodes
    # starting vals are all the empty set represented as ∅
    # this is equivalent to: 0L ≜ ∅ is the empty language
    path_expressions = []
    for i in range(n):
        tmp_row = []
        for j in range(n):
            tmp_row.append(EMPTY)
        path_expressions.append(tmp_row)

    # vals along the diagonal represent paths from a node to itself
    # these are represented with {ϵ}, the singleton language containing the empty word
    # this is equivalent to: 1L ≜ {ϵ} is the singleton language containing the empty word
    for i in range(n):
        path_expressions[i][i] = EPSILON

    # Initialize direct edges
    for edge in edges:
        u, v = edge
        u_index, v_index = node_name_to_index[u], node_name_to_index[v]
        path_expressions[u_index][v_index] = intern(Edge(u, v))

    # kleene's path expression algo
    for i in range(n - 1, -1, -1): # for i = n downto 0 do
        for j in range(i, -1, -1): # for j = i downto 0 do
            for k in range(n - 1, -1, -1): # for k = n downto 0 do
                # compute:
                    # pe(vj,vk) ←pe(vj,vk) + pe(vj,vi)·pe(vi,vi)∗ ·pe(vi,vk)
                # init vars
                vj_vk = path_expressions[j][k]
                vj_vi = path_expressions[j][i]
                vi_vi = path_expressions[i][i]
                vi_vk = path_expressions[i][k]
                # apply functions
                star_ii = kleene_closure(vi_vi)
                concat_ji_iistar = kleene_concatenation(vj_vi, star_ii)
                concat_jiiistar_ik = kleene_concatenation(concat_ji_iistar, vi_vk)
                union_jk_jiiiistarik = kleene_union(vj_vk, concat_jiiistar_ik)
                # update val
                path_expressions[j][k] = union_jk_jiiiistarik

    # get path expressions from start to all possible nodes
    start_index = node_name_to_index[start]
    start_to_pe = {}
    for node in nodes:
        node_index = node_name_to_index[node]
        start_to_pe[node] = path_expressions[start_index][node_index]
    return start_to_pe


# apply final post-processing simplification rule: ({ϵ} + p)* = p*
# NOTE: I tried to implement this at the end of the KleeneStar() function for each call, but that actually
# exploded the size of the path expression, so I'm just doing this at the end now
def simplify_epsilon_star(obj):
    if obj is EPSILON or obj is EMPTY:
        return obj

    if isinstance(obj, KleeneUnion):
        left = simplify_epsilon_star(obj.left)
        right = simplify_epsilon_star(obj.right)
        return intern(KleeneUnion(left, right))

    if isinstance(obj, KleeneConcat):
        left = simplify_epsilon_star(obj.left)
        right = simplify_epsilon_star(obj.right)
        return intern(KleeneConcat(left, right))

    if isinstance(obj, KleeneStar):
        pe = simplify_epsilon_star(obj.pe)

        # ({ϵ} + p)* = (p + {ϵ})* = p*
        if isinstance(pe, KleeneUnion):
            if pe.left is EPSILON:
                return intern(KleeneStar(pe.right))
            if pe.right is EPSILON:
                return intern(KleeneStar(pe.left))

        return intern(KleeneStar(pe))

    return obj


###
# distance interpretation for kleene algebra
###
def get_distance_interpretation(node, weight_dict, memo=None):
    """
    computes the distance interpretation for the DAG
    :param node: root node for the DAG of KleeneObjects
    :param weight_dict: dict mapping each edge to its weight
    :param memo: dict to store the distance interpretation
    :return:
    """
    # init vars in base case
    if memo is None:
        memo = {}

    if node in memo:
        return memo[node]

    # assign leaf nodes
    if isinstance(node, Edge):
        w = weight_dict[(node.u, node.v)]
        memo[node] = w
        return w

    if isinstance(node, EmptySet):
        # ∅ interpreted as +∞
        memo[node] = float('inf')
        return memo[node]

    if isinstance(node, Epsilon):
        # 1 interpreted as 0
        memo[node] = 0
        return 0

    # recursively assign distance algebra for Kleene functions
    if isinstance(node, KleeneUnion):
        # union: the minimum of left/right
        d = min(get_distance_interpretation(node.left, weight_dict, memo),
                get_distance_interpretation(node.right, weight_dict, memo))
        memo[node] = d
        return d

    if isinstance(node, KleeneConcat):
        # concat: the sum of left/right
        d = get_distance_interpretation(node.left, weight_dict, memo) + \
            get_distance_interpretation(node.right, weight_dict, memo)
        memo[node] = d
        return d

    if isinstance(node, KleeneStar):
        # star: -∞ if d < 0, 0 otherwise
        pe = get_distance_interpretation(node.pe, weight_dict, memo)
        # closure rule
        d = float('-inf') if pe < 0 else 0
        memo[node] = d
        return d

    raise TypeError(f"Unknown node type: {node}")

###
# build the actual networkx graph representation of the DAG
###
def build_graph_from_kleene(root):
    # init vars
    dag_graph = nx.DiGraph()
    mapping = {}
    counter = [0]

    # init helper function to traverse root node and add all edges
    # also update mapping
    def visit(node):
        if node in mapping:
            return mapping[node]

        # assign a unique id
        nid = f"N{counter[0]}"
        counter[0] += 1
        mapping[node] = nid

        dag_graph.add_node(nid, obj=node)
        # recurse on children
        if isinstance(node, KleeneUnion):
            dag_graph.add_edge(nid, visit(node.left))
            dag_graph.add_edge(nid, visit(node.right))

        elif isinstance(node, KleeneConcat):
            dag_graph.add_edge(nid, visit(node.left))
            dag_graph.add_edge(nid, visit(node.right))

        elif isinstance(node, KleeneStar):
            dag_graph.add_edge(nid, visit(node.pe))

        return nid

    # create and return graph
    visit(root)
    return dag_graph, mapping


###
# runner function to plot the Kleene DAG
###
def plot_kleene_dag(root, weight_dict, filename="/kleene_dag.pdf"):
    # get graph and mapping
    dag_graph, mapping = build_graph_from_kleene(root)

    # compute distance interpretation from bottom-up
    distance_interpretation_memo = {}
    get_distance_interpretation(root, weight_dict, distance_interpretation_memo)

    # label nodes
    for nid, data in dag_graph.nodes(data=True):
        obj = data["obj"]

        if isinstance(obj, KleeneUnion):
            label = "+"
        elif isinstance(obj, KleeneConcat):
            label = "·"
        elif isinstance(obj, KleeneStar):
            label = "∗"
        elif isinstance(obj, Edge):
            label = f"<{obj.u},{obj.v}> ({weight_dict[(obj.u,obj.v)]})"
        elif isinstance(obj, Epsilon):
            label = "ε"
        elif isinstance(obj, EmptySet):
            label = "∅"
        else:
            label = "?"

        # format label for node and distance at this point
        distance_value = distance_interpretation_memo[obj]
        label = f"{label}\nD={distance_value}"

        # add node to graph
        dag_graph.nodes[nid]["label"] = label

    # make and save graph
    final_graph = to_agraph(dag_graph)
    final_graph.layout("dot")
    final_graph.draw(filename)


def main():
    """
    test case with five-node graph from figure 1 in lecture notes
    solution should be:
        (⟨a,b⟩⟨b,d⟩(⟨d,e⟩⟨e,d⟩)∗⟨d,a⟩)∗⟨a,b⟩(⟨b,c⟩+ ⟨b,d⟩(⟨d,e⟩⟨e,d⟩)∗⟨d,c⟩)
    current solution from this code is:
        Unsimplified:
        ((({ϵ} + (<a,b>((<b,d>(({ϵ} + (<d,e><e,d>)))*)((({ϵ} + (<d,e><e,d>)))*<d,a>)))))*(<a,b>(<b,c> + ((<b,d>(({ϵ} + (<d,e><e,d>)))*)((({ϵ} + (<d,e><e,d>)))*<d,c>)))))
        Simplified:
        (((<a,b>((<b,d>((<d,e><e,d>))*)(((<d,e><e,d>))*<d,a>))))*(<a,b>(<b,c> + ((<b,d>((<d,e><e,d>))*)(((<d,e><e,d>))*<d,c>)))))
    """

    # init graph, nodes, and starting node
    nodes = ["a", "b", "c", "d", "e"]
    edges_to_weights = {("a", "b"): 1,
                        ("b", "c"): 2,
                        ("b", "d"): -1,
                        ("d", "e"): -1,
                        ("e", "d"): 2,
                        ("d", "a"): 0,
                        ("d", "c"): 1}
    edges = [x for x in edges_to_weights.keys()]
    start = "a"

    # run kleene algo
    result = run_kleene_algorithm(nodes, edges, start)

    # get path expression for a -> c
    # this returns a node in the DAG
    pe_a_to_c = result['c']

    # print the path expression for this node
    print(f'Unsimplified:\n{pe_a_to_c}')

    # simplify the path expression according to the following rule:
    # ({ϵ} + p)* = p*
    pe_a_to_c_simplified = simplify_epsilon_star(pe_a_to_c)
    print(f'Simplified:\n{pe_a_to_c_simplified}')

    # make and save graph with distance interpretation
    plot_kleene_dag(pe_a_to_c_simplified, edges_to_weights)


if __name__ == '__main__':
    main()
