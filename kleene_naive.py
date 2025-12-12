"""
File: kleene_naive.py
Name: Jason Greenfield
Purpose: implement Kleene's algorithm using naive string representations for path expressions
Note: based on algorithm 1 in lecture notes 16 for COS 516, taught by Professor Zachary Kincaid at Princeton University
"""


def kleene_union(u, v):
    """
    Union returns the union of both sets of paths through either u or v
    For simplification:
        Case 1: consider the empty language ∅
            if either u or v is ∅, just return the other
    Otherwise, return (<u> + <v>)
    """
    if u == '∅':
        return v
    if v == '∅':
        return u
    if u == v:
        return u
    return f"({u} + {v})"


def kleene_concatenation(u, v):
    """
    Concatenation returns the sequence of u then v
    Two cases for simplification:
        Case 1: consider the empty language ∅
            if either u or v is '∅', return '∅' since you can't go through no nodes
        Case 2: consider the empty set / singleton word {ϵ}
            if either u or v is {ϵ}, return the other since sequencing through the empty set has no effect
    Otherwise, return (<u><v>)
    """
    if u == '∅' or v == '∅':
        return '∅'
    if u == '{ϵ}':
        return v
    if v == '{ϵ}':
        return u
    return f"({u})({v})"


def kleene_closure(u):
    """
    Closure returns the sequence of u then v
    Two cases for simplification:
        Case 1: consider the empty language ∅
            if u is ∅, return ∅ since you can't loop around nothing
        Case 2: consider the empty set / singleton word  {ϵ}
            same as above, if u is {ϵ}, return {ϵ} since you can't loop around nothing
    Otherwise, return (<u><v>)
    """
    if u == '{ϵ}' or u == '∅':
        # ∅* = {ϵ} and {ϵ}* = {ϵ}
        return '{ϵ}'
    return f"({u})*"


def kleene_path_expressions(nodes, edges, start):
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
    path_expressions = [['∅' for i in range(n)] for j in range(n)]

    # vals along the diagonal represent paths from a node to itself
    # these are represented with {ϵ}, the singleton language containing the empty word
    # this is equivalent to: 1L ≜ {ϵ} is the singleton language containing the empty word
    for i in range(n):
        path_expressions[i][i] = '{ϵ}'

    # Initialize direct edges
    for edge in edges:
        u, v = edge
        u_index, v_index = node_name_to_index[u], node_name_to_index[v]
        path_expressions[u_index][v_index] = f"<{u},{v}>"

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


def main():
    # init graph, nodes, and starting node
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"),
             ("b", "c"),
             ("b", "d"),
             ("d", "e"),
             ("e", "d"),
             ("d", "a"),
             ("d", "c")]
    start = "a"

    # run kleene algo
    result = kleene_path_expressions(nodes, edges, start)

    # get path expression for a -> c
    pe_a_c = result['c']
    print(pe_a_c)


if __name__ == '__main__':
    main()
