
def get_shared_neighbors(working_graph, walk):
    if walk is None:
        return set()
    
    trans = set()
    nodes_to_remove = set()
    for ss, dd in zip(walk[:-1], walk[1:]):
        t1 = set(working_graph.successors(ss)) & set(working_graph.predecessors(dd))
        t2 = {t^1 for t in t1}
        trans = trans | t1 | t2
    nodes_to_remove.update(trans)
    nodes_to_remove.update({node ^ 1 for node in trans})
    nodes_to_remove.update(set(walk))
    nodes_to_remove.update({node ^ 1 for node in walk})

    return nodes_to_remove

def get_1_hop_neighbors(working_graph, walk):
    nodes_to_remove = set()
    
    # Get all successors and predecessors of nodes in the walk
    for node in walk:
        # Add successors
        nodes_to_remove.update(working_graph.successors(node))
        # Add predecessors
        nodes_to_remove.update(working_graph.predecessors(node))
    
    # Add the walk nodes themselves
    nodes_to_remove.update(set(walk))
    # Add the reverse complement of all nodes
    nodes_to_remove.update({node ^ 1 for node in nodes_to_remove})
    
    return nodes_to_remove

def get_2_hop_neighbors(working_graph, walk):
    nodes_to_remove = set()
    
    # First get the 1-hop neighborhood
    one_hop_nodes = set()
    
    # Get all successors and predecessors of nodes in the walk
    for node in walk:
        # Add successors
        one_hop_nodes.update(working_graph.successors(node))
        # Add predecessors
        one_hop_nodes.update(working_graph.predecessors(node))
    
    # Add the walk nodes themselves
    one_hop_nodes.update(set(walk))
    
    # Now get the second hop - successors and predecessors of the 1-hop nodes
    for node in one_hop_nodes:
        # Add successors
        nodes_to_remove.update(working_graph.successors(node))
        # Add predecessors
        nodes_to_remove.update(working_graph.predecessors(node))
    
    # Add the 1-hop nodes to the result
    nodes_to_remove.update(one_hop_nodes)
    
    # Add the reverse complement of all nodes
    nodes_to_remove.update({node ^ 1 for node in nodes_to_remove})
    
    return nodes_to_remove