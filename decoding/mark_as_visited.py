from dijkstra import dijkstra

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

def double_dijkstra(working_graph, walk, kmer_count=True, precomputed_weights=None):
    """
    Combine dijkstra path nodes with shared neighbors for marking as visited.
    Runs dijkstra for both hap='m' and hap='p' to get comprehensive coverage.
    
    Args:
        working_graph: NetworkX graph
        walk: List of nodes representing the walk
        kmer_count: Boolean indicating whether to use kmer count for scoring
        precomputed_weights: Optional dict of precomputed edge weights for efficiency
    
    Returns:
        Set of nodes to mark as visited
    """
    if walk is None or len(walk) < 2:
        return get_shared_neighbors(working_graph, walk)
    
    # Get nodes from shared neighbors
    nodes_to_remove = get_shared_neighbors(working_graph, walk)
    
    # Get dijkstra path from first to last node of walk
    source = walk[0]
    sink = walk[-1]
    
    # Run dijkstra for both haplotypes
    for hap in ['m', 'p']:
        dijkstra_path, _ = dijkstra(working_graph, source, sink, hap, kmer_count, precomputed_weights)
        
        if dijkstra_path is not None:
            # Add all nodes from dijkstra path
            nodes_to_remove.update(set(dijkstra_path))
            # Add complements of all dijkstra path nodes
            nodes_to_remove.update({node ^ 1 for node in dijkstra_path})
    
    return nodes_to_remove

def mark_tips(working_graph, hap=None, graphic_preds=None, threshold=0.1, min_wrong_kmers=0.1):
    """
    Mark tips (nodes with in_degree=0 or out_degree=0) from the wrong haplotype as visited.
    
    Args:
        working_graph: NetworkX graph
        walk: List of nodes representing the walk
        hap: Target haplotype ('m' or 'p'), if None no haplotype filtering is applied
        threshold: Threshold for node predictions (nodes with predictions < -threshold are marked)
        min_wrong_kmers: Minimum number of wrong haplotype kmers required to flag a node
    
    Returns:
        Set of nodes to mark as visited
    """
    # If no haplotype specified, just return shared neighbors
    # Determine the YAK attribute to check
    if graphic_preds is not None:
        yak_mode = False
    else:
        yak_mode = True
        yak_attr = f'yak_{hap}'
        wrong_hap = 'p' if hap == 'm' else 'm'
        wrong_kmer_attr = f'kmer_count_{wrong_hap}'
    
    def is_wrong_haplotype(node):
        """Check if a node belongs to the wrong haplotype"""
        node_data = working_graph.nodes[node]
        if yak_mode:
            #print(node_data[wrong_kmer_attr] )
            return node_data[wrong_kmer_attr] > min_wrong_kmers
        else:
            return node_data['graphic_score'] < -threshold

    def is_neutral(node):
        """Check if a node is neutral (yak_hap < 1)"""
        node_data = working_graph.nodes[node]
        if yak_mode:
            return node_data[yak_attr] < 1
        else:
            return -threshold <= node_data['graphic_score'] <= threshold

    # Keep marking tips until no new tips are created
    nodes_to_remove = set()
    changed = True
    
    while changed:
        changed = False
        previous_size = len(nodes_to_remove)
        
        # Find all current tips (nodes with in_degree=0 or out_degree=0)
        # excluding nodes already marked for removal
        remaining_nodes = set(working_graph.nodes()) - nodes_to_remove
        current_tips = set()
        
        for node in remaining_nodes:
            # Calculate effective degrees (excluding nodes marked for removal)
            in_neighbors = set(working_graph.predecessors(node)) - nodes_to_remove
            out_neighbors = set(working_graph.successors(node)) - nodes_to_remove
            
            # Check if node is a tip (no in edges or no out edges)
            if len(in_neighbors) == 0 or len(out_neighbors) == 0:
                current_tips.add(node)
        
        # Filter tips to only include those from wrong haplotype
        for tip in current_tips:
            if is_wrong_haplotype(tip):
                nodes_to_remove.add(tip)
                nodes_to_remove.add(tip ^ 1)  # Also mark complement
        
        # Handle neutral nodes that are only connected to wrong haplotype nodes
        for node in remaining_nodes:
            if is_neutral(node):
                # Calculate effective neighbors (excluding nodes marked for removal)
                in_neighbors = set(working_graph.predecessors(node)) - nodes_to_remove
                out_neighbors = set(working_graph.successors(node)) - nodes_to_remove
                
                # Check if all neighbors are from wrong haplotype
                all_neighbors = in_neighbors | out_neighbors
                if all_neighbors and all(is_wrong_haplotype(neighbor) for neighbor in all_neighbors):
                    nodes_to_remove.add(node)
                    nodes_to_remove.add(node ^ 1)  # Also mark complement
        
        # Check if any new nodes were added
        if len(nodes_to_remove) != previous_size:
            changed = True
    
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

def get_dual_walk_1_hop(working_graph, walk, alt_walk):
    nodes_from_walk_1_hop = get_1_hop_neighbors(working_graph, walk)
    nodes_from_alt_walk_1_hop = get_1_hop_neighbors(working_graph, alt_walk)
    all_nodes_to_remove = nodes_from_walk_1_hop | nodes_from_alt_walk_1_hop
    return all_nodes_to_remove

def get_dual_walk_shared(working_graph, walk, alt_walk):
    nodes_to_remove = set()
    nodes_to_remove.update(get_shared_neighbors(working_graph, walk))
    nodes_to_remove.update(get_shared_neighbors(working_graph, alt_walk))
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