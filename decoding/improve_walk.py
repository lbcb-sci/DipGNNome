import networkx as nx
from dijkstra import dijkstra, dijkstra_score

def min_wrong_haplo_dijkstra(working_graph, walk, hap, config, print_scores=False):
    """
    Convert a single component path to a haplotype-specific path.
    Uses Dijkstra's algorithm with a modified weight function to find the path with lowest
    opposite haplotype YAK scores.
    
    Args:
        comp_path: Single path through the component
        working_graph: NetworkX graph of the component
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary
    
    Returns:
        Haplotype-specific path or None if no valid path exists
    """
    # Check if the path is empty or too short
    if not walk or len(walk) < 3:
        return []

    # Get prefix_length attributes for all edges
    prefix_lengths = nx.get_edge_attributes(working_graph, 'prefix_length')
    
    # Calculate original path length
    orig_length = sum(prefix_lengths[(u, v)]
                     for u, v in zip(walk[:-1], walk[1:]))
    
    # Use opposite haplotype's YAK scores
    yak_attr = 'yak_p' if hap == 'm' else 'yak_m'
    
    print(f"Graph has {working_graph.number_of_nodes():,} nodes and {working_graph.number_of_edges():,} edges")
    print(f"Original path length: {orig_length:,}bp")

    # Create a modified graph for Dijkstra's algorithm
    weighted_graph = working_graph.copy()
    
    # Set edge weights based on target node's YAK score
    for u, v, data in weighted_graph.edges(data=True):
        target_yak = max(0, working_graph.nodes[v][yak_attr])
        data['weight'] = target_yak

    # Find shortest path using Dijkstra
    #improved_walk = nx.dijkstra_path(weighted_graph, walk[0], walk[-1], weight='weight')
    improved_walk, dijkstra_cost = dijkstra(weighted_graph, walk[0], walk[-1], hap)
    
    if improved_walk is None:
        print("Warning, No Dijkstra path found")
        return walk
    
    # Calculate lengths (reusing the prefix_lengths dictionary)
    orig_length = sum(prefix_lengths[(u, v)]
                     for u, v in zip(walk[:-1], walk[1:]))
    imp_length = sum(prefix_lengths[(u, v)]
                    for u, v in zip(improved_walk[:-1], improved_walk[1:]))

    # Calculate dijkstra scores
    orig_dijkstra_score = dijkstra_score(weighted_graph, walk, hap)
    imp_dijkstra_score = dijkstra_score(weighted_graph, improved_walk, hap)

    if print_scores:
        # Convert scores to per million base pairs
        orig_score_per_mbp = (orig_dijkstra_score / orig_length) * 1_000_000 if orig_length > 0 else 0
        imp_score_per_mbp = (imp_dijkstra_score / imp_length) * 1_000_000 if imp_length > 0 else 0

        print("\nComparison of walks:")
        print(f"Original walk ({len(walk)} nodes, {orig_length:,}bp):")
        print(f"- Dijkstra score: {orig_dijkstra_score:.1f} ({orig_score_per_mbp:.1f} per Mbp)")

        print(f"\nImproved walk ({len(improved_walk)} nodes, {imp_length:,}bp):")
        print(f"- Dijkstra score: {imp_dijkstra_score:.1f} ({imp_score_per_mbp:.1f} per Mbp)")

        print(f"\nChanges:")
        print(f"- Length: {imp_length-orig_length:+,}bp ({imp_length/orig_length*100:.1f}% of original)")
        print(f"- Dijkstra score: {imp_dijkstra_score-orig_dijkstra_score:+.1f} ({imp_score_per_mbp-orig_score_per_mbp:+.1f} per Mbp)")

    return improved_walk

def min_wrong_haplo_bellman_ford(working_graph, walk, hap, config, length_threshold=0.95):
    """
    Convert a single component path to a haplotype-specific path.
    Uses Bellman-Ford algorithm with a modified weight function to find the path with lowest
    opposite haplotype YAK scores while favoring correct and ambiguous nodes.
    
    Args:
        comp_path: Single path through the component
        working_graph: NetworkX graph of the component
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        length_threshold: Not used in this implementation
    
    Returns:
        Haplotype-specific path or None if no valid path exists
    """
    # Check if the path is empty or too short
    if not walk or len(walk) < 3:
        return []

    # Get prefix_length attributes for all edges
    prefix_lengths = nx.get_edge_attributes(working_graph, 'prefix_length')
    
    # Calculate original path length
    orig_length = sum(prefix_lengths[(u, v)]
                     for u, v in zip(walk[:-1], walk[1:]))
    
    # Use opposite haplotype's YAK scores
    yak_attr = 'yak_p' if hap == 'm' else 'yak_m'
    
    print(f"Graph has {working_graph.number_of_nodes():,} nodes and {working_graph.number_of_edges():,} edges")
    print(f"Original path length: {orig_length:,}bp")

    # Create a modified graph for Bellman-Ford algorithm
    weighted_graph = working_graph.copy()
    
    # Set edge weights based on target node's YAK score
    for u, v, data in weighted_graph.edges(data=True):
        target_yak = weighted_graph.nodes[v][yak_attr]
        # If the node is not wrong haplotype (0 or 1), give it a negative weight to favor these paths
        if target_yak != -1:
            data['weight'] = -0.1
        else:
            data['weight'] = 1  # Penalize wrong haplotype nodes
            
    # Find shortest path using Bellman-Ford
    improved_walk = nx.shortest_path(weighted_graph, walk[0], walk[-1], weight='weight', method='bellman-ford')
    
    # Calculate lengths (reusing the prefix_lengths dictionary)
    orig_length = sum(prefix_lengths[(u, v)]
                     for u, v in zip(walk[:-1], walk[1:]))
    imp_length = sum(prefix_lengths[(u, v)]
                    for u, v in zip(improved_walk[:-1], improved_walk[1:]))

    # Calculate dijkstra scores
    orig_dijkstra_score = dijkstra_score(weighted_graph, walk, hap)
    imp_dijkstra_score = dijkstra_score(weighted_graph, improved_walk, hap)
    
    # Convert scores to per million base pairs
    orig_score_per_mbp = (orig_dijkstra_score / orig_length) * 1_000_000 if orig_length > 0 else 0
    imp_score_per_mbp = (imp_dijkstra_score / imp_length) * 1_000_000 if imp_length > 0 else 0

    print("\nComparison of walks:")
    print(f"Original walk ({len(walk)} nodes, {orig_length:,}bp):")
    print(f"- Dijkstra score: {orig_dijkstra_score:.1f} ({orig_score_per_mbp:.1f} per Mbp)")

    print(f"\nImproved walk ({len(improved_walk)} nodes, {imp_length:,}bp):")
    print(f"- Dijkstra score: {imp_dijkstra_score:.1f} ({imp_score_per_mbp:.1f} per Mbp)")

    print(f"\nChanges:")
    print(f"- Length: {imp_length-orig_length:+,}bp ({imp_length/orig_length*100:.1f}% of original)")
    print(f"- Dijkstra score: {imp_dijkstra_score-orig_dijkstra_score:+.1f} ({imp_score_per_mbp-orig_score_per_mbp:+.1f} per Mbp)")

    return improved_walk


def crop_path_by_haplotype(working_graph, walk, hap, config):
    """
    Crop a path into multiple valid segments by removing sections with too many wrong haplotype nodes.
    Similar to chop_chop_martin but works with NetworkX graph.
    Neutral (yak=0) nodes don't count towards wrong_haplo count but also don't reset it.
    
    Args:
        path: List of nodes representing the path
        working_graph: NetworkX graph containing the nodes
        hap: Target haplotype ('m' or 'p')
        max_wrong_haplo: Maximum number of consecutive wrong haplotype nodes allowed
    
    Returns:
        List of valid path segments
    """
    max_wrong_haplo = config['max_wrong_haplo']
    if not walk:
        return []
        
    valid_segments = []
    current_segment = []
    wrong_haplo_count = 0
    wrong_haplo_total = 0
    just_started = True
    
    for i, node in enumerate(walk):
        yak_score = working_graph.nodes[node][f'yak_{hap}']
        
        if yak_score == -1:  # Wrong haplotype
            wrong_haplo_count += 1
            wrong_haplo_total += 1
        elif yak_score == 1:  # Correct haplotype
            just_started = False
            wrong_haplo_count = 0
            wrong_haplo_total = 0
        else:  # Ambiguous (0) - don't reset counter but don't increment it either
            just_started = False
            wrong_haplo_total += 1
            
        if not just_started:
            current_segment.append(node)
            
        if wrong_haplo_count >= max_wrong_haplo:
            if len(current_segment) > max_wrong_haplo:  # Only keep segments longer than max_wrong_haplo
                valid_segment = current_segment[:-wrong_haplo_total]
                if len(valid_segment) > 2:
                    valid_segments.append(valid_segment)
            current_segment = []
            wrong_haplo_count = 0
            wrong_haplo_total = 0
            just_started = True
            
            # Start new segment after the wrong haplotype section
            next_good_idx = i + 1
            while next_good_idx < len(walk):
                next_node = walk[next_good_idx]
                next_score = working_graph.nodes[next_node][f'yak_{hap}']
                if next_score != -1:  # If not wrong haplotype
                    break
                next_good_idx += 1
            
            # Skip the wrong haplotype section
            if next_good_idx < len(walk):
                i = next_good_idx - 1  # -1 because loop will increment i
            
    # Handle the last segment if it's valid
    if len(current_segment) > max_wrong_haplo:
        valid_segments.append(current_segment)
        
    return valid_segments

def improve_greedy_walk_by_exploration(nx_graph, original_path, hap, config, penalty_wrong_hap=0.6, penalty_ambiguous=0.05, max_iterations=1000, max_exploration_diff=0.6, length_difference_factor=0.05):
    """Improve a greedy walk by exploring alternative paths from each node.
    
    Args:
        nx_graph: NetworkX graph
        original_path: List of nodes representing the original greedy walk
        hap: Target haplotype ('m' or 'p' or None)
        penalty_wrong_hap: Penalty to apply to edges leading to wrong haplotype nodes
        penalty_ambiguous: Penalty to apply to edges leading to ambiguous nodes
        max_iterations: Maximum number of improvement iterations to perform
        max_exploration_diff: Maximum score difference to consider a branch for exploration
        length_difference_factor: Maximum allowed relative length difference for replacement
    
    Returns:
        tuple: (improved_path, improved_length, wrong_haplo_count) where improved_path is list of nodes
    """
    penalty_wrong_hap = config['penalty_wrong_hap']
    #penalty_ambiguous = config['penalty_ambiguous']
    #max_iterations = config['max_iterations']
    #max_exploration_diff = config['max_exploration_diff']
    #length_difference_factor = config['length_difference_factor']

    if not original_path or len(original_path) < 2:
        return original_path, 0, 0
    
    current_path = original_path.copy()
    improved = True
    iteration = 0
    
    # Get all attributes once for efficiency
    edge_scores = nx.get_edge_attributes(nx_graph, 'score')
    edge_lengths = nx.get_edge_attributes(nx_graph, 'prefix_length')
    node_read_lengths = nx.get_node_attributes(nx_graph, 'read_length')
    node_ambiguous = nx.get_node_attributes(nx_graph, 'ambigious')
    
    # Get haplotype attributes if specified
    if hap is not None:
        node_yak_scores = nx.get_node_attributes(nx_graph, f'yak_{hap}')
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        print(f"Improvement iteration {iteration}")
        
        # Try to improve from each node in the current path
        for start_idx in range(len(current_path) - 1):  # Don't start from the last node
            start_node = current_path[start_idx]
            
            # Get all possible neighbors (not just the one in the current path)
            out_edges = list(nx_graph.out_edges(start_node, data=True))
            if not out_edges:
                continue
            
            # Find the current next node in the path and its score
            current_next_node = current_path[start_idx + 1]
            current_edge_score = edge_scores[(start_node, current_next_node)]
            
            # Create visited set up to the start node (including complements and transitive nodes)
            visited = set()
            for i in range(start_idx + 1):  # Include the start node
                node = current_path[i]
                visited.add(node)
                visited.add(node ^ 1)
                
                # Add transitive nodes for edges in the path so far
                if i > 0:
                    prev_node = current_path[i - 1]
                    transitive_nodes = set(nx_graph.successors(prev_node)) & set(nx_graph.predecessors(node))
                    visited.update(transitive_nodes)
                    visited.update({node ^ 1 for node in transitive_nodes})

            # Create set of nodes that come after start_idx in the current path
            nodes_to_come = set(current_path[start_idx + 1:])

            # Try each neighbor as an alternative
            for src, dst, data in out_edges:
                # Skip if this is the same edge already in the path
                if dst == current_next_node:
                    continue
                
                # Only consider branch if score difference is within max_exploration_diff
                alt_edge_score = data['score']
                if abs(alt_edge_score - current_edge_score) > max_exploration_diff:
                    continue
                
                # Skip if the alternative destination is already visited
                if dst in visited:
                    continue
                
                # Perform greedy walk from this alternative neighbor with nodes_to_come
                alt_path, alt_length, alt_wrong_haplo, intersection_node = walk_graph_from_node(
                    nx_graph, dst, visited.copy(), hap=hap, 
                    penalty_wrong_hap=penalty_wrong_hap, 
                    penalty_ambiguous=penalty_ambiguous,
                    nodes_to_come=nodes_to_come
                )
                
                if not alt_path:
                    continue
                
                if intersection_node is not None:
                    # Paths intersect - find where in original path
                    orig_segment_end = current_path.index(intersection_node, start_idx + 1)
                    
                    # Calculate length of original segment
                    orig_segment_length = 0
                    for i in range(start_idx, orig_segment_end):
                        orig_segment_length += edge_lengths[(current_path[i], current_path[i + 1])]
                    
                    # Calculate length of alternative segment (including the initial edge)
                    alt_segment_length = edge_lengths[(start_node, dst)]  # Edge from start_node to dst
                    for i in range(len(alt_path) - 1):
                        alt_segment_length += edge_lengths[(alt_path[i], alt_path[i + 1])]
                    
                    # Only check if alternative is too short
                    if alt_segment_length < orig_segment_length * (1 - length_difference_factor):
                        continue
                    
                    # Calculate bad haplotype scores if haplotype is specified
                    if hap is not None:
                        # Original segment bad haplo score
                        orig_bad_haplo_score = 0
                        for i in range(start_idx, orig_segment_end):  # Use edges, not nodes
                            if i + 1 < len(current_path):
                                edge = (current_path[i], current_path[i + 1])
                                target_node = current_path[i + 1]
                                if node_yak_scores.get(target_node, 0) == -1:
                                    orig_bad_haplo_score += edge_lengths.get(edge, 0)
                        
                        # Alternative segment bad haplo score
                        alt_bad_haplo_score = 0
                        # Include the initial edge from start_node to dst
                        if node_yak_scores.get(dst, 0) == -1:
                            alt_bad_haplo_score += edge_lengths.get((start_node, dst), 0)
                        # Add edges within alt_path
                        for i in range(len(alt_path) - 1):
                            edge = (alt_path[i], alt_path[i + 1])
                            target_node = alt_path[i + 1]
                            if node_yak_scores.get(target_node, 0) == -1:
                                alt_bad_haplo_score += edge_lengths.get(edge, 0)
                        
                        # Only replace if alternative has lower bad haplotype score
                        if alt_bad_haplo_score >= orig_bad_haplo_score:
                            continue
                    
                    # Construct new path
                    new_path = (current_path[:start_idx + 1] + 
                              alt_path + 
                              current_path[orig_segment_end + 1:])
                    
                    print(f"  Improved segment from node {start_idx}: "
                          f"original length {orig_segment_length} -> {alt_segment_length}")
                    if hap is not None:
                        print(f"    Bad haplo score: {orig_bad_haplo_score} -> {alt_bad_haplo_score}")
                    current_path = new_path
                    improved = True
                    break  # Move to next start node
                else:
                    # Alternative path doesn't intersect - compare with remainder of original path
                    orig_remainder_length = 0
                    for i in range(start_idx, len(current_path) - 1):
                        orig_remainder_length += edge_lengths[(current_path[i], current_path[i + 1])]
                    # Add length of last node in original path
                    orig_remainder_length += node_read_lengths.get(current_path[-1], 0)
                    
                    # Calculate alternative path length (including initial edge)
                    alt_total_length = edge_lengths[(start_node, dst)] + alt_length
                    
                    # Check length difference criterion
                    length_diff_ratio = abs(alt_total_length - orig_remainder_length) / orig_remainder_length if orig_remainder_length > 0 else 0
                    if length_diff_ratio > length_difference_factor:
                        continue
                    
                    # Calculate bad haplotype scores if haplotype is specified
                    if hap is not None:
                        # Original remainder bad haplo score
                        orig_bad_haplo_score = 0
                        for i in range(start_idx, len(current_path) - 1):  # Use edges, not nodes
                            edge = (current_path[i], current_path[i + 1])
                            target_node = current_path[i + 1]
                            if node_yak_scores.get(target_node, 0) == -1:
                                orig_bad_haplo_score += edge_lengths.get(edge, 0)
                        
                        # Alternative path bad haplo score
                        alt_bad_haplo_score = 0
                        # Include the initial edge from start_node to dst
                        if node_yak_scores.get(dst, 0) == -1:
                            alt_bad_haplo_score += edge_lengths.get((start_node, dst), 0)
                        # Add edges within alt_path
                        for i in range(len(alt_path) - 1):
                            edge = (alt_path[i], alt_path[i + 1])
                            target_node = alt_path[i + 1]
                            if node_yak_scores.get(target_node, 0) == -1:
                                alt_bad_haplo_score += edge_lengths.get(edge, 0)
                        
                        # Only replace if alternative has lower bad haplotype score
                        if alt_bad_haplo_score >= orig_bad_haplo_score:
                            continue
                    
                    new_path = current_path[:start_idx + 1] + alt_path
                    print(f"  Improved remainder from node {start_idx}: "
                          f"original length {orig_remainder_length} -> {alt_total_length}")
                    if hap is not None:
                        print(f"    Bad haplo score: {orig_bad_haplo_score} -> {alt_bad_haplo_score}")
                    current_path = new_path
                    improved = True
                    break  # Move to next start node
            
            if improved:
                break  # Start over from the beginning with the improved path
    
    # Calculate final path statistics
    final_length = 0
    final_wrong_haplo = 0
    
    for i in range(len(current_path) - 1):
        edge = (current_path[i], current_path[i + 1])
        final_length += edge_lengths[edge]
        
        # Count wrong haplotype length
        if hap is not None:
            next_node = current_path[i + 1]
            if node_yak_scores.get(next_node, 0) == -1:  # Wrong haplotype
                final_wrong_haplo += edge_lengths[edge]
    
    # Add length of last node
    if current_path:
        final_length += node_read_lengths.get(current_path[-1], 0)
    
    print(f"Walk improvement completed after {iteration} iterations")
    correct_both_ends_haplo(nx_graph, current_path, hap, config)
    return current_path

def walk_graph_from_node(nx_graph, start_node, initial_visited, hap=None, penalty_wrong_hap=0.6, penalty_ambiguous=0.05, nodes_to_come=None):
    """Perform greedy walk from a specific node with pre-existing visited nodes.
    
    This is a helper function for improve_greedy_walk that allows starting a walk
    with some nodes already marked as visited.
    
    Args:
        nx_graph: NetworkX graph
        start_node: Node to start path from
        initial_visited: Set of nodes already visited (including complements)
        hap: Target haplotype ('m' or 'p' or None)
        penalty_wrong_hap: Penalty to apply to edges leading to wrong haplotype nodes
        penalty_ambiguous: Penalty to apply to edges leading to ambiguous nodes
        nodes_to_come: Set of nodes from the main path that we might intersect with
    
    Returns:
        tuple: (path, path_length, wrong_haplo_count, intersection_node)
    """
    if start_node not in nx_graph:
        return [], 0, 0, None
    
    # Get attributes once for efficiency
    edge_lengths = nx.get_edge_attributes(nx_graph, 'prefix_length')
    node_read_lengths = nx.get_node_attributes(nx_graph, 'read_length')
    node_ambiguous = nx.get_node_attributes(nx_graph, 'ambigious')
    
    # Get haplotype attributes if specified
    if hap is not None:
        node_yak_scores = nx.get_node_attributes(nx_graph, f'yak_{hap}')
    
    path = [start_node]
    visited = initial_visited.copy()
    visited.add(start_node)
    visited.add(start_node ^ 1)
    wrong_haplo_len = 0
    current_node = start_node
    intersection_node = None
    
    # Check if start_node is already in nodes_to_come
    if nodes_to_come and start_node in nodes_to_come:
        intersection_node = start_node
    
    while intersection_node is None:
        # Get all possible next edges
        out_edges = list(nx_graph.out_edges(current_node, data=True))
        if not out_edges:
            break
            
        # Filter out edges leading to visited nodes or their complements
        valid_edges = []
        for src, dst, data in out_edges:
            if dst not in visited:
                valid_edges.append((src, dst, data))
        
        if not valid_edges:
            break
        
        # Greedy mode: sort edges by score, applying penalty for wrong haplotype nodes
        if hap is not None:
            # Apply penalty to edges leading to wrong haplotype nodes
            scored_edges = []
            for src, dst, data in valid_edges:
                score = data['score']
                # Check haplotype of destination node
                if node_ambiguous.get(dst, 0) == 1:  
                    score -= penalty_ambiguous
                if node_yak_scores.get(dst, 0) < 0:  
                    score -= penalty_wrong_hap
                scored_edges.append((src, dst, data, score))
            next_edges = sorted(scored_edges, key=lambda x: x[3], reverse=True)
            
            if next_edges:
                _, next_node, _, _ = next_edges[0]
            else:
                break
        else:
            # No haplotype constraints, sort by raw score
            next_edges = sorted(valid_edges, key=lambda x: x[2]['score'], reverse=True)
            if next_edges:
                _, next_node, _ = next_edges[0]
            else:
                break
        
        # Check if next_node is in nodes_to_come (intersection with main path)
        if nodes_to_come and next_node in nodes_to_come:
            intersection_node = next_node
        
        # If using haplotype constraints, count wrong haplotype length
        if hap is not None:
            if node_yak_scores.get(next_node, 0) == -1:  # Wrong haplotype
                edge = (current_node, next_node)
                wrong_haplo_len += edge_lengths.get(edge, 0)
        
        path.append(next_node)
        visited.add(next_node)
        visited.add(next_node ^ 1)  # Add complement
        
        # Get nodes that are both successors of current and predecessors of next_node
        transitive_nodes = set(nx_graph.successors(current_node)) & set(nx_graph.predecessors(next_node))
        # Add these nodes and their complements to visited
        visited.update(transitive_nodes)
        visited.update({node ^ 1 for node in transitive_nodes})
        current_node = next_node
    
    # Calculate path length
    path_length = 0
    for i in range(len(path) - 1):
        path_length += edge_lengths[(path[i], path[i + 1])]
    
    # Add length of last node
    if path:
        path_length += node_read_lengths.get(path[-1], 0)
    
    return path, path_length, wrong_haplo_len, intersection_node

def improve_double_dfs(nx_graph, original_path, hap, config, choking=False):
    """
    Improve a greedy walk using double DFS approach.
    
    Performs DFS from start to end and from end to start on reversed graph,
    creates a subgraph from the intersection, removes low-scoring edges,
    and finds the path with minimum wrong haplotype node length.
    
    Args:
        nx_graph: NetworkX graph
        original_path: List of nodes representing the original greedy walk
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary
        choking: If True, use choking points approach; if False, find direct shortest path
    
    Returns:
        List of nodes representing the improved path
    """
    if not original_path or len(original_path) < 2:
        return original_path
    
    start_node = original_path[0]
    end_node = original_path[-1]
    
    print(f"Performing double DFS from {start_node} to {end_node}")
    
    # Step 1: DFS from start to find all reachable nodes
    forward_reachable = set(nx.dfs_postorder_nodes(nx_graph, source=start_node))
    print(f"Forward DFS found {len(forward_reachable)} reachable nodes")
    
    # Step 2: DFS from end on reversed graph to find all nodes that can reach end
    reversed_graph = nx_graph.reverse()
    backward_reachable = set(nx.dfs_postorder_nodes(reversed_graph, source=end_node))
    print(f"Backward DFS found {len(backward_reachable)} nodes that can reach end")
    
    # Step 3: Create subgraph from intersection (nodes on paths from start to end)
    path_nodes = forward_reachable.intersection(backward_reachable)
    print(f"Intersection contains {len(path_nodes)} nodes on paths from start to end")
    
    if not path_nodes:
        print("No valid paths found between start and end nodes")
        return original_path
    
    # Create subgraph with only path nodes
    subgraph = nx_graph.subgraph(path_nodes).copy()
    print(f"Created subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    
    # Step 4: Remove edges whose score is more than 0.1 smaller than highest scoring out edge
    edges_to_remove = []
    
    for node in subgraph.nodes():
        out_edges = list(subgraph.out_edges(node, data=True))
        if len(out_edges) <= 1:
            continue  # Skip nodes with 0 or 1 outgoing edges
        
        # Find maximum score among outgoing edges
        max_score = max(data['score'] for _, _, data in out_edges)
        
        # Mark edges for removal if they're more than 0.1 below max score
        for src, dst, data in out_edges:
            if data['score'] < max_score - 0.1:
                edges_to_remove.append((src, dst))
    
    # Remove the marked edges
    subgraph.remove_edges_from(edges_to_remove)
    print(f"Removed {len(edges_to_remove)} low-scoring edges")
    print(f"Final subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    
    # Step 5: Check if result is acyclic
    is_acyclic = nx.is_directed_acyclic_graph(subgraph)
    print(f"Subgraph is {'acyclic' if is_acyclic else 'contains cycles'}")
    
    # Step 6: Find best path based on choking parameter
    if hap is None:
        print("No haplotype specified, returning original path")
        return original_path
    
    # Get node attributes
    node_read_lengths = nx.get_node_attributes(nx_graph, 'read_length')
    node_yak_scores = nx.get_node_attributes(nx_graph, f'yak_{hap}')
    
    if not choking:
        # Direct shortest path approach
        print("Finding direct shortest path from start to end")
        
        # Create weighted graph
        weighted_graph = subgraph.copy()
        
        # Set edge weights based on target node's wrong haplotype length
        for u, v, data in weighted_graph.edges(data=True):
            target_node = v
            if node_yak_scores.get(target_node, 0) == -1:  # Wrong haplotype
                weight = node_read_lengths.get(target_node, 0)
            else:
                weight = 0  # No penalty for correct or ambiguous nodes
            data['weight'] = weight
        
        try:
            best_path = nx.dijkstra_path(weighted_graph, start_node, end_node, weight='weight')
            
            # Calculate wrong haplotype length for reporting
            wrong_haplo_length = sum(node_read_lengths.get(node, 0) 
                                   for node in best_path 
                                   if node_yak_scores.get(node, 0) == -1)
            
            print(f"Best path has {wrong_haplo_length:,}bp of wrong haplotype nodes")
            print(f"Best path has {len(best_path)} nodes")
            
            return best_path
            
        except nx.NetworkXNoPath:
            print("No path found, returning original path")
            return original_path
    
    else:
        # Choking points approach
        choking_points = [start_node]  # Always include start node
        candidate_nodes = [node for node in original_path[1:-1] if node in subgraph.nodes()]
        
        print("Identifying choking points from original greedy path...")
        for node in candidate_nodes:
            # Create a temporary graph without this node
            temp_nodes = [n for n in subgraph.nodes() if n != node]
            temp_subgraph = subgraph.subgraph(temp_nodes)
            
            # Check if there's still a path from start to end
            try:
                nx.shortest_path(temp_subgraph, start_node, end_node)
            except nx.NetworkXNoPath:
                # No path exists without this node, so it's a choking point
                choking_points.append(node)
        
        choking_points.append(end_node)  # Always include end node
        choking_points = sorted(set(choking_points), key=lambda x: original_path.index(x))  # Keep original order
        print(f"Found {len(choking_points)} choking points: {choking_points}")
        
        # Find best path segments between consecutive choking points
        best_full_path = []
        
        for i in range(len(choking_points) - 1):
            segment_start = choking_points[i]
            segment_end = choking_points[i + 1]
            
            # Find shortest path between choking points based on wrong haplotype length
            print(f"Finding shortest path between choking points {segment_start} -> {segment_end}")
            
            # Create weighted subgraph for this segment
            segment_subgraph = subgraph.copy()
            
            # Set edge weights based on target node's wrong haplotype length
            for u, v, data in segment_subgraph.edges(data=True):
                target_node = v
                if node_yak_scores.get(target_node, 0) == -1:  # Wrong haplotype
                    weight = node_read_lengths.get(target_node, 0)
                else:
                    weight = 0  # No penalty for correct or ambiguous nodes
                data['weight'] = weight
            
            try:
                best_segment = nx.dijkstra_path(segment_subgraph, segment_start, segment_end, weight='weight')
                
                # Calculate wrong haplotype length for reporting
                wrong_haplo_length = sum(node_read_lengths.get(node, 0) 
                                       for node in best_segment 
                                       if node_yak_scores.get(node, 0) == -1)
                
                print(f"Best segment has {wrong_haplo_length:,}bp of wrong haplotype nodes")
                
            except nx.NetworkXNoPath:
                print(f"No path found for segment {segment_start} -> {segment_end}, using direct connection")
                best_segment = [segment_start, segment_end]
            
            # Add segment to full path (avoid duplicating choking points)
            if i == 0:
                best_full_path.extend(best_segment)
            else:
                best_full_path.extend(best_segment[1:])  # Skip first node to avoid duplication
        
        print(f"Final best path has {len(best_full_path)} nodes")
        return best_full_path
        #return correct_both_ends_haplo(nx_graph, best_full_path, hap, config)

def correct_end_haplo(working_graph, walk, hap, config):
    """
    Correct the end of a walk if it ends with heterozygous wrong haplotype nodes.
    
    Checks if the last node of the walk is heterozygous and from the wrong haplotype.
    If yes, goes back to the first correct haplotype node and runs a greedy path from there,
    which never includes any wrong haplotype nodes (stops rather than include them).
    
    Args:
        working_graph: NetworkX graph containing the nodes
        walk: List of nodes representing the walk
        hap: Target haplotype ('m' or 'p')
        config: Configuration dictionary
    
    Returns:
        Corrected walk or original walk if no correction needed
    """
    if not walk or len(walk) < 2:
        return walk
    
    # Check if last node is heterozygous and wrong haplotype
    last_node = walk[-1]
    yak_score = working_graph.nodes[last_node][f'yak_{hap}']
    
    # If last node is not both heterozygous and wrong haplotype, return original walk
    if not yak_score == -1:
        return walk
    
    print(f"Last node {last_node} is heterozygous and wrong haplotype, correcting...")
    
    # Find the last correct haplotype node going backwards
    correct_start_idx = None
    for i in range(len(walk) - 1, -1, -1):
        node = walk[i]
        node_yak = working_graph.nodes[node][f'yak_{hap}']
        if node_yak != -1:  # Correct haplotype
            correct_start_idx = i
            break
    
    if correct_start_idx is None:
        print("No correct haplotype node found in walk")
        return walk
    
    # Keep path up to the correct haplotype node
    corrected_walk = walk[:correct_start_idx + 1]
    start_node = walk[correct_start_idx]
    
    print(f"Starting greedy extension from node {start_node} at position {correct_start_idx}")
    
    # Run greedy path from the correct haplotype node, avoiding wrong haplotype
    extended_path = greedy_extend_avoiding_wrong_haplo(working_graph, start_node, hap, config)
    
    # Combine the paths (avoid duplicating the start node)
    if extended_path and len(extended_path) > 1:
        corrected_walk.extend(extended_path[1:])
    
    print(f"Corrected walk: {len(walk)} -> {len(corrected_walk)} nodes")
    return corrected_walk

def correct_both_ends_haplo(working_graph, walk, hap, config):
    """
    Correct both the start and end of a walk if they have wrong haplotype nodes.
    
    First corrects the end using correct_end_haplo, then reverses the graph and walk
    to correct the start using the same function, then reverses back.
    
    Args:
        working_graph: NetworkX graph containing the nodes
        walk: List of nodes representing the walk
        hap: Target haplotype ('m' or 'p')
        config: Configuration dictionary
    
    Returns:
        Walk corrected at both ends
    """
    if not walk or len(walk) < 2:
        return walk
    
    print("Correcting both ends of walk...")
    
    # Step 1: Correct the end
    print("Step 1: Correcting end of walk")
    corrected_walk = correct_end_haplo(working_graph, walk, hap, config)
    
    # Step 2: Correct the start by reversing everything
    print("Step 2: Correcting start of walk (via reversal)")
    
    # Reverse the walk
    reversed_walk = corrected_walk[::-1]
    
    # Create reversed graph
    reversed_graph = working_graph.reverse()
    
    # Correct the "end" of the reversed walk (which is actually the start of the original)
    corrected_reversed_walk = correct_end_haplo(reversed_graph, reversed_walk, hap, config)
    
    # Reverse back to get the final corrected walk
    final_corrected_walk = corrected_reversed_walk[::-1]
    
    print(f"Final corrected walk: {len(walk)} -> {len(final_corrected_walk)} nodes")
    return final_corrected_walk

def greedy_extend_avoiding_wrong_haplo(working_graph, start_node, hap, config):
    
    """

    Perform greedy extension from a node, stopping if wrong haplotype nodes are encountered.
    
    Args:
        working_graph: NetworkX graph
        start_node: Node to start extension from
        hap: Target haplotype ('m' or 'p')
        config: Configuration dictionary
    
    Returns:
        List of nodes representing the extended path

    """
    path = [start_node]
    visited = {start_node, start_node ^ 1}
    current_node = start_node
    
    # Get attributes for efficiency
    node_yak_scores = nx.get_node_attributes(working_graph, f'yak_{hap}')
    
    while True:
        # Get all possible next edges
        out_edges = list(working_graph.out_edges(current_node, data=True))
        if not out_edges:
            break
        
        # Filter out edges leading to visited nodes or their complements
        # Also filter out edges leading to wrong haplotype nodes
        valid_edges = []
        for src, dst, data in out_edges:
            if dst not in visited:
                # Check if destination node is wrong haplotype
                dst_yak = node_yak_scores[dst]
                if dst_yak != -1:  # Not wrong haplotype (correct or ambiguous)
                    valid_edges.append((src, dst, data))
        
        if not valid_edges:
            # No valid edges that don't lead to wrong haplotype nodes
            break
        
        # Sort by score and take the best
        best_edge = max(valid_edges, key=lambda x: x[2]['score'])
        _, next_node, _ = best_edge
        
        path.append(next_node)
        visited.add(next_node)
        visited.add(next_node ^ 1)
        
        # Add transitive nodes to visited set
        transitive_nodes = set(working_graph.successors(current_node)) & set(working_graph.predecessors(next_node))
        visited.update(transitive_nodes)
        visited.update({node ^ 1 for node in transitive_nodes})
        
        current_node = next_node
    
    return path

def bubble_dijkstra(working_graph, walk, hap, config):
    """
    Replace segments of wrong haplotype nodes in a walk with Dijkstra-optimized paths.
    
    For each contiguous segment of wrong haplotype nodes (yak=-1), finds the closest
    non-wrong node (C or N) before and after, then replaces with Dijkstra path if found.
    
    Node types: yak=1 (Correct), yak=0 (Neutral), yak=-1 (Wrong)
    
    Args:
        working_graph: NetworkX graph containing the nodes
        walk: List of nodes representing the walk
        hap: Target haplotype ('m' or 'p')
        config: Configuration dictionary
    
    Returns:
        List of nodes representing the corrected walk
    """
    if not walk or len(walk) < 3:
        return walk
    
    # Determine which YAK attribute to check
    yak_attr = 'yak_p' if hap == 'm' else 'yak_m'
    
    corrected_walk = walk.copy()
    total_attempts = 0
    total_successes = 0
    
    print(f"Starting bubble_dijkstra correction for haplotype {hap}")
    print(f"Original walk has {len(walk)} nodes")
    
    # Find all wrong haplotype segments in one pass
    wrong_segments = []
    i = 0
    while i < len(corrected_walk):
        node = corrected_walk[i]
        yak_value = working_graph.nodes[node][yak_attr]
        
        if yak_value == -1:  # Found start of wrong segment
            segment_start = i
            # Find end of wrong segment
            while (i < len(corrected_walk) and 
                   working_graph.nodes[corrected_walk[i]][yak_attr] == -1):
                i += 1
            segment_end = i - 1  # Last wrong node
            wrong_segments.append((segment_start, segment_end))
        else:
            i += 1
    
    print(f"Found {len(wrong_segments)} wrong haplotype segments")
    
    # Process each wrong segment (in reverse order to maintain indices)
    for segment_start, segment_end in reversed(wrong_segments):
        segment_length = segment_end - segment_start + 1
        print(f"Processing wrong segment: {segment_length} nodes (indices {segment_start}-{segment_end})")
        
        # Find closest non-wrong node before the segment
        start_idx = None
        for j in range(segment_start - 1, -1, -1):
            node_yak = working_graph.nodes[corrected_walk[j]][yak_attr]
            if node_yak != -1:  # Correct (1) or Neutral (0)
                start_idx = j
                break
        
        # Find closest non-wrong node after the segment
        end_idx = None
        for j in range(segment_end + 1, len(corrected_walk)):
            node_yak = working_graph.nodes[corrected_walk[j]][yak_attr]
            if node_yak != -1:  # Correct (1) or Neutral (0)
                end_idx = j
                break
        
        if start_idx is None or end_idx is None:
            print(f"  No boundary nodes found")
            continue
        
        source_node = corrected_walk[start_idx]
        target_node = corrected_walk[end_idx]
        
        print(f"  Attempting Dijkstra from node {source_node} to {target_node}")
        total_attempts += 1
        
        try:
            dijkstra_path, dijkstra_cost = dijkstra(working_graph, source_node, target_node, hap)
            
            if dijkstra_path is not None and len(dijkstra_path) >= 2:
                # Replace the segment with Dijkstra path
                new_walk = (corrected_walk[:start_idx] + 
                          dijkstra_path + 
                          corrected_walk[end_idx + 1:])
                
                corrected_walk = new_walk
                total_successes += 1
                
                old_segment_length = end_idx - start_idx + 1
                print(f"  SUCCESS: Replaced segment")
                print(f"    Length: {old_segment_length} -> {len(dijkstra_path)} nodes")
            else:
                print(f"  FAILED: No valid Dijkstra path found")
                
        except Exception as e:
            print(f"  ERROR: Dijkstra failed: {e}")
    
    print(f"Bubble Dijkstra completed: {total_successes}/{len(wrong_segments)} segments improved")
    print(f"Total attempts: {total_attempts}")
    print(f"Final walk: {len(corrected_walk)} nodes")
    
    return corrected_walk