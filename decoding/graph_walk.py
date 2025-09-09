from tqdm import tqdm
import random
import networkx as nx
import numpy as np

def compute_beam_edge_score(nx_graph, src, dst, data, hap=None,
                           beam_score_shift=0.5, edge_penalty=100, 
                           kmer_penalty_factor=1, graphic_preds=None, use_hifiasm_result=False,
                           wrong_kmer_fraction=None,
                           alpha=0.00001, beta=10, gamma=10):
    delta = 0.3
    #delta = 0.05
    #alpha=0.00001, beta=1, gamma=25 and no log on node length
    thr_graphic = 0.1 #threshold for graphic prediction
    #print(data['score'])
    #print(data['to_cut'])
    #print()
    #if graphic_preds is not None:
    #    return compute_beam_with_graphic_preds(nx_graph, src, dst, data, hap, graphic_preds)
    """
    Compute edge score using an alternative beam search heuristic with logarithmic scaling.
    """
    
    # Calculate node length (sequence length added to assembly)
    node_length = nx_graph.nodes[dst]['read_length'] - data['overlap_length']
    read_add_length = node_length / nx_graph.nodes[dst]['read_length']

    # Calculate wrong kmer count for the target haplotype
    if graphic_preds is None:
        not_hap = 'm' if hap == 'p' else 'p'
        """yak_score = nx_graph.nodes[dst][f'yak_{not_hap}']
        wrong_kmer_count = 1 if yak_score == 1 else 0"""
        wrong_kmer_count = nx_graph.nodes[dst][f'kmer_count_{not_hap}']
        correct_kmer_count = nx_graph.nodes[dst][f'kmer_count_{hap}']
        wrong_hap_penalty = beta * wrong_kmer_count * read_add_length
    else:
        # Calculate wrong kmer count for the target haplotype
        
        if graphic_preds[dst] < -thr_graphic:
            wrong_hap_penalty =  - graphic_preds[dst] * beta
        else:
            wrong_hap_penalty = 0
    
    # Get edge score (bounded in [0, 1])
    edge_score = data['score']
    
    # Apply the formula: α⋅log(L(n)+1) − β⋅log(W(n)+1) + γ⋅S(e)
    """length_reward = alpha * np.log(node_length)
    wrong_kmer_penalty = beta * np.log(wrong_kmer_count*1000 * read_add_length+ 1)
    edge_score_component = gamma * (1 - edge_score) **2"""

    length_reward = alpha * node_length
    
    #correct_kmer_reward = delta * correct_kmer_count * read_add_length #and beta=3
    #correct_kmer_reward = delta * (correct_kmer_count*1000 * read_add_length)**2
    edge_score_component = (gamma * (1 - edge_score)) **2

    if edge_score<0.1:
        edge_score_component = 1000000
    #if wrong_kmer_count*1000*read_add_length > 10000:
    #    wrong_kmer_penalty = 1000000
    #print(wrong_kmer_penalty)
    #print(wrong_kmer_count)
    #print(length_reward, wrong_kmer_penalty, edge_score_component)

    beam_score = length_reward - wrong_hap_penalty - edge_score_component # + correct_kmer_reward
    
    return beam_score

def compute_graph_wrong_kmer_fraction(nx_graph, hap):
    """
    Compute the wrong kmer fraction for the entire graph.
    
    This function iterates over all nodes in the graph and calculates:
    1. Total wrong kmer counts across all nodes
    2. Total node lengths across all nodes
    3. Wrong kmer fraction (wrong_kmers_per_base)
    
    Args:
        nx_graph: NetworkX graph
        hap: Target haplotype ('m' or 'p')
    
    Returns:
        float: Wrong kmer fraction (wrong_kmers_per_base)
    """
    not_hap = 'm' if hap == 'p' else 'p'
    
    total_wrong_kmers = 0
    total_node_lengths = 0
    
    for node in nx_graph.nodes():
        wrong_kmer_count = nx_graph.nodes[node][f'kmer_count_{not_hap}']
        read_length = nx_graph.nodes[node]['read_length']
        
        total_wrong_kmers += wrong_kmer_count
        total_node_lengths += read_length
    
    # Calculate wrong kmer fraction (wrong kmers per base)
    wrong_kmer_fraction = total_wrong_kmers / total_node_lengths if total_node_lengths > 0 else 0
    
    return wrong_kmer_fraction

def sample_edges(nx_graph, hap, sample_size, sampling_by_score=False, score_attr='score', largest_component_only=False):
    """Sample edges from the graph, either randomly or weighted by score.
    
    Args:
        nx_graph: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
        sample_size: Number of edges to sample
        sampling_by_score: If True, sample proportional to edge score; if False, sample uniformly at random
        score_attr: Edge attribute to use for scoring when sampling_by_score=True
        largest_component_only: If True, only sample from largest connected component; if False, sample from full graph
    
    Returns:
        list: Sampled edges
    """
    # Get all edges
    all_edges = list(nx_graph.edges())
    
    # Filter edges based on haplotype
    if hap is not None:
        filtered_edges = []
        for src, dst in all_edges:
            if nx_graph.nodes[src][f'yak_{hap}'] != -1 and nx_graph.nodes[dst][f'yak_{hap}'] != -1:
                    filtered_edges.append((src, dst))

    else:
        filtered_edges = all_edges
    
    # If no edges after filtering, return empty list
    if not filtered_edges:
        return []
    
    # Determine which edges to sample from
    if largest_component_only:
        # Find connected components and identify the largest one
        print("Analyzing graph components for edge sampling...")
        
        # Use appropriate component finding method based on graph type
        components = list(nx.weakly_connected_components(nx_graph))
        
        print(f"Found {len(components)} connected components")
        
        if len(components) == 0:
            return []
        
        # Find the largest component by accumulated read length
        def component_read_length(component):
            return sum(nx_graph.nodes[node]['read_length'] for node in component)
        largest_component = max(components, key=component_read_length)
        largest_component_size = len(largest_component)
        
        # Print component statistics
        component_sizes = [len(comp) for comp in components]
        print(f"Component sizes: min={min(component_sizes)}, max={max(component_sizes)}, average={sum(component_sizes)/len(component_sizes):.1f}")
        print(f"Largest component has {largest_component_size} nodes ({largest_component_size/nx_graph.number_of_nodes()*100:.1f}% of total)")
        
        # Filter edges to only include those within the largest component
        component_edges = []
        for src, dst in filtered_edges:
            if src in largest_component and dst in largest_component:
                component_edges.append((src, dst))
        
        print(f"Filtered edges: {len(filtered_edges)} -> {len(component_edges)} (kept {len(component_edges)/len(filtered_edges)*100:.1f}% within largest component)")
        
        # If no edges in largest component, return empty list
        if not component_edges:
            return []
        
        edges_to_sample = component_edges
    else:
        # Use all filtered edges (no component filtering)
        print(f"Sampling from full graph: {len(filtered_edges)} edges available")
        edges_to_sample = filtered_edges
    
    # Sample from selected edges
    if sampling_by_score:

        # Score-based sampling
        # Get scores for each edge, filtering out edges with score 0
        edge_scores = []
        valid_edges = []
        for src, dst in edges_to_sample:
            score = nx_graph[src][dst].get(score_attr, 0)
            if score > 0:  # Only include edges with positive scores
                edge_scores.append(score)
                valid_edges.append((src, dst))
        
        # If no valid edges with positive scores, return empty list
        if not valid_edges:
            return []
        
        # Sample edges proportional to their scores
        if len(valid_edges) <= sample_size:
            sampled_edges = valid_edges
        else:
            # Normalize scores to probabilities
            total_score = sum(edge_scores)
            probabilities = [score/total_score for score in edge_scores]
            
            # Sample without replacement
            indices = random.choices(range(len(valid_edges)), weights=probabilities, k=sample_size)
            sampled_edges = [valid_edges[i] for i in indices]
    else:
        # Uniform random sampling
        if len(edges_to_sample) <= sample_size:
            sampled_edges = edges_to_sample
        else:
            sampled_edges = random.sample(edges_to_sample, sample_size)

    return sampled_edges


def merge_beams(beam1, beam2, common_node, nx_graph, score_attr='score', beam_score_shift=0.5, edge_penalty=100, kmer_penalty_factor=1, hap=None, graphic_preds=None, use_hifiasm_result=False, wrong_kmer_fraction=None):
    
    """
    Merge two beams that share a common node.
    
    Args:
        beam1: First beam tuple (path, visited_nodes, total_score, path_length, edge_count)
        beam2: Second beam tuple (path, visited_nodes, total_score, path_length, edge_count)
        common_node: The node that appears in both beams
        nx_graph: NetworkX graph
        score_attr: Edge attribute to use for scoring
        beam_score_shift: Score shift parameter
        edge_penalty: Penalty for edges with low scores (likely skips or malicious)
        kmer_penalty_factor: Factor for kmer penalty
        hap: Target haplotype ('m' or 'p' or None)
        graphic_preds: Optional dictionary of graphic predictions for nodes
        wrong_kmer_fraction: Precomputed wrong kmer fraction for the graph (optional)
    
    Returns:
        tuple: Merged beam or None if beam should be deleted
    """

    path1, visited1, total_score1, path_length1, edge_count1 = beam1
    path2, visited2, total_score2, path_length2, edge_count2 = beam2
    
    # Find position of common_node in both paths
    node_pos1 = path1.index(common_node) if common_node in path1 else -1
    node_pos2 = path2.index(common_node) if common_node in path2 else -1
    
    # If common_node is at the end of both beams, keep the better one
    if node_pos1 == len(path1) - 1 and node_pos2 == len(path2) - 1:
        # Both beams end at the same node, keep the one with better total score
        if total_score1 >= total_score2:
            return beam1
        else:
            return beam2
    
    # If common_node is at the end of beam1 but not beam2
    if node_pos1 == len(path1) - 1 and node_pos2 < len(path2) - 1:
        # Calculate score for beam2 up to the common node
        beam2_score_to_node = 0
        beam2_path_length_to_node = 0
        
        for i in range(node_pos2):
            if i > 0:
                src = path2[i-1]
                dst = path2[i]
                edge_data = nx_graph[src][dst]
                
                # Use consistent scoring function
                edge_score = compute_beam_edge_score(nx_graph, src, dst, edge_data, hap=hap, 
                                                   beam_score_shift=beam_score_shift, 
                                                   edge_penalty=edge_penalty, 
                                                   kmer_penalty_factor=kmer_penalty_factor, 
                                                   graphic_preds=graphic_preds,
                                                   use_hifiasm_result=use_hifiasm_result,
                                                   wrong_kmer_fraction=wrong_kmer_fraction)
                
                beam2_score_to_node += edge_score
                beam2_path_length_to_node += edge_data['prefix_length']
        
        # Compare beam1's total score with beam2's total score up to the common node
        if beam2_score_to_node > total_score1:
            # Beam2 is better up to common node, delete beam1
            return None
        else:
            # Merge: take beam1 up to common node, then continue with beam2 after common node
            merged_path = path1.copy()
            if node_pos2 < len(path2) - 1:
                merged_path.extend(path2[node_pos2 + 1:])
            
            # Calculate merged metrics
            merged_visited = visited1.copy()
            merged_visited.update(visited2)
            
            # Calculate total score for merged path
            merged_total_score = total_score1
            merged_path_length = path_length1
            merged_edge_count = edge_count1
            
            # Add scores from beam2 after the common node
            for i in range(node_pos2 + 1, len(path2)):
                if i > node_pos2 + 1 or len(path1) > 1:  # There's an edge to score
                    src_idx = len(path1) - 1 + (i - node_pos2 - 1) if i > node_pos2 + 1 else len(path1) - 1
                    if src_idx < len(merged_path) - 1:
                        src = merged_path[src_idx]
                        dst = merged_path[src_idx + 1]
                        edge_data = nx_graph[src][dst]
                        
                        # Use consistent scoring function
                        edge_score = compute_beam_edge_score(nx_graph, src, dst, edge_data, hap=hap, 
                                                   beam_score_shift=beam_score_shift, 
                                                   edge_penalty=edge_penalty, 
                                                   kmer_penalty_factor=kmer_penalty_factor, 
                                                   graphic_preds=graphic_preds,
                                                   use_hifiasm_result=use_hifiasm_result,
                                                   wrong_kmer_fraction=wrong_kmer_fraction)
                        
                        merged_total_score += edge_score
                        merged_path_length += edge_data['prefix_length']
                        merged_edge_count += 1
            
            return (merged_path, merged_visited, merged_total_score, merged_path_length, merged_edge_count)
    
    # If common_node is at the end of beam2 but not beam1
    if node_pos2 == len(path2) - 1 and node_pos1 < len(path1) - 1:
        # Calculate score for beam1 up to the common node
        beam1_score_to_node = 0
        beam1_path_length_to_node = 0
        
        for i in range(node_pos1):
            if i > 0:
                src = path1[i-1]
                dst = path1[i]
                edge_data = nx_graph[src][dst]
                
                # Use consistent scoring function
                edge_score = compute_beam_edge_score(nx_graph, src, dst, edge_data, hap=hap, 
                                                   beam_score_shift=beam_score_shift, 
                                                   edge_penalty=edge_penalty, 
                                                   kmer_penalty_factor=kmer_penalty_factor, 
                                                   graphic_preds=graphic_preds,
                                                   use_hifiasm_result=use_hifiasm_result,
                                                   wrong_kmer_fraction=wrong_kmer_fraction)
                
                beam1_score_to_node += edge_score
                beam1_path_length_to_node += edge_data['prefix_length']
        
        # Compare beam2's total score with beam1's total score up to the common node
        if beam1_score_to_node > total_score2:
            # Beam1 is better up to common node, delete beam2
            return None
        else:
            # Merge: take beam2 up to common node, then continue with beam1 after common node
            merged_path = path2.copy()
            if node_pos1 < len(path1) - 1:
                merged_path.extend(path1[node_pos1 + 1:])
            
            # Calculate merged metrics
            merged_visited = visited2.copy()
            merged_visited.update(visited1)
            
            # Calculate total score for merged path
            merged_total_score = total_score2
            merged_path_length = path_length2
            merged_edge_count = edge_count2
            
            # Add scores from beam1 after the common node
            for i in range(node_pos1 + 1, len(path1)):
                if i > node_pos1 + 1 or len(path2) > 1:  # There's an edge to score
                    src_idx = len(path2) - 1 + (i - node_pos1 - 1) if i > node_pos1 + 1 else len(path2) - 1
                    if src_idx < len(merged_path) - 1:
                        src = merged_path[src_idx]
                        dst = merged_path[src_idx + 1]
                        edge_data = nx_graph[src][dst]
                        
                        # Use consistent scoring function
                        edge_score = compute_beam_edge_score(nx_graph, src, dst, edge_data, hap=hap, 
                                                   beam_score_shift=beam_score_shift, 
                                                   edge_penalty=edge_penalty, 
                                                   kmer_penalty_factor=kmer_penalty_factor, 
                                                   graphic_preds=graphic_preds,
                                                   use_hifiasm_result=use_hifiasm_result,
                                                   wrong_kmer_fraction=wrong_kmer_fraction)
                        
                        merged_total_score += edge_score
                        merged_path_length += edge_data['prefix_length']
                        merged_edge_count += 1
            
            return (merged_path, merged_visited, merged_total_score, merged_path_length, merged_edge_count)
    
    # If common_node is in the middle of both paths, keep the better beam
    # This is a more complex case that might require more sophisticated merging logic
    if total_score1 >= total_score2:
        return beam1
    else:
        return beam2


def walk_beamsearch(nx_graph, start_node, hap=None, score_attr='score', beam_width=3, beam_score_shift=0.5, edge_penalty=100, kmer_penalty_factor=1, use_best_intermediate=True, visited=None, graphic_preds=None, use_hifiasm_result=False):
    """Find a path through the graph using beam search from a given start node.
    
    Args:
        nx_graph: NetworkX graph
        start_node: Node to start path from
        hap: Target haplotype ('m' or 'p' or None)
        score_attr: Edge attribute to use for scoring
        beam_width: Number of candidate paths to maintain at each step (default: 3)
        beam_score_shift: Score shift parameter
        edge_penalty: Penalty for edges with low scores (likely skips or malicious)
        kmer_penalty_factor: Factor for kmer penalty
        use_best_intermediate: Whether to use best intermediate walk if better than final (default: True)
        visited: Set of already visited nodes to avoid conflicts (optional)
        graphic_preds: Optional dictionary of graphic predictions for nodes
    
    Returns:
        tuple: (path, path_length, visited_set) where path is list of nodes, 
               path_length is sum of prefix_lengths, and visited_set is the set of visited nodes
    """
    if len(nx_graph) == 0 or start_node not in nx_graph:
        return [], 0, set()
    
    # Compute wrong kmer fraction for the entire graph if haplotype is specified
    wrong_kmer_fraction = None
    if hap is not None:
        wrong_kmer_fraction = compute_graph_wrong_kmer_fraction(nx_graph, hap)
    
    # Initialize visited set
    if visited is None:
        initial_visited = {start_node, start_node ^ 1}
    else:
        initial_visited = visited.copy()
        initial_visited.add(start_node)
        initial_visited.add(start_node ^ 1)
    
    # Initialize beam with the start node
    # Each beam item contains: (path, visited_nodes, total_score, path_length, edge_count)
    beam = [(
        [start_node],                      # path
        initial_visited,                   # visited nodes (including complements)
        0,                                 # total score
        0,                                 # path length
        0                                  # edge count (for averaging)
    )]
    
    # Track the best intermediate walk during beam search (if enabled)
    best_intermediate = None
    best_intermediate_score = float('-inf')
    
    # Keep expanding until all paths in the beam can't be extended further
    iteration = 0
    
    while beam:  # Add iteration limit to prevent infinite loops
        iteration += 1
        next_beam = []
        
        # Track the best path in the current beam for intermediate comparison (if enabled)
        if use_best_intermediate:
            current_best = max(beam, key=lambda x: x[2])  # Sort by total score
            if current_best[2] > best_intermediate_score:
                best_intermediate = current_best
                best_intermediate_score = current_best[2]
        
        # Process each path in the current beam
        for path, visited_nodes, total_score, path_length, edge_count in beam:
            if len(path) == 0:
                continue
                
            current_node = path[-1]
            
            # Get all possible next edges
            out_edges = list(nx_graph.out_edges(current_node, data=True))
            if not out_edges:
                # If this path can't be extended, keep it as a candidate in the next beam
                next_beam.append((path, visited_nodes, total_score, path_length, edge_count))
                continue
            
            # Filter out edges leading to visited nodes or their complements
            valid_edges = []
            for src, dst, data in out_edges:
                if dst not in visited_nodes:
                    valid_edges.append((src, dst, data))
            
            if not valid_edges:
                # If no valid edges, keep this path as a candidate
                next_beam.append((path, visited_nodes, total_score, path_length, edge_count))
                continue
            
            # Score all valid edges
            scored_edges = []
            for src, dst, data in valid_edges:
                edge_score = compute_beam_edge_score(nx_graph, src, dst, data, hap=hap,
                                                   beam_score_shift=beam_score_shift,
                                                   edge_penalty=edge_penalty,
                                                   kmer_penalty_factor=kmer_penalty_factor,
                                                   graphic_preds=graphic_preds,
                                                   use_hifiasm_result=use_hifiasm_result,
                                                   wrong_kmer_fraction=wrong_kmer_fraction)
                scored_edges.append((src, dst, data, edge_score))

            # Sort edges by score (descending)
            scored_edges.sort(key=lambda x: x[3], reverse=True)
            
            # Take top edges to expand this path
            for i, (src, dst, data, edge_score) in enumerate(scored_edges):
                if i >= beam_width:
                    break
                
                # Create new visited set
                new_visited = visited_nodes.copy()
                new_visited.add(dst)
                new_visited.add(dst ^ 1)  # Also mark complement as visited
                
                # Get nodes that are both successors of current and predecessors of dst
                transitive_nodes = set(nx_graph.successors(current_node)) & set(nx_graph.predecessors(dst))
                # Add these nodes and their complements to visited
                new_visited.update(transitive_nodes)
                new_visited.update({node ^ 1 for node in transitive_nodes})
                
                # Update path length with this edge's prefix length
                new_path_length = path_length + data['prefix_length']
                
                # Create new path
                new_path = path.copy()
                new_path.append(dst)
                
                # Update edge count and scores
                new_edge_count = edge_count + 1
                new_total_score = total_score + edge_score
                
                # Add to candidates for next beam
                next_beam.append((new_path, new_visited, new_total_score, new_path_length, new_edge_count))
        
        # If no paths were extended, we're done
        if next_beam == beam:
            break
        
        # Check for duplicate last nodes and merge beams
        # Group beams by their last node
        last_node_groups = {}
        for beam_item in next_beam:
            last_node = beam_item[0][-1]  # Get last node from path
            if last_node not in last_node_groups:
                last_node_groups[last_node] = []
            last_node_groups[last_node].append(beam_item)
        
        # Merge beams that share the same last node
        merged_beam = []
        for last_node, beams_with_same_end in last_node_groups.items():
            if len(beams_with_same_end) == 1:
                # Only one beam ends with this node, keep it as is
                merged_beam.append(beams_with_same_end[0])
            else:
                # Multiple beams end with the same node, merge them
                # Start with the first beam as the base
                best_beam = beams_with_same_end[0]
                for i in range(1, len(beams_with_same_end)):
                    other_beam = beams_with_same_end[i]
                    merged_result = merge_beams(best_beam, other_beam, last_node, nx_graph, 
                                              score_attr, 
                                              beam_score_shift=beam_score_shift, edge_penalty=edge_penalty, 
                                              kmer_penalty_factor=kmer_penalty_factor, 
                                              graphic_preds=graphic_preds,
                                              use_hifiasm_result=use_hifiasm_result,
                                              wrong_kmer_fraction=wrong_kmer_fraction)
                    if merged_result is not None:
                        best_beam = merged_result
                    # If merged_result is None, it means one beam was deleted
                
                merged_beam.append(best_beam)
        
        # Select top beam_width candidates for next iteration
        # Sort by total score (descending) instead of average score
        merged_beam.sort(key=lambda x: x[2], reverse=True)
        beam = merged_beam[:beam_width]
    
    # If no valid paths found
    if not beam:
        return [], 0, set()
    
    # Select the best path from final beam (highest total score)
    final_best_path, final_best_visited, final_best_total_score, final_best_path_length, _ = max(beam, key=lambda x: x[2])
    
    # If use_best_intermediate is enabled, compare with best intermediate path
    if use_best_intermediate and best_intermediate is not None:
        if best_intermediate[2] > final_best_total_score:
            final_best_path, final_best_visited, final_best_total_score, final_best_path_length, _ = best_intermediate
    
    return final_best_path, final_best_path_length, final_best_visited


def get_walk(nx_graph, hap, config, random_mode=False, walk_mode=None, graphic_preds=None):
    """Find paths through the graph using either greedy, random, or beam search walks from sampled edges.
    
    Args:
        nx_graph: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary with parameters
        random_mode: If True, use random walks; if False, use greedy walks (deprecated, use walk_mode instead)
        walk_mode: Walking algorithm to use ('greedy', 'random', or 'beamsearch'). If None, falls back to random_mode.
        graphic_preds: Optional dictionary of graphic predictions for nodes
    
    Returns:
        list: Best path found
    """
    # Determine the walk type based on parameters
    if walk_mode is None:
        # Fall back to random_mode for backward compatibility
        walk_mode = "random" if random_mode else "greedy"

    print(f"=== GET_WALK DEBUG ===")
    print(f"Input parameters: hap={hap}, random_mode={random_mode}, walk_mode={walk_mode}")
    print(f"Final walk_mode determined: {walk_mode}")

    # Compute wrong kmer fraction once at the beginning if haplotype is specified and using beam search
    wrong_kmer_fraction = None
    if hap is not None and (walk_mode in ["beam_search", "beamsearch", "beam_search_no_wh"]):
        wrong_kmer_fraction = compute_graph_wrong_kmer_fraction(nx_graph, hap)
        print(f"Computed wrong kmer fraction for {hap}: {wrong_kmer_fraction:.2e}")

    walk_type = walk_mode
    sampled_edges = sample_edges(nx_graph, hap, config['sample_size'], sampling_by_score=config['sample_by_score'])
    print(f"\nSampled {len(sampled_edges)} edges to create {len(sampled_edges)} different {walk_type} walks...")
    if len(sampled_edges) == 0:
        return [], 0
    
    best_length = 0
    best_walk = []
    best_penalized_length = 0
    src = sampled_edges[0][0] ## example node

    #reversed_graph = nx_graph.reverse()
    #reversed = True
    if src^1 not in nx_graph.nodes():
        reversed_graph = nx_graph.reverse()
        reversed = True
    else:
        reversed=False
        reversed_graph = nx_graph
    init_visited = {src, src^1}  # Fix syntax error - use set literal instead of set(src, src^1)

    # Create different walks using each sampled edge as a starting point
    for walk_idx, (src, dst) in enumerate(tqdm(sampled_edges, desc=f"Trying {walk_type} walks")):
        
        # Only print debug info for first few walks to avoid spam
        debug_this_walk = False #(walk_idx < 3)
        
        if debug_this_walk:
            print(f"\n--- Walk {walk_idx+1}/{len(sampled_edges)} ---")
            print(f"Current walk_mode: '{walk_mode}'")
            print(f"Sampled edge: {src} -> {dst}")
        
        # Try path from source node
        if debug_this_walk:
            print("✓ EXECUTING beam_search forward path")
        forward_path, forward_length, forward_visited = walk_beamsearch(
            nx_graph, dst, hap=hap,
            score_attr='score',
            beam_width=config['beam_width'],
            beam_score_shift=config['beam_score_shift'],
            edge_penalty=config['edge_penalty'],
            kmer_penalty_factor=config['kmer_penalty_factor'],
            use_best_intermediate=config['beam_intermediate'],
            visited=init_visited,
            graphic_preds=graphic_preds,
            use_hifiasm_result=config.get('use_hifiasm_result', False)
        )
        forward_wrong_haplo_len = 0  # Beam search doesn't track wrong haplotype length
        if debug_this_walk:
            print(f"Forward path length: {len(forward_path)}, Forward visited: {len(forward_visited)}")
    
        if not reversed:
            src = src^1
            
        # Try path from target node on reversed graph
        if debug_this_walk:
            print(f"✓ EXECUTING beam_search backward from {src}, avoiding {len(forward_visited)} nodes")
        backward_path, backward_length, backward_visited = walk_beamsearch(
            reversed_graph, src, hap=hap,
            score_attr='score',
            beam_width=config['beam_width'],
            beam_score_shift=config['beam_score_shift'],
            edge_penalty=config['edge_penalty'],
            kmer_penalty_factor=config['kmer_penalty_factor'],
            use_best_intermediate=config['beam_intermediate'],
            visited=forward_visited,
            graphic_preds=graphic_preds,
            use_hifiasm_result=config.get('use_hifiasm_result', False)
        )
        backward_wrong_haplo_len = 0  # Beam search doesn't track wrong haplotype length
        if debug_this_walk:
            print(f"Backward path length: {len(backward_path)}")

        # For each target node's path, reverse it and try to combine with source path
        if backward_path:
            if reversed:
                n_backward_path = [n for n in backward_path[::-1]]  # Reverse the path and complement nodes
            else:
                n_backward_path = [n^1 for n in backward_path[::-1]]  # Reverse the path and complement nodes
            
            if debug_this_walk:
                # Debug: Check junction point
                print(f"Forward path starts with: {forward_path[:3] if len(forward_path) >= 3 else forward_path}")
                print(f"Forward path ends with: {forward_path[-3:] if len(forward_path) >= 3 else forward_path}")
                print(f"Backward path (original) starts with: {backward_path[:3] if len(backward_path) >= 3 else backward_path}")
                print(f"Backward path (original) ends with: {backward_path[-3:] if len(backward_path) >= 3 else backward_path}")
                print(f"Backward path (processed) starts with: {n_backward_path[:3] if len(n_backward_path) >= 3 else n_backward_path}")
                print(f"Backward path (processed) ends with: {n_backward_path[-3:] if len(n_backward_path) >= 3 else n_backward_path}")
                
            combined_path = n_backward_path + forward_path
                        
            combined_wrong_haplo_len = forward_wrong_haplo_len + backward_wrong_haplo_len
        else:
            print("❌ No backward path found!")
            exit()
            combined_path = forward_path
            combined_wrong_haplo_len = forward_wrong_haplo_len

        # Compute alternative path length based on prefix lengths and final node length
        path_length = 0
        for i in range(len(combined_path)-1):
            src = combined_path[i]
            dst = combined_path[i+1]
            path_length += nx_graph[src][dst]['prefix_length']
        path_length += nx_graph.nodes[combined_path[-1]]['read_length']
        #print(f"Full path length: {path_length}")

        penalized_path_length = path_length
        
        # Update best path if this one is longer
        if penalized_path_length > best_penalized_length:
            best_walk = combined_path
            best_length = path_length
            best_penalized_length = penalized_path_length

    print(f"Best {walk_type} path found: {len(best_walk)} nodes, length={best_length:,}bp")

    return best_walk, best_length
