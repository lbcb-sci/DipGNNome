from tqdm import tqdm
import random
import networkx as nx
import numpy as np

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

def compute_beam_with_graphic_preds(nx_graph, src, dst, data, hap, graphic_preds):
    node_length = nx_graph.nodes[dst]['read_length'] - data['overlap_length']
    read_add_length = node_length / nx_graph.nodes[dst]['read_length']

    # Get all edge types and their counts
    
    alpha = 0.00001
    beta = 0
    gamma = 10  #penalty for 0 edges
    thr = 0.1 #threshold for graphic prediction
    edge_score = data['score']
    length_reward = alpha * node_length
    #print(graphic_preds[dst])
    if graphic_preds[dst] < -thr:
        graphic =  beta #graphic_preds[dst] * beta
    else:
        graphic = 0
    #graphic = graphic_preds[dst] * beta
    edge_score_component = gamma * edge_score**2
    beam_score = length_reward - graphic - edge_score_component
    #print(length_reward, graphic, edge_score_component)
    return beam_score

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

def compute_beam_edge_score_(nx_graph, src, dst, data, hap=None,
                           beam_score_shift=0.5, edge_penalty=100, 
                           kmer_penalty_factor=1, graphic_preds=None, use_hifiasm_result=False,
                           wrong_kmer_fraction=None,
                           alpha=0.00001, beta=3, gamma=10):
    
    epsilon = 25 #for chrom changes
    delta = 0.3
    thr_graphic = 0.1 #threshold for graphic prediction
    #alpha=0.00001, beta=1, gamma=25 and no log on node length

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

    if graphic_preds is None:
        not_hap = 'm' if hap == 'p' else 'p'
        #yak_score = nx_graph.nodes[dst][f'yak_{not_hap}']
        #wrong_kmer_count = 1 if yak_score == 1 else 0
        wrong_kmer_count = nx_graph.nodes[dst][f'kmer_count_{not_hap}']
        correct_kmer_count = nx_graph.nodes[dst][f'kmer_count_{hap}']
        wrong_hap_penalty = beta * wrong_kmer_count * read_add_length
    else:
        # Calculate wrong kmer count for the target haplotype
        
        if graphic_preds[dst] < -thr_graphic:
            wrong_hap_penalty =  -graphic_preds[dst] * beta
        else:
            wrong_hap_penalty = 0

    
    # Get edge score (bounded in [0, 1])
    edge_score = data['score']
    
    # Apply the formula: α⋅log(L(n)+1) − β⋅log(W(n)+1) + γ⋅S(e)
    """length_reward = alpha * np.log(node_length)
    wrong_kmer_penalty = beta * np.log(wrong_kmer_count*1000 * read_add_length+ 1)
    edge_score_component = gamma * (1 - edge_score) **2"""

    length_reward = alpha * node_length

    """if correct_kmer_count > 10 * wrong_kmer_count:
        wrong_kmer_penalty = 0
    else:
        """
    

    #correct_kmer_reward = delta * correct_kmer_count * read_add_length #and beta=3

    #correct_kmer_reward = delta * (correct_kmer_count*1000 * read_add_length)**2
    edge_score_component = (gamma * edge_score) **2
    cut_score_component = (epsilon * data['to_cut']) **2

    #print(f"Edge score: {edge_score}, To cut: {data['to_cut']}, Edge score component: {edge_score_component}, Cut score component: {cut_score_component}")

    if edge_score > 0.9:
        edge_score_component = 10000000
    
    if data['to_cut'] > 0.9:
        cut_score_component = 100000000
        
    #if wrong_kmer_count*1000*read_add_length > 10000:
    #    wrong_kmer_penalty = 1000000
    #print(wrong_kmer_penalty)
    #print(wrong_kmer_count)
    #print(length_reward, wrong_kmer_penalty, edge_score_component)

    beam_score = length_reward - wrong_hap_penalty - edge_score_component - cut_score_component #+ correct_kmer_reward
    
    return beam_score

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

def walk_graph(nx_graph, start_node, hap=None, max_wrong_haplo=5, score_attr='score', random_mode=False, penalty_wrong_hap=0.6, penalty_ambiguous=0.05):
    """Find a path through the graph by following edges from a given start node.
    
    Args:
        nx_graph: NetworkX graph
        start_node: Node to start path from
        hap: Target haplotype ('m' or 'p' or None)
        max_wrong_haplo: Maximum number of consecutive wrong haplotype nodes allowed
        score_attr: Edge attribute to use for scoring
        random_mode: If True, randomly select edges; if False, greedily select highest-scoring edges
        penalty_wrong_hap: Penalty to apply to edges leading to wrong haplotype nodes
        penalty_ambiguous: Penalty to apply to edges leading to ambiguous nodes
    
    Returns:
        tuple: (path, path_length, wrong_haplo_len, total_wrong_kmer_hits, visited_set) where path is list of nodes, 
               path_length is sum of prefix_lengths, and visited_set is the set of visited nodes
    """
    if hap is None:
        other_hap = None
        wrong_kmer_hits = None
        total_wrong_kmer_hits = 0
    else:   
        other_hap = 'm' if hap == 'p' else 'p'
        #wrong_kmer_hits = nx.get_node_attributes(nx_graph, f"kmer_count_{other_hap}")
        total_wrong_kmer_hits = 0

    if len(nx_graph) == 0 or start_node not in nx_graph:
        if start_node not in nx_graph:
            print(f"Start node {start_node} not in graph")
            exit()
        return [], 0, 0, 0, set()
    
    path = [start_node]
    # Keep track of both visited nodes and their complements
    visited = {start_node, start_node ^ 1}  # Using XOR to get complement
    wrong_haplo_len = 0
    current_node = start_node
    
    while True:
        # Get all possible next edges
        out_edges = list(nx_graph.out_edges(current_node, data=True))
        if not out_edges:
            break
            
        # Filter out edges leading to visited nodes or their complements
        valid_edges = []
        for src, dst, data in out_edges:
            if dst not in visited:  # This checks both the node and its complement
                valid_edges.append((src, dst, data))
        
        if not valid_edges:
            break
        
        # Select next edge based on mode
        if random_mode:
            # Random mode: randomly select the next edge
            next_edge = random.choice(valid_edges)
            _, next_node, _ = next_edge
        else:
            # Greedy mode: sort edges by score, applying penalty for wrong haplotype nodes
            if hap is not None:
                # Apply penalty to edges leading to wrong haplotype nodes
                scored_edges = []
                for src, dst, data in valid_edges:
                    score = data['score']
                    # Check haplotype of destination node
                    #if nx_graph.nodes[dst][f'ambiguous'] == 1:  
                    #    score -= penalty_ambiguous
                    if hap is not None:
                        if nx_graph.nodes[dst][f'yak_{hap}'] < 0:  
                            #print("wrong haplotype")
                            #continue
                            score -= penalty_wrong_hap
                            #score -= wrong_kmer_hits[dst]/10
                    # Penalize edges that cross between haplotypes (M to P or P to M)
                    """if 'read_haplo' in nx_graph.nodes[src] and 'read_variant' in nx_graph.nodes[dst]:
                        src_haplo = nx_graph.nodes[src]['read_variant']
                        dst_haplo = nx_graph.nodes[dst]['read_variant']
                        if src_haplo in ['M', 'P'] and dst_haplo in ['M', 'P'] and src_haplo != dst_haplo:
                            score -= 0.2  # Penalty for crossing between haplotypes
                    """
                    scored_edges.append((src, dst, data, score))
                next_edges = sorted(scored_edges, key=lambda x: x[3], reverse=True)
            else:
                # No haplotype constraints, sort by raw score
                next_edges = sorted(valid_edges, key=lambda x: x[2][score_attr], reverse=True)

            if next_edges:
                if hap is not None:
                    _, next_node, _, _ = next_edges[0]
                else:
                    _, next_node, _ = next_edges[0]
            else:
                break
        
        # If using haplotype constraints
        if hap is not None:
            # Check haplotype of current node
            yak_score = nx_graph.nodes[next_node][f'yak_{hap}']
            if yak_score == -1:  # Wrong haplotype
                wrong_haplo_len += nx_graph.nodes[next_node]['read_length']
        
        path.append(next_node)
        visited.add(next_node)
        visited.add(next_node ^ 1)  # Add complement
        # Get nodes that are both successors of current and predecessors of next_node
        transitive_nodes = set(nx_graph.successors(current_node)) & set(nx_graph.predecessors(next_node))
        # Add these nodes and their complements to visited
        visited.update(transitive_nodes)
        visited.update({node ^ 1 for node in transitive_nodes})
        current_node = next_node
        #total_wrong_kmer_hits += wrong_kmer_hits[next_node]
    
    # Calculate path length
    path_length = 0
    for i in range(len(path)-1):
        edge_data = nx_graph[path[i]][path[i+1]]
        path_length += edge_data['prefix_length']
    
    # Add length of last node
    if path:
        path_length += nx_graph.nodes[path[-1]]['read_length']
    
    return path, path_length, wrong_haplo_len, total_wrong_kmer_hits, visited

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
    """if config.get('use_hifiasm_result', False):
        compute_malicious_edges_3(nx_graph)
        evaluate_edge_predictions(nx_graph)
        #exit()"""
    print(f"=== GET_WALK DEBUG ===")
    print(f"Input parameters: hap={hap}, random_mode={random_mode}, walk_mode={walk_mode}")
    print(f"Final walk_mode determined: {walk_mode}")
    
    """from collections import Counter
    yak_tester = nx.get_node_attributes(nx_graph, "yak_m")
    yak_values = list(yak_tester.values())
    value_counts = Counter(yak_values)
    print("\nDistribution of yak_m values:")
    for value, count in sorted(value_counts.items()):
        print(f"Value {value}: {count} nodes ({count/len(yak_values)*100:.2f}%)")
    print("killed after prins (need delete later)")
    exit()"""  

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
        # Compute malicious edges for this specific sampled edge if use_hifiasm_result is True
        #if config.get('use_hifiasm_result', False):
        #    compute_malicious_edges(nx_graph, (src, dst), hap)
        
        # Only print debug info for first few walks to avoid spam
        debug_this_walk = False #(walk_idx < 3)
        
        if debug_this_walk:
            print(f"\n--- Walk {walk_idx+1}/{len(sampled_edges)} ---")
            print(f"Current walk_mode: '{walk_mode}'")
            print(f"Sampled edge: {src} -> {dst}")
        
        # Try path from source node
        if walk_mode == "beam_search" or walk_mode == "beamsearch":
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
        elif walk_mode == "beam_search_no_wh":
            if debug_this_walk:
                print("✓ EXECUTING beam_search_no_wh forward path")
            forward_path, forward_length, forward_wrong_haplo_len, forward_visited = beam_search_no_wh(
                nx_graph, dst, hap=hap,
                score_attr='score',
                penalty_wrong_hap=config['penalty_wrong_hap'],
                penalty_ambiguous=config['penalty_ambiguous'],
                beam_width=config['beam_width'],
                beam_score_shift=config['beam_score_shift'],
                wrong_hap_threshold=config['wrong_hap_threshold'],
                use_best_intermediate=config['beam_intermediate'],
                visited=init_visited,
                wrong_kmer_fraction=wrong_kmer_fraction
            )
            if debug_this_walk:
                print(f"Forward path length: {len(forward_path)}, Forward visited: {len(forward_visited)}")
        else:
            if debug_this_walk:
                print(f"✓ EXECUTING walk_graph with random_mode={walk_mode == 'random'}")
            # Use original random/greedy mode
            random_walk_mode = (walk_mode == "random")
            forward_path, forward_length, forward_wrong_haplo_len, total_wrong_kmer_hits, forward_visited = walk_graph(
                nx_graph, dst, hap=hap, 
                max_wrong_haplo=config['max_wrong_haplo'],
                score_attr='score',
                random_mode=random_walk_mode,
                penalty_wrong_hap=config['penalty_wrong_hap'],
                penalty_ambiguous=config['penalty_ambiguous'],
            )
            if debug_this_walk:
                print(f"Forward path length: {len(forward_path)}, Forward visited: {len(forward_visited)}")
            
        if not reversed:
            src = src^1
            
        # Try path from target node on reversed graph
        if walk_mode == "beam_search" or walk_mode == "beamsearch":
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
        elif walk_mode == "beam_search_no_wh":
            if debug_this_walk:
                print(f"✓ EXECUTING beam_search_no_wh backward from {src}, avoiding {len(forward_visited)} nodes")
            backward_path, backward_length, backward_wrong_haplo_len, backward_visited = beam_search_no_wh(
                reversed_graph, src, hap=hap,
                score_attr='score',
                penalty_wrong_hap=config['penalty_wrong_hap'],
                penalty_ambiguous=config['penalty_ambiguous'],
                beam_width=config['beam_width'],
                beam_score_shift=config['beam_score_shift'],
                wrong_hap_threshold=config['wrong_hap_threshold'],
                use_best_intermediate=config['beam_intermediate'],
                visited=forward_visited,
                wrong_kmer_fraction=wrong_kmer_fraction
            )
            if debug_this_walk:
                print(f"Backward path length: {len(backward_path)}")
        else:
            if debug_this_walk:
                print(f"✓ EXECUTING walk_graph backward with random_mode={walk_mode == 'random'}")
            # Use original random/greedy mode
            random_walk_mode = (walk_mode == "random")
            backward_path, backward_length, backward_wrong_haplo_len, total_wrong_kmer_hits, backward_visited = walk_graph(
                reversed_graph, src, hap=hap, 
                max_wrong_haplo=config['max_wrong_haplo'],
                score_attr='score',
                random_mode=random_walk_mode,
                penalty_wrong_hap=config['penalty_wrong_hap'],
                penalty_ambiguous=config['penalty_ambiguous'],
            )
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
            
            """# Check for duplicates in the combined path
            seen_nodes = set()
            duplicates = []
            for i, node in enumerate(combined_path):
                if node in seen_nodes or (node ^ 1) in seen_nodes:
                    duplicates.append((i, node))
                seen_nodes.add(node)
            
            if duplicates:
                print(f"⚠️  WARNING: Found {len(duplicates)} duplicates in combined path:")
                for i, node in duplicates[:5]:  # Show first 5 duplicates
                    print(f"  Position {i}: node {node}")
                    
                # Check if duplicates are at the junction
                junction_point = len(n_backward_path)
                junction_duplicates = [d for d in duplicates if abs(d[0] - junction_point) <= 2]
                if junction_duplicates:
                    print(f"Junction duplicates found around position {junction_point}:")
                    for i, node in junction_duplicates:
                        print(f"  Position {i}: node {node}")
            else:
                print("✓ No duplicates found in combined path")"""
                        
            combined_wrong_haplo_len = forward_wrong_haplo_len + backward_wrong_haplo_len
        else:
            print("❌ No backward path found!")
            exit()
            combined_path = forward_path
            combined_wrong_haplo_len = forward_wrong_haplo_len

        """path_length = forward_length + backward_length
        if combined_path:
            path_length += nx_graph.nodes[combined_path[-1]]['read_length']
        
        print(f"Original path length: {path_length}")"""
        # Compute alternative path length based on prefix lengths and final node length
        path_length = 0
        for i in range(len(combined_path)-1):
            src = combined_path[i]
            dst = combined_path[i+1]
            path_length += nx_graph[src][dst]['prefix_length']
        path_length += nx_graph.nodes[combined_path[-1]]['read_length']
        #print(f"Full path length: {path_length}")

        if config['select_walk'] == 'haplo_penalty':
            penalized_path_length = path_length - combined_wrong_haplo_len * config['hap_length_penalty']
        elif config['select_walk'] == 'kmer_penalty':
            penalized_path_length = path_length - total_wrong_kmer_hits * 100000 #config['kmer_penalty']
        elif config['select_walk'] == 'longest_no_whs':
            combined_path, path_length = filter_wrong_haplotype_subpaths(nx_graph, combined_path, hap, config)
            penalized_path_length = path_length
        elif config['select_walk'] == 'longest_no_whs_kmer':
            combined_path, path_length = filter_wrong_haplotype_subpaths_kmer(nx_graph, combined_path, hap, config)
            penalized_path_length = path_length
        else:
            penalized_path_length = path_length
        
        # Update best path if this one is longer
        if penalized_path_length > best_penalized_length:
            best_walk = combined_path
            best_length = path_length
            best_penalized_length = penalized_path_length

    print(f"Best {walk_type} path found: {len(best_walk)} nodes, length={best_length:,}bp")

    return best_walk, best_length

def get_dijkstra_walk(nx_graph, hap, config, graphic_preds=None, walk_mode=None, chop_critical_hamming=1):
    """Find paths through the graph using a combination of greedy/random walks to find endpoints,
    followed by Dijkstra's algorithm to find optimal paths between those endpoints.
    
    Args:
        nx_graph: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary with parameters
        graphic_preds: Optional node predictions from graphic model
        walk_mode: Walking algorithm to use ('greedy', 'random', or 'beamsearch'). If None, defaults to 'greedy'.
        chop_critical_hamming: Optional threshold for chopping critical regions. If not None, 
                              analyze paths for regions where edge weights exceed this threshold
                              and keep only the longest contiguous chunk without critical regions.
    
    Returns:
        tuple: (best_walk, best_length) where best_walk is list of nodes and best_length is the total length
    """

    if graphic_preds is not None:
        print("Using graphic predictions for dijkstra optimization...")
        dijkstra_score_factor = 1
    else:
        print("Using kmer counts for dijkstra optimization...")
        dijkstra_score_factor = 1000000
    # Determine the walk type based on parameters
    if walk_mode is None:
        walk_mode = "greedy"
    
    walk_type = walk_mode
    sampled_edges = sample_edges(nx_graph, hap, config['sample_size'], sampling_by_score=config['sample_by_score'])
    print(f"\nSampled {len(sampled_edges)} edges to create {len(sampled_edges)} different {walk_type} walks...")
    if len(sampled_edges) == 0:
        return [], 0

    # Precompute edge weights once for all dijkstra calls (major optimization)
    if graphic_preds is not None:
        print("Precomputing graphic edge weights for dijkstra optimization...")
        precomputed_weights = precompute_graphic_edge_weights(nx_graph, graphic_preds, threshold=0.5)
    else:
        print("Precomputing edge weights for dijkstra optimization...")
        precomputed_weights = precompute_edge_weights(nx_graph, hap, kmer_count=True)

    # Check if we need to reverse the graph
    src = sampled_edges[0][0]  # example node
    if src^1 not in nx_graph.nodes():
        reversed_graph = nx_graph.reverse()
        reversed = True
    else:
        reversed = False
        reversed_graph = nx_graph

    # Store all valid paths with their metrics
    all_candidates = []

    # Create different walks using each sampled edge as a starting point
    for walk_idx, (src, dst) in enumerate(tqdm(sampled_edges, desc=f"Trying {walk_type} walks")):
        # First do greedy walks to find potential endpoints
        random_walk_mode = (walk_mode == "random")
        
        # Forward greedy walk
        forward_path, forward_length, forward_wrong_haplo_len, _, _ = walk_graph(
            nx_graph, dst, hap=hap, 
            max_wrong_haplo=config['max_wrong_haplo'],
            score_attr='score',
            random_mode=random_walk_mode,
            penalty_wrong_hap=config['penalty_wrong_hap'],
            penalty_ambiguous=config['penalty_ambiguous']
        )
        
        if not forward_path:
            continue

        if not reversed:
            src = src^1
            
        # Backward greedy walk
        backward_path, backward_length, backward_wrong_haplo_len, _, _ = walk_graph(
            reversed_graph, src, hap=hap, 
            max_wrong_haplo=config['max_wrong_haplo'],
            score_attr='score',
            random_mode=random_walk_mode,
            penalty_wrong_hap=config['penalty_wrong_hap'],
            penalty_ambiguous=config['penalty_ambiguous']
        )
        
        if not backward_path:
            continue

        # Get the endpoints from the greedy walks
        if reversed:
            start_node = backward_path[-1]  # Last node of backward path
            n_backward_path = [n for n in backward_path[::-1]]  # Just reverse
        else:
            start_node = backward_path[-1]^1  # Last node of backward path (complemented)
            n_backward_path = [n^1 for n in backward_path[::-1]]  # Reverse and complement
        
        end_node = forward_path[-1]  # Last node of forward path

        # Now do a Dijkstra walk between these endpoints
        if graphic_preds is not None:
            dijkstra_path, dijkstra_length = default_dijkstra_graphic(nx_graph, start_node, end_node, graphic_preds, threshold=config['graphic_epsilon'], precomputed_weights=precomputed_weights)
        else:
            dijkstra_path, dijkstra_length = default_dijkstra(nx_graph, start_node, end_node, hap, kmer_count=True, precomputed_weights=precomputed_weights)

        if not dijkstra_path:
            continue

        # Chop critical regions if requested
        if chop_critical_hamming is not None:
            dijkstra_path, dijkstra_length = chop_critical_regions(nx_graph, dijkstra_path, precomputed_weights, chop_critical_hamming, debug= True)
            if not dijkstra_path:
                continue

        # Calculate actual path length and other metrics
        path_length = 0
        log_prob_sum = 0
        for i in range(len(dijkstra_path)-1):
            src = dijkstra_path[i]
            dst = dijkstra_path[i+1]
            path_length += nx_graph[src][dst]['prefix_length']
            # Calculate log probability if using edge scores
            edge_score = nx_graph[src][dst]['score']
            sigmoid_score = 1 / (1 + np.exp(-edge_score))
            log_prob_sum += np.log(sigmoid_score)
        
        # Add length of last node
        path_length += nx_graph.nodes[dijkstra_path[-1]]['read_length']
        
        # Calculate average log probability
        log_prob_avg = log_prob_sum / len(dijkstra_path)

        # Calculate path score based on selection method
        if config['select_walk'] == 'haplo_penalty':
            #dijkstra_score = 1-(dijkstra_length*1000000/path_length) if path_length > 0 else 0
            
            score = path_length - dijkstra_length * dijkstra_score_factor
            #print(path_length, dijkstra_length)
            #score = dijkstra_score
        elif config['select_walk'] == 'additive':
            alpha = 1
            beta = 1
            gamma = 0.2
            path_length_score = alpha * np.log(path_length)
            pred_score = beta * log_prob_avg
            dijkstra_score = -gamma * np.log(dijkstra_length + 0.00001)
            score = path_length_score + pred_score + dijkstra_score
        elif config['select_walk'] == 'multiplicative':
            score = np.log(dijkstra_length) * log_prob_sum / np.log(path_length)
        else:
            score = path_length

        # Store this candidate
        all_candidates.append((dijkstra_path, path_length, dijkstra_length, score))

    # Select best path based on candidates
    if not all_candidates:
        return [], 0

    # Find the longest path length
    max_length = max(candidate[1] for candidate in all_candidates)
    
    # Take top N longest paths and select best by score
    n_longest = config['sample_size']//2
    all_candidates_by_length = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    top_n_candidates = all_candidates_by_length[:n_longest]
    
    # Select the best candidate by score
    best_candidate = max(top_n_candidates, key=lambda x: x[3])  # x[3] is score
    best_path, best_length, dijkstra_length, best_score = best_candidate
    
    print(f"Found {len(all_candidates)} total paths, filtered to top {n_longest} longest paths")
    print(f"Best {walk_type} path found: {len(best_path)} nodes, length={best_length:,}bp (score: {best_score:.2f}, dijkstra_length: {dijkstra_length:,})")

    return best_path, best_length

def chop_critical_regions(nx_graph, path, precomputed_weights, threshold, debug=False):
    """
    Analyze a path for critical regions where edge weights exceed the threshold,
    and return the longest contiguous chunk without critical regions.
    
    Args:
        nx_graph: NetworkX graph
        path: List of nodes representing the path
        precomputed_weights: Dictionary of precomputed edge weights
        threshold: Threshold above which regions are considered critical
        debug: If True, print detailed information about critical and normal regions
    
    Returns:
        tuple: (chopped_path, recalculated_dijkstra_length) where chopped_path is the longest valid subpath
    """
    if len(path) <= 1:
        return path, 0
    
    if debug:
        print(f"\n=== CHOP CRITICAL REGIONS DEBUG ===")
        print(f"Path length: {len(path)} nodes")
        print(f"Critical threshold: {threshold}")
        print(f"Path: {path[:10]}{'...' if len(path) > 10 else ''}")
    
    # Identify critical regions by analyzing edge weights
    critical_edges = []
    edge_details = []
    
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        edge_weight = precomputed_weights.get((src, dst), 0)
        edge_length = nx_graph[src][dst]['prefix_length']
        
        edge_details.append({
            'index': i,
            'src': src,
            'dst': dst,
            'weight': edge_weight,
            'length': edge_length,
            'is_critical': edge_weight > threshold
        })
        
        # If edge weight exceeds threshold, this edge is in a critical region
        if edge_weight > threshold:
            critical_edges.append(i)
    
    if debug:
        print(f"\n--- EDGE ANALYSIS ---")
        for detail in edge_details:
            status = "CRITICAL" if detail['is_critical'] else "normal"
            print(f"Edge {detail['index']}: {detail['src']} -> {detail['dst']} | "
                  f"weight: {detail['weight']:.4f} | length: {detail['length']} | {status}")
    
    if not critical_edges:
        # No critical regions, return original path
        dijkstra_length = sum(precomputed_weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        if debug:
            print(f"\nNo critical regions found. Returning original path.")
            print(f"Total dijkstra length: {dijkstra_length:.4f}")
        return path, dijkstra_length
    
    # Find contiguous regions of critical edges
    critical_regions = []
    if critical_edges:
        start = critical_edges[0]
        end = start
        
        for i in range(1, len(critical_edges)):
            if critical_edges[i] == critical_edges[i-1] + 1:
                # Contiguous critical edge
                end = critical_edges[i]
            else:
                # Gap found, save current region and start new one
                critical_regions.append((start, end))
                start = critical_edges[i]
                end = start
        
        # Add the last region
        critical_regions.append((start, end))
    
    if debug:
        print(f"\n--- CRITICAL REGIONS ---")
        for i, (edge_start, edge_end) in enumerate(critical_regions):
            region_weight = sum(edge_details[j]['weight'] for j in range(edge_start, edge_end + 1))
            region_length = sum(edge_details[j]['length'] for j in range(edge_start, edge_end + 1))
            node_start = edge_start
            node_end = edge_end + 1
            print(f"Critical region {i+1}: edges {edge_start}-{edge_end} (nodes {node_start}-{node_end})")
            print(f"  Total weight: {region_weight:.4f}, Total length: {region_length}")
            print(f"  Nodes: {path[node_start:node_end+1]}")
    
    # Create list of valid ranges (node indices, not edge indices)
    valid_ranges = []
    last_end = -1
    
    for edge_start, edge_end in critical_regions:
        # Convert edge indices to node indices
        # Critical region affects nodes from edge_start to edge_end+1
        node_start = edge_start
        node_end = edge_end + 1
        
        # Add the range before this critical region
        if node_start > last_end + 1:
            valid_ranges.append((last_end + 1, node_start - 1))
        last_end = node_end
    
    # Add the final range after the last critical region
    if last_end < len(path) - 1:
        valid_ranges.append((last_end + 1, len(path) - 1))
    
    if debug:
        print(f"\n--- VALID (NORMAL) REGIONS ---")
        for i, (start_idx, end_idx) in enumerate(valid_ranges):
            region_nodes = path[start_idx:end_idx + 1]
            region_length = 0
            region_weight = 0
            
            # Calculate length and weight for this region
            for j in range(len(region_nodes) - 1):
                edge_idx = start_idx + j
                if edge_idx < len(edge_details):
                    region_length += edge_details[edge_idx]['length']
                    region_weight += edge_details[edge_idx]['weight']
            
            # Add final node length
            if region_nodes:
                region_length += nx_graph.nodes[region_nodes[-1]]['read_length']
            
            print(f"Normal region {i+1}: nodes {start_idx}-{end_idx} (length: {len(region_nodes)})")
            print(f"  Total weight: {region_weight:.4f}, Total length: {region_length}")
            print(f"  Nodes: {region_nodes}")
    
    # Find the longest valid range
    if not valid_ranges:
        if debug:
            print(f"\nNo valid ranges found after removing critical regions!")
        return [], 0
    
    longest_range = max(valid_ranges, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_range
    
    # Extract the longest valid subpath
    chopped_path = path[start_idx:end_idx + 1]
    
    # Recalculate dijkstra length for chopped path
    dijkstra_length = 0
    for i in range(len(chopped_path) - 1):
        src, dst = chopped_path[i], chopped_path[i + 1]
        dijkstra_length += precomputed_weights.get((src, dst), 0)
    
    if debug:
        print(f"\n--- FINAL RESULT ---")
        print(f"Original path length: {len(path)} nodes")
        print(f"Chopped path length: {len(chopped_path)} nodes")
        print(f"Selected range: nodes {start_idx}-{end_idx}")
        print(f"Chopped path: {chopped_path}")
        print(f"Final dijkstra length: {dijkstra_length:.4f}")
        
        # Calculate actual path length for chopped path
        actual_length = 0
        for i in range(len(chopped_path) - 1):
            actual_length += nx_graph[chopped_path[i]][chopped_path[i+1]]['prefix_length']
        if chopped_path:
            actual_length += nx_graph.nodes[chopped_path[-1]]['read_length']
        print(f"Final actual length: {actual_length}")
        print(f"=== END CHOP CRITICAL REGIONS DEBUG ===\n")
    
    return chopped_path, dijkstra_length

def dag_longest_walk(G, hap=None, config=None, kmer_count=True):
    
    """
    Find the longest path in a DAG that doesn't contain complement node pairs and respects haplotype constraints.
    
    Args:
        G: NetworkX DAG
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary with parameters
        kmer_count: If True, use kmer counts for penalty calculation; if False, use yak values
    """

    hap_penalty = config['hap_length_penalty']
    
    # Set up wrong haplotype identifier for kmer counting
    if kmer_count and hap:
        not_hap = 'm' if hap == 'p' else 'p'
    
    topo_order = list(nx.topological_sort(G))
    # Store (length, path_set, predecessor) for each node
    dist = {}  
    
    # Initialize progress bar
    pbar = tqdm(total=len(topo_order), desc=f"Finding longest {hap} path", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} nodes [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Track nodes processed for batch updates
    nodes_processed = 0
    batch_size = 1000
    
    # Initialize distances
    for v in topo_order:
        # Initialize with path_set containing current node and its complement
        v_complement = v ^ 1
        initial_path_set = {v, v_complement}
            
        dist[v] = (0, initial_path_set, None)
        
        # Get predecessors and their potential contributions
        predecessors = []
        for u, data in G.pred[v].items():
            prev_length, prev_path_set, _ = dist[u]
            
            # O(1) check if node is already in path
            if v in prev_path_set:
                continue
                
            # Calculate length contribution based on haplotype
            length_contribution = G.nodes[v]['read_length'] - data['overlap_length']
            
            if hap:
                if kmer_count:
                    # Use kmer count for penalty calculation
                    kmer_count_val = G.nodes[v][f'kmer_count_{not_hap}']
                    kmer_count_correct = G.nodes[v][f'kmer_count_{hap}']
                    # Apply penalty proportional to wrong haplotype kmer count
                    #penalty = max(0, length_contribution / G.nodes[v]['read_length'] * kmer_count_val)
                    #if kmer_count_val > kmer_count_correct:
                    #    length_contribution = - kmer_count_val * hap_penalty * length_contribution  # length_contribution - hap_penalty * penalty
                    #else:
                    #    length_contribution = length_contribution
                    length_contribution *=  1 - kmer_count_val * hap_penalty
                    #print(kmer_count_val * hap_penalty)
                else:
                    # Use yak values for penalty calculation (original logic)
                    yak_val = G.nodes[v][f'yak_{hap}']
                    if yak_val == -1:  # Wrong haplotype - reverse length
                        length_contribution = - hap_penalty * length_contribution
            
            #if G.nodes[v][f'ambiguous'] == 1:
            #    length_contribution = 0.5 * length_contribution

            predecessors.append((
                prev_length + length_contribution,
                u
            ))
        
        # Update distance if valid predecessors exist
        if predecessors:
            # Choose predecessor with longest path
            best_length, best_pred = max(predecessors, key=lambda x: x[0])
            
            # Create new path set by adding current node and its complement
            new_path_set = dist[best_pred][1].copy()
            new_path_set.add(v)
            v_complement = v ^ 1
            new_path_set.add(v_complement)
            
            # Get nodes that are both successors of current node's complement
            # and predecessors of best_pred's complement
            best_pred_complement = best_pred ^ 1
            if v_complement in G and best_pred_complement in G:
                # Get nodes that are both successors of current and predecessors of best_pred
                transitive_nodes = set(G.successors(v_complement)) & set(G.predecessors(best_pred_complement))
                # Add these nodes to the path set
                new_path_set.update(transitive_nodes)
                
            dist[v] = (best_length, new_path_set, best_pred)
            
        # Update progress bar in batches
        nodes_processed += 1
        if nodes_processed >= batch_size:
            pbar.update(nodes_processed)
            if len(dist[v][1]) > 1:  # If path has more than one node
                pbar.set_postfix({'path_len': len(dist[v][1])})
            nodes_processed = 0
    
    # Update progress bar with remaining nodes
    if nodes_processed > 0:
        pbar.update(nodes_processed)
    
    pbar.close()
    
    # Find the end node with the longest path
    max_length = float('-inf')
    best_end_node = None
    
    for v in dist:
        length = dist[v][0]
        
        # Add final node's read length (with penalty if wrong haplotype)
        if hap:
            if kmer_count:
                # Use kmer count for final node penalty
                kmer_count_val = G.nodes[v][f'kmer_count_{not_hap}']
                kmer_count_correct = G.nodes[v][f'kmer_count_{hap}']
                if kmer_count_val > kmer_count_correct:
                    length -= kmer_count_val * hap_penalty * G.nodes[v]['read_length']
                else:
                    length += G.nodes[v]['read_length']
            else:
                # Use yak values for final node penalty (original logic)
                if G.nodes[v][f'yak_{hap}'] == -1:
                    length -= hap_penalty * G.nodes[v]['read_length']
                else:
                    length += G.nodes[v]['read_length']
        else:
            length += G.nodes[v]['read_length']
        
        if length > max_length:
            max_length = length
            best_end_node = v
    
    # Reconstruct path by following predecessors
    best_path = []
    current = best_end_node
    while current is not None:
        best_path.append(current)
        current = dist[current][2]  # Get predecessor
    
    # Create compressed value string for path analysis
    if hap:
        if kmer_count:
            kmer_values = [G.nodes[n][f'kmer_count_{not_hap}'] for n in best_path[::-1]]  # Reverse to match path order
            compressed = []
            if kmer_values:
                current_val = kmer_values[0]
                count = 1
                for val in kmer_values[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        compressed.append(f"({count}x {current_val})")
                        current_val = val
                        count = 1
                compressed.append(f"({count}x {current_val})")  # Add final group
                kmer_summary = " ".join(compressed)
                print(f"Path kmer_count_{not_hap} values: {kmer_summary}")
        else:
            yak_values = [G.nodes[n][f'yak_{hap}'] for n in best_path[::-1]]  # Reverse to match path order
            compressed = []
            if yak_values:
                current_val = yak_values[0]
                count = 1
                for val in yak_values[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        compressed.append(f"({count}x {current_val})")
                        current_val = val
                        count = 1
                compressed.append(f"({count}x {current_val})")  # Add final group
                yak_summary = " ".join(compressed)
                print(f"Path YAK values: {yak_summary}")
    
    return best_path[::-1], max_length  # Reverse path to get correct order

def dag_longest_walk_single_strand(G, hap=None, config=None, kmer_count=False):
    """
    Find the longest path in a DAG that respects haplotype constraints.
    This version assumes there are no complement nodes in the graph.
    
    Args:
        G: NetworkX DAG
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary with parameters
        kmer_count: If True, use kmer counts for penalty calculation; if False, use yak values
    """
    hap_penalty = config['hap_length_penalty']
    
    # Set up wrong haplotype identifier for kmer counting
    if kmer_count and hap:
        not_hap = 'm' if hap == 'p' else 'p'
    
    topo_order = list(nx.topological_sort(G))
    # Store (length, predecessor) for each node
    dist = {}  
    
    # Initialize progress bar
    pbar = tqdm(total=len(topo_order), desc=f"Finding longest {hap} path", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} nodes [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Track nodes processed for batch updates
    nodes_processed = 0
    batch_size = 1000
    
    # Initialize distances
    for v in topo_order:
        # Initialize with length 0 and no predecessor
        dist[v] = (0, None)
        
        # Get predecessors and their potential contributions
        predecessors = []
        for u, data in G.pred[v].items():
            prev_length, _ = dist[u]
                
            # Calculate length contribution based on haplotype
            length_contribution = G.nodes[v]['read_length'] - data['overlap_length']
            if hap:
                if kmer_count:
                    # Use kmer count for penalty calculation
                    kmer_count_val = G.nodes[v][f'kmer_count_{not_hap}']
                    # Apply penalty proportional to wrong haplotype kmer count
                    penalty = max(0, length_contribution / G.nodes[v]['read_length'] * kmer_count_val)
                    length_contribution = length_contribution - hap_penalty * penalty
                else:
                    # Use yak values for penalty calculation (original logic)
                    yak_val = G.nodes[v][f'yak_{hap}']
                    if yak_val == -1:  # Wrong haplotype - reverse length
                        length_contribution = - hap_penalty * length_contribution
            
            #if G.nodes[v][f'ambiguous'] == 1:
            #    length_contribution = 0.5 * length_contribution

            predecessors.append((
                prev_length + length_contribution,
                u
            ))
        
        # Update distance if valid predecessors exist
        if predecessors:
            # Choose predecessor with longest path
            best_length, best_pred = max(predecessors, key=lambda x: x[0])
            dist[v] = (best_length, best_pred)
            
        # Update progress bar in batches
        nodes_processed += 1
        if nodes_processed >= batch_size:
            pbar.update(nodes_processed)
            nodes_processed = 0
    
    # Update progress bar with remaining nodes
    if nodes_processed > 0:
        pbar.update(nodes_processed)
    
    pbar.close()
    
    # Find the end node with the longest path
    max_length = float('-inf')
    best_end_node = None
    
    for v in dist:
        length = dist[v][0]
        
        # Add final node's read length (with penalty if wrong haplotype)
        if hap:
            if kmer_count:
                # Use kmer count for final node penalty
                kmer_count_val = G.nodes[v][f'kmer_count_{not_hap}']
                penalty = max(0, kmer_count_val)
                length = length + G.nodes[v]['read_length'] - hap_penalty * penalty
            else:
                # Use yak values for final node penalty (original logic)
                if G.nodes[v][f'yak_{hap}'] == -1:
                    length -= hap_penalty * G.nodes[v]['read_length']
                else:
                    length += G.nodes[v]['read_length']
        else:
            length += G.nodes[v]['read_length']
        
        if length > max_length:
            max_length = length
            best_end_node = v
    
    # Reconstruct path by following predecessors
    best_path = []
    current = best_end_node
    while current is not None:
        best_path.append(current)
        current = dist[current][1]  # Get predecessor
    
    # Create compressed value string for path analysis
    if hap:
        if kmer_count:
            kmer_values = [G.nodes[n][f'kmer_count_{not_hap}'] for n in best_path[::-1]]  # Reverse to match path order
            compressed = []
            if kmer_values:
                current_val = kmer_values[0]
                count = 1
                for val in kmer_values[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        compressed.append(f"({count}x {current_val})")
                        current_val = val
                        count = 1
                compressed.append(f"({count}x {current_val})")  # Add final group
                kmer_summary = " ".join(compressed)
                print(f"Path kmer_count_{not_hap} values: {kmer_summary}")
        else:
            yak_values = [G.nodes[n][f'yak_{hap}'] for n in best_path[::-1]]  # Reverse to match path order
            compressed = []
            if yak_values:
                current_val = yak_values[0]
                count = 1
                for val in yak_values[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        compressed.append(f"({count}x {current_val})")
                        current_val = val
                        count = 1
                compressed.append(f"({count}x {current_val})")  # Add final group
                yak_summary = " ".join(compressed)
                print(f"Path YAK values: {yak_summary}")
    
    return best_path[::-1], max_length  # Reverse path to get correct order

def source_sink_walk(G, hap=None, config=None, graphic_preds=None, source_sink=False, len_thr_percentage=0.9, n_longest=10):
    """
    Find the best path in a DAG from a source node to a sink node that respects haplotype constraints.
    Uses a modified Dijkstra algorithm to find paths with minimal wrong haplotype nodes.
    Ensures that both a node n and its complement n^1 cannot be in the same path.
    
    Args:
        G: NetworkX DAG
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary with parameters
        graphic_preds: Graph predictions for node filtering (optional)
        source_sink: Whether to use all source-sink pairs or just greedy pairs
        len_thr_percentage: Percentage threshold for filtering by length (default: 0.9)
        n_longest: Number of longest paths to consider when filter_candidates == 'n_longest' (default: 10)
    
    Returns:
        tuple: (path, path_length) where path is list of nodes and path_length is the total length
    """
    
    # Find source nodes (nodes with no in-edges) and sink nodes (nodes with no out-edges)
    source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not source_nodes or not sink_nodes:
        print("No source or sink nodes found in the graph")
        return [], 0
    
    # Filter source and sink nodes based on haplotype if specified
    if hap is not None:
        # First try direct filtering
        valid_source_nodes = [n for n in source_nodes if G.nodes[n][f'yak_{hap}'] != -1]
        valid_sink_nodes = [n for n in sink_nodes if G.nodes[n][f'yak_{hap}'] != -1]
        
        # If no valid nodes found, fall back to using all source/sink nodes
        # The algorithm will still prefer correct haplotype paths through edge weighting
        if not valid_source_nodes:
            print(f"No valid {hap} source nodes found directly, using all source nodes")
            valid_source_nodes = source_nodes
        if not valid_sink_nodes:
            print(f"No valid {hap} sink nodes found directly, using all sink nodes")
            valid_sink_nodes = sink_nodes
    elif graphic_preds is not None:
        valid_source_nodes = [n for n in source_nodes if graphic_preds[n] > -0.1]
        valid_sink_nodes = [n for n in sink_nodes if graphic_preds[n] > -0.1]
    else:
        valid_source_nodes = source_nodes
        valid_sink_nodes = sink_nodes
    
    print(f"Found {len(valid_source_nodes)} valid source nodes and {len(valid_sink_nodes)} valid sink nodes")

    # Pre-filter to find valid pairs (not complement and has path)
    print("Pre-filtering to find valid source-sink pairs...")
    valid_pairs = []
    
    if source_sink:
        # Create all possible source-sink combinations and randomize order
        all_possible_pairs = [(s, t) for s in valid_source_nodes for t in valid_sink_nodes]
        random.shuffle(all_possible_pairs)
        
        # Check pairs in random order until we find 100 valid ones
        max_valid_pairs = 100
        total_checked = 0
        
        for source, sink in tqdm(all_possible_pairs, desc="Finding valid pairs"):
            total_checked += 1
            
            # Skip if source is complement of sink
            if source == sink^1:
                continue
            # Skip if no path exists
            if not nx.has_path(G, source, sink):
                continue
            
            # This is a valid pair
            valid_pairs.append((source, sink))
            
            # Stop if we have enough valid pairs
            if len(valid_pairs) >= max_valid_pairs:
                break
        
        print(f"Found {len(valid_pairs)} valid pairs after checking {total_checked} source-sink combinations")
    
    # Add greedy pairs (these should already be valid, but we'll filter them too)
    greedy_pairs = find_greedy_pairs(G, valid_source_nodes, hap, config, num_greedy_pairs=25)
    for source, sink in tqdm(greedy_pairs, desc="Processing greedy pairs"):
        if source != sink^1 and nx.has_path(G, source, sink):
            if (source, sink) not in valid_pairs:  # Avoid duplicates
                valid_pairs.append((source, sink))
    
    # No need to sample since we already limited to max_valid_pairs
    if source_sink:
        pairs_to_test = valid_pairs + greedy_pairs
    else:
        pairs_to_test = greedy_pairs
    
    print(f"Testing {len(pairs_to_test)} valid source-sink pairs")

    # Precompute edge weights once for all dijkstra calls (major optimization)
    if graphic_preds is not None:
        print("Precomputing graphic edge weights for dijkstra optimization...")
        precomputed_weights = precompute_graphic_edge_weights(G, graphic_preds, threshold=0.1)
    else:
        print("Precomputing edge weights for dijkstra optimization...")
        precomputed_weights = precompute_edge_weights(G, hap, kmer_count=True)

    # Track progress
    pbar = tqdm(total=len(pairs_to_test), desc="Finding best source-sink path")
    
    # Store all valid paths with their metrics
    all_candidates = []
    
    for source, sink in pairs_to_test:
        # No need to check validity again since we pre-filtered
        if graphic_preds is not None:
            path, dijkstra_length = default_dijkstra_graphic(G, source, sink, graphic_preds, threshold=config['graphic_epsilon'], precomputed_weights=precomputed_weights)
        else:
            path, dijkstra_length = default_dijkstra(G, source, sink, hap, kmer_count=True, precomputed_weights=precomputed_weights)
        
        # Chop critical regions if requested
        if chop_critical_hamming is not None and path:
            path, dijkstra_length = chop_critical_regions(G, path, precomputed_weights, chop_critical_hamming, debug=config.get('debug_chop_critical', False))
            if not path:
                pbar.update(1)
                continue
        
        # Check for complement pairs in the path and chop if necessary
        original_path = path
        path, removed_pairs_count = remove_complement_pairs_from_path(path)
    
        if path is not None:
            # Skip single node paths
            if len(path) <= 1:
                print(f"Skipping path with {len(path)} nodes (source: {source}, sink: {sink})")
                pbar.update(1)
                continue
            
            #print(f"Processing valid path with {len(path)} nodes (source: {source}, sink: {sink})")
                            
            # Calculate actual path length (not weighted)
            path_length = 0
            for i in range(len(path) - 1):
                path_length += G[path[i]][path[i+1]]['prefix_length']
            
            # Add length of last node
            path_length += G.nodes[path[-1]]['read_length']
            
            # Calculate score_sum and log_prob_sum
            score_sum = 0
            log_prob_sum = 0
            for i in range(len(path) - 1):
                edge_score = G[path[i]][path[i+1]]['score']
                sigmoid_score = 1 / (1 + np.exp(-edge_score))
                score_sum += sigmoid_score
                log_prob_sum += np.log(sigmoid_score)
            
            # Score is the difference between actual length and weighted Dijkstra length
            # A smaller difference means fewer wrong haplotype nodes
            score_avg = score_sum/len(path)
            score_log_avg = -log_prob_sum/len(path)
            dijkstra_score = 1-(dijkstra_length*1000000/path_length) if path_length > 0 else 0
            gamma = 0.2

            if config['select_walk'] == 'longest':
                score = path_length
            elif config['select_walk'] == 'haplo_penalty':
                score = dijkstra_score
                #score = path_length - dijkstra_score * 10
            elif config['select_walk'] == 'log_prob':
                score = (1-gamma) * score_log_avg + (gamma * dijkstra_score)
            elif config['select_walk'] == 'prob':
                score = score_avg * dijkstra_score
            else:
                raise ValueError(f"Invalid select_walk: {config['select_walk']}")
            

            # Store this candidate for later selection (including dijkstra_score to avoid recalculation)
            all_candidates.append((path, path_length, dijkstra_length, score, dijkstra_score))

        
        pbar.update(1)
    
    pbar.close()
    

    # Filter candidates and select best one
    print(f"Found {len(all_candidates)} total candidates")
    if not all_candidates:
        best_path = []
        best_length = 0
    else:
        # Find the longest path length
        max_length = max(candidate[1] for candidate in all_candidates)
        filter_candidates = 'n_longest'  # Can be False, True, or 'n_longest'
        n_longest = max(1, len(all_candidates)//2)  # Ensure at least 1 candidate is selected
        len_thr_percentage = 0.9
        
        if filter_candidates == True:
            # Filter for paths with at least 90% of the longest length
            length_threshold = len_thr_percentage * max_length
            filtered_candidates = [candidate for candidate in all_candidates if candidate[1] >= length_threshold]
            
            # Among filtered candidates, select the one with highest score
            best_candidate = max(filtered_candidates, key=lambda x: x[3])  # x[3] is score
            best_path, best_length, dijkstra_length, best_score, dijkstra_score = best_candidate
            
            print(f"Found {len(all_candidates)} total paths, {len(filtered_candidates)} paths >= 90% of max length ({max_length:,}bp)")
            print(f"Selected path with length {best_length:,}bp (score: {best_score:.2f}, dijkstra_score: {dijkstra_score:.2f}, dijkstra_length: {dijkstra_length:,})")
        elif filter_candidates == 'n_longest':
            # Sort all candidates by length (descending) and take the top n
            all_candidates_by_length = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            top_n_candidates = all_candidates_by_length[:n_longest]
            
            # Among the top n longest paths, select the one with highest score
            best_candidate = max(top_n_candidates, key=lambda x: x[3])  # x[3] is score
            best_path, best_length, dijkstra_length, best_score, dijkstra_score = best_candidate
            
            print(f"Found {len(all_candidates)} total paths, filtered to top {len(top_n_candidates)} longest paths")
            print(f"Selected path with length {best_length:,}bp (score: {best_score:.2f}, dijkstra_score: {dijkstra_score:.2f}, dijkstra_length: {dijkstra_length:,})")
        else:
            # No filtering, just select the best candidate by score
            best_candidate = max(all_candidates, key=lambda x: x[3])  # x[3] is score
            best_path, best_length, dijkstra_length, best_score, dijkstra_score = best_candidate
            
            print(f"Found {len(all_candidates)} total paths")
            print(f"Selected path with length {best_length:,}bp (score: {best_score:.2f}, dijkstra_score: {dijkstra_score:.2f}, dijkstra_length: {dijkstra_length:,})")
    
    return best_path, best_length

# Wrapper functions for backward compatibility
def greedy_walk(nx_graph, hap, config):
    return get_walk(nx_graph, hap, config, random_mode=False)

def random_walk(nx_graph, hap, config):
    return get_walk(nx_graph, hap, config, random_mode=True)

def dag_longest_walk_source_sink(G, hap=None, config=None, kmer_count=False):
    """
    Find the longest path in a DAG from a source node (no in-edges) to a sink node (no out-edges)
    that respects haplotype constraints.
    
    Args:
        G: NetworkX DAG
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary with parameters
        kmer_count: If True, use kmer counts for penalty calculation; if False, use yak values
    
    Returns:
        tuple: (path, path_length) where path is list of nodes and path_length is the total length
    """
    hap_penalty = config['hap_length_penalty']
    
    # Set up wrong haplotype identifier for kmer counting
    if kmer_count and hap:
        not_hap = 'm' if hap == 'p' else 'p'
    
    # Find source nodes (nodes with no in-edges) and sink nodes (nodes with no out-edges)
    source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not source_nodes or not sink_nodes:
        print("No source or sink nodes found in the graph")
        return [], 0
    
    print(f"Found {len(source_nodes)} source nodes and {len(sink_nodes)} sink nodes")
    
    # Process nodes in topological order
    topo_order = list(nx.topological_sort(G))
    
    # For each node, store: (best_path_length, visited_nodes_set, predecessor)
    dist = {}
    
    # Initialize progress bar
    pbar = tqdm(total=len(topo_order), desc=f"Finding longest {hap} source-sink path")
    
    # Initialize all nodes
    for v in topo_order:
        v_complement = v ^ 1
        
        # Source nodes start with length 0, all others with -infinity
        if v in source_nodes:
            dist[v] = (0, {v, v_complement}, None)
        else:
            dist[v] = (float('-inf'), {v, v_complement}, None)
        
        # For each node, find the best predecessor
        best_length = float('-inf')
        best_pred = None
        best_path_set = {v, v_complement}
        
        # Check all predecessors
        for u in G.predecessors(v):
            prev_length, prev_path_set, _ = dist[u]
            
            # Skip if previous node is not reachable or current node is already in path
            if prev_length == float('-inf') or v in prev_path_set:
                continue
            
            # Calculate edge length (with haplotype penalty if applicable)
            edge_length = G[u][v]['prefix_length']
            if hap:
                if kmer_count:
                    # Use kmer count for penalty calculation
                    kmer_count_val = G.nodes[v][f'kmer_count_{not_hap}']
                    # Apply penalty proportional to wrong haplotype kmer count
                    penalty = max(0, kmer_count_val)
                    edge_length = edge_length - hap_penalty * penalty
                else:
                    # Use yak values for penalty calculation (original logic)
                    if G.nodes[v][f'yak_{hap}'] == -1:  # Wrong haplotype
                        edge_length = -hap_penalty * edge_length
            
            # Calculate total path length through this predecessor
            path_length = prev_length + edge_length
            
            # Update best predecessor if this path is longer
            if path_length > best_length:
                best_length = path_length
                best_pred = u
                best_path_set = prev_path_set.copy()
                best_path_set.add(v)
                best_path_set.add(v_complement)
        
        # Update node if we found a valid predecessor
        if best_pred is not None:
            dist[v] = (best_length, best_path_set, best_pred)
        
        pbar.update(1)
    
    pbar.close()
    
    # Find the sink node with the longest path
    best_end_node = None
    max_length = float('-inf')
    
    for v in sink_nodes:
        length = dist[v][0]
        
        # Skip unreachable sink nodes
        if length == float('-inf'):
            continue
        
        # Add final node's read length (with penalty if wrong haplotype)
        if hap:
            if kmer_count:
                # Use kmer count for final node penalty
                kmer_count_val = G.nodes[v][f'kmer_count_{not_hap}']
                penalty = max(0, kmer_count_val)
                length = length + G.nodes[v]['read_length'] - hap_penalty * penalty
            else:
                # Use yak values for final node penalty (original logic)
                if G.nodes[v][f'yak_{hap}'] == -1:
                    length -= hap_penalty * G.nodes[v]['read_length']
                else:
                    length += G.nodes[v]['read_length']
        else:
            length += G.nodes[v]['read_length']
        
        if length > max_length:
            max_length = length
            best_end_node = v
    
    # If no valid path found
    if best_end_node is None:
        print("No valid source-sink path found")
        return [], 0
    
    # Reconstruct path by following predecessors
    best_path = []
    current = best_end_node
    while current is not None:
        best_path.append(current)
        current = dist[current][2]  # Get predecessor
    
    # Report haplotype composition if applicable
    if hap:
        if kmer_count:
            kmer_values = [G.nodes[n][f'kmer_count_{not_hap}'] for n in best_path[::-1]]
            compressed = []
            if kmer_values:
                current_val = kmer_values[0]
                count = 1
                for val in kmer_values[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        compressed.append(f"({count}x {current_val})")
                        current_val = val
                        count = 1
                compressed.append(f"({count}x {current_val})")
                kmer_summary = " ".join(compressed)
                print(f"Path kmer_count_{not_hap} values: {kmer_summary}")
        else:
            yak_values = [G.nodes[n][f'yak_{hap}'] for n in best_path[::-1]]
            compressed = []
            if yak_values:
                current_val = yak_values[0]
                count = 1
                for val in yak_values[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        compressed.append(f"({count}x {current_val})")
                        current_val = val
                        count = 1
                compressed.append(f"({count}x {current_val})")
                yak_summary = " ".join(compressed)
                print(f"Path YAK values: {yak_summary}")
    
    print(f"Best source-sink path found: {len(best_path)} nodes, length={max_length:,}bp")
    return best_path[::-1], max_length  # Reverse path to get correct order

def dag_longest_walk__(G, hap=None, config=None):
    """
    Find the longest path in a DAG that doesn't contain complement node pairs and respects haplotype constraints.
    This version filters out walks with complement conflicts in the neighborhood of final nodes,
    as well as all subpaths of filtered walks.
    
    Args:
        G: NetworkX DAG
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary with parameters
    """
    hap_penalty = config['hap_length_penalty']
    topo_order = list(nx.topological_sort(G))
    # Store (length, path_set, predecessor) for each node
    dist = {}  
    
    # Initialize progress bar
    pbar = tqdm(total=len(topo_order), desc=f"Finding longest {hap} path", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} nodes [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Track nodes processed for batch updates
    nodes_processed = 0
    batch_size = 1000
    
    # Initialize distances
    for v in topo_order:
        # Initialize with path_set containing current node and its complement
        v_complement = v ^ 1
        initial_path_set = {v, v_complement}
            
        dist[v] = (0, initial_path_set, None)
        
        # Get predecessors and their potential contributions
        predecessors = []
        for u, data in G.pred[v].items():
            prev_length, prev_path_set, _ = dist[u]
            
            # O(1) check if node is already in path
            if v in prev_path_set:
                continue
                
            # Calculate length contribution based on haplotype
            length_contribution = G.nodes[v]['read_length'] - data['overlap_length']
            if hap:
                yak_val = G.nodes[v][f'yak_{hap}']
                if yak_val == -1:  # Wrong haplotype - reverse length
                    length_contribution = - hap_penalty * length_contribution
            
            #if G.nodes[v][f'ambiguous'] == 1:
            #    length_contribution = 0.5 * length_contribution

            predecessors.append((
                prev_length + length_contribution,
                u
            ))
        
        # Update distance if valid predecessors exist
        if predecessors:
            # Choose predecessor with longest path
            best_length, best_pred = max(predecessors, key=lambda x: x[0])
            
            # Create new path set by adding current node and its complement
            new_path_set = dist[best_pred][1].copy()
            new_path_set.add(v)
            v_complement = v ^ 1
            new_path_set.add(v_complement)
            
            # Get nodes that are both successors of current node's complement
            # and predecessors of best_pred's complement
            best_pred_complement = best_pred ^ 1
            if v_complement in G and best_pred_complement in G:
                # Get nodes that are both successors of current and predecessors of best_pred
                transitive_nodes = set(G.successors(v_complement)) & set(G.predecessors(best_pred_complement))
                # Add these nodes to the path set
                new_path_set.update(transitive_nodes)
                
            dist[v] = (best_length, new_path_set, best_pred)
            
        # Update progress bar in batches
        nodes_processed += 1
        if nodes_processed >= batch_size:
            pbar.update(nodes_processed)
            if len(dist[v][1]) > 1:  # If path has more than one node
                pbar.set_postfix({'path_len': len(dist[v][1])})
            nodes_processed = 0
    
    # Update progress bar with remaining nodes
    if nodes_processed > 0:
        pbar.update(nodes_processed)
    
    pbar.close()
    
    # First, collect all paths and check for conflicts
    all_paths = {}  # Dictionary mapping end nodes to their paths
    invalid_paths = set()  # Set of paths (as tuples) that have conflicts
    
    for v in dist:
        # Reconstruct path for this end node
        path = []
        current = v
        while current is not None:
            path.append(current)
            current = dist[current][2]  # Get predecessor
        path = path[::-1]  # Reverse to get correct order
        
        # Store the path
        all_paths[v] = path
        
        # Check for complement conflicts in the 1-hop neighborhood
        has_conflict = False
        v_complement = v ^ 1
        
        if v_complement in G and len(path) > 1:
            last_node = path[-1]
            last_node_complement = last_node ^ 1
            
            if last_node_complement in G:
                # Check if there's a path from any node in the path to the complement of the last node
                for node in path[:-1]:  # Exclude the last node itself
                    if G.has_edge(node, last_node_complement):
                        has_conflict = True
                        break
                    
                    # Check 1-hop neighborhood
                    node_successors = set(G.successors(node))
                    last_complement_predecessors = set(G.predecessors(last_node_complement))
                    
                    if node_successors & last_complement_predecessors:
                        has_conflict = True
                        break
        
        if has_conflict:
            invalid_paths.add(tuple(path))
    
    # Now filter out all subpaths of invalid paths
    filtered_candidates = []
    
    for end_node, path in all_paths.items():
        path_tuple = tuple(path)
        
        # Skip if this path is already marked as invalid
        if path_tuple in invalid_paths:
            continue
        
        # Check if this path is a subpath of any invalid path
        is_subpath = False
        for invalid_path in invalid_paths:
            # Check if path is a subpath (subsequence) of invalid_path
            if len(path) < len(invalid_path):
                # Try to find path as a contiguous subsequence in invalid_path
                for i in range(len(invalid_path) - len(path) + 1):
                    if path_tuple == invalid_path[i:i+len(path)]:
                        is_subpath = True
                        break
            
            if is_subpath:
                break
        
        if not is_subpath:
            # Calculate final length including the last node's read length
            length = dist[end_node][0]
            if hap and G.nodes[end_node][f'yak_{hap}'] == -1:
                length -= hap_penalty * G.nodes[end_node]['read_length']
            else:
                length += G.nodes[end_node]['read_length']
                
            filtered_candidates.append((end_node, length))
    
    # Select the best candidate
    if filtered_candidates:
        best_end_node, max_length = max(filtered_candidates, key=lambda x: x[1])
        best_path = all_paths[best_end_node]
    else:
        print("No valid paths found without complement conflicts")
        return [], 0
    
    # Create compressed yak value string
    if hap:
        yak_values = [G.nodes[n][f'yak_{hap}'] for n in best_path]
        compressed = []
        if yak_values:
            current_val = yak_values[0]
            count = 1
            for val in yak_values[1:]:
                if val == current_val:
                    count += 1
                else:
                    compressed.append(f"({count}x {current_val})")
                    current_val = val
                    count = 1
            compressed.append(f"({count}x {current_val})")  # Add final group
            yak_summary = " ".join(compressed)
            print(f"Path YAK values: {yak_summary}")
    
    print(f"Best path found: {len(best_path)} nodes, length={max_length:,}bp")
    return best_path[::-1], max_length  # Reverse path to get correct order

def find_greedy_pairs(G, valid_source_nodes, hap, config, num_greedy_pairs=25):
    """
    Find greedy walk sink nodes for each given source node.
    
    Args:
        G: NetworkX graph
        valid_source_nodes: List of source nodes to start greedy walks from
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary with parameters
    
    Returns:
        list: List of (source, sink) tuples where sink is reached via greedy walk from source
    """
    greedy_pairs = []
    
    if len(valid_source_nodes) > num_greedy_pairs:
        sampled_source_nodes = random.sample(valid_source_nodes, num_greedy_pairs)
        print(f"Sampling {num_greedy_pairs} source nodes from {len(valid_source_nodes)} valid source nodes for greedy walks")
    else:
        sampled_source_nodes = valid_source_nodes
        print(f"Using all {len(valid_source_nodes)} source nodes for greedy walks")
    
    # Create a minimal config for greedy walk focused on single paths
    greedy_config = config.copy()
    greedy_config['sample_size'] = 1  # Only need one path per source
    greedy_config['sample_by_score'] = False  # Use the source node we specify
    
    for source_node in sampled_source_nodes:
        # Create a temporary graph starting from this source to force the walk to begin there
        # We'll modify the sample_edges function behavior by ensuring our source is selected
        
        # Create a subgraph that includes only edges reachable from this source
        temp_graph = G.copy()
        
        # Temporarily override the sample_edges approach by making this source have a high-scoring edge
        # This is a bit of a hack, but it leverages the existing greedy_walk infrastructure
        
        # Actually, let's use walk_graph directly since greedy_walk uses sampling
        # which might not start from our desired source node
        path, path_length, wrong_haplo_len, total_wrong_kmer_hits, _ = walk_graph(
            temp_graph, source_node, hap=hap
        )
        
        # If we have a valid path that ends at a sink node, add the pair
        if len(path) > 1:
            sink_node = path[-1]
            greedy_pairs.append((source_node, sink_node))
    
    return greedy_pairs

def identify_wrong_haplotype_subpaths(G, path, hap, wrong_hap_threshold):
    """
    Identify contiguous subpaths that contain only wrong haplotype or neutral nodes.
    
    Args:
        G: NetworkX graph
        path: List of nodes representing the path
        hap: Target haplotype ('m' or 'p')
        wrong_hap_threshold: Length threshold for considering a wrong haplotype subpath as problematic
    
    Returns:
        list: List of (start_idx, end_idx, subpath_length) tuples for problematic wrong haplotype subpaths
    """
    if hap is None or len(path) <= 1:
        return []
    
    problematic_subpaths = []
    current_subpath_start = None
    current_subpath_length = 0
    
    for i, node in enumerate(path):
        yak_val = G.nodes[node].get(f'yak_{hap}', 0)
        
        # Check if this node is wrong haplotype (-1) or neutral (0)
        if yak_val <= 0:
            # Start a new wrong haplotype subpath if not already in one
            if current_subpath_start is None:
                current_subpath_start = i
                current_subpath_length = G.nodes[node]['read_length']
            else:
                # Continue the current subpath
                if i > 0:
                    # Add edge length
                    current_subpath_length -= G[path[i-1]][path[i]]['overlap_length']
                current_subpath_length += G.nodes[node]['read_length']
        else:
            # This node is correct haplotype, end current subpath if exists
            if current_subpath_start is not None:
                # Check if the subpath exceeds threshold
                if current_subpath_length > wrong_hap_threshold:
                    problematic_subpaths.append((current_subpath_start, i-1, current_subpath_length))
                
                # Reset for next potential subpath
                current_subpath_start = None
                current_subpath_length = 0
    
    # Handle case where path ends with a wrong haplotype subpath
    if current_subpath_start is not None:
        if current_subpath_length > wrong_hap_threshold:
            problematic_subpaths.append((current_subpath_start, len(path)-1, current_subpath_length))
    
    return problematic_subpaths

def filter_wrong_haplotype_subpaths(G, path, hap, config):
    """
    Remove problematic wrong haplotype subpaths and return the longest remaining contiguous subpath.
    
    Args:
        G: NetworkX graph
        path: List of nodes representing the path
        hap: Target haplotype ('m' or 'p')
        wrong_hap_threshold: Length threshold for removing wrong haplotype subpaths
    
    Returns:
        tuple: (filtered_path, filtered_length) where filtered_path is the longest remaining subpath
    """
    wrong_hap_threshold = config['max_wrong_haplo']
    if hap is None or len(path) <= 1:
        # Calculate original path length
        path_length = 0
        for i in range(len(path)-1):
            path_length += G[path[i]][path[i+1]]['prefix_length']
        if path:
            path_length += G.nodes[path[-1]]['read_length']
        return path, path_length
    
    # Identify problematic subpaths
    problematic_subpaths = identify_wrong_haplotype_subpaths(G, path, hap, wrong_hap_threshold)
    
    if not problematic_subpaths:
        # No problematic subpaths, return original path
        path_length = 0
        for i in range(len(path)-1):
            path_length += G[path[i]][path[i+1]]['prefix_length']
        path_length += G.nodes[path[-1]]['read_length']
        return path, path_length
    
    # Create list of valid ranges (excluding problematic subpaths)
    valid_ranges = []
    last_end = -1
    
    for start_idx, end_idx, _ in problematic_subpaths:
        # Add the range before this problematic subpath
        if start_idx > last_end + 1:
            valid_ranges.append((last_end + 1, start_idx - 1))
        last_end = end_idx
    
    # Add the final range after the last problematic subpath
    if last_end < len(path) - 1:
        valid_ranges.append((last_end + 1, len(path) - 1))
    
    # Find the longest valid range
    if not valid_ranges:
        return [], 0
    
    longest_range = max(valid_ranges, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_range
    
    # Extract the longest valid subpath
    filtered_path = path[start_idx:end_idx + 1]
    
    # Calculate length of filtered path
    filtered_length = 0
    for i in range(len(filtered_path) - 1):
        filtered_length += G[filtered_path[i]][filtered_path[i+1]]['prefix_length']
    if filtered_path:
        filtered_length += G.nodes[filtered_path[-1]]['read_length']
    
    return filtered_path, filtered_length

def source_sink_walk_2(G, hap=None, config=None, graphic_preds=None, source_sink=False, chop_critical_hamming=None):
    """
    Find the best path in a DAG from a source node to a sink node using simplified selection.
    Stores length, dijkstra score, and log_prob_sum for each walk, then applies 
    additive or multiplicative formulas based on config path_selection.
    
    Args:
        G: NetworkX DAG
        hap: Target haplotype ('m' for maternal or 'p' for paternal)
        config: Configuration dictionary with parameters including 'path_selection'
        graphic_preds: Optional node predictions from graphic model
        source_sink: Whether to use all source-sink pairs or just greedy pairs
        chop_critical_hamming: Optional threshold for chopping critical regions. If not None, 
                              analyze paths for regions where edge weights exceed this threshold
                              and keep only the longest contiguous chunk without critical regions.
    
    Returns:
        tuple: (path, path_length) where path is list of nodes and path_length is the total length
    """
    
    # Find source nodes (nodes with no in-edges) and sink nodes (nodes with no out-edges)
    source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not source_nodes or not sink_nodes:
        print("No source or sink nodes found in the graph")
        return [], 0
    
    print(f"Found {len(source_nodes)} valid source nodes and {len(sink_nodes)} valid sink nodes")
    
    # Pre-filter to find valid pairs (not complement and has path)
    print("Pre-filtering to find valid source-sink pairs...")
    valid_pairs = []
    
    if source_sink:
        # Create all possible source-sink combinations and randomize order
        all_possible_pairs = [(s, t) for s in source_nodes for t in sink_nodes]
        random.shuffle(all_possible_pairs)
        
        # Check pairs in random order until we find 100 valid ones
        max_valid_pairs = 100
        total_checked = 0
        
        for source, sink in tqdm(all_possible_pairs, desc="Finding valid pairs"):
            total_checked += 1
            
            # Skip if source is complement of sink
            if source == sink^1:
                continue
            # Skip if no path exists
            if not nx.has_path(G, source, sink):
                continue
            
            # This is a valid pair
            valid_pairs.append((source, sink))
            
            # Stop if we have enough valid pairs
            if len(valid_pairs) >= max_valid_pairs:
                break
        
        print(f"Found {len(valid_pairs)} valid pairs after checking {total_checked} source-sink combinations")
    
    # Add greedy pairs (these should already be valid, but we'll filter them too)
    greedy_pairs = find_random_walk_pairs(G, hap, config)
    for source, sink in greedy_pairs:
        if (source, sink) not in valid_pairs:  # Avoid duplicates
            valid_pairs.append((source, sink))
    
    # No need to sample since we already limited to max_valid_pairs
    if source_sink:
        pairs_to_test = valid_pairs + greedy_pairs
    else:
        pairs_to_test = greedy_pairs
    
    print(f"Testing {len(pairs_to_test)} valid source-sink pairs")

    # Precompute edge weights once for all dijkstra calls (major optimization)
    if graphic_preds is not None:
        print("Precomputing graphic edge weights for dijkstra optimization...")
        precomputed_weights = precompute_graphic_edge_weights(G, graphic_preds, threshold=0.1)
    else:
        print("Precomputing edge weights for dijkstra optimization...")
        precomputed_weights = precompute_edge_weights(G, hap, kmer_count=True)

    # Track progress
    pbar = tqdm(total=len(pairs_to_test), desc="Finding best source-sink path")
    
    # Store all valid paths with their metrics
    all_candidates = []
    
    for source, sink in pairs_to_test:
        # No need to check validity again since we pre-filtered
        if graphic_preds is not None:
            path, dijkstra_length = default_dijkstra_graphic(G, source, sink, graphic_preds, threshold=config['graphic_epsilon'], precomputed_weights=precomputed_weights)
        else:
            path, dijkstra_length = default_dijkstra(G, source, sink, hap, kmer_count=True, precomputed_weights=precomputed_weights)
        
        # Chop critical regions if requested
        if chop_critical_hamming is not None and path:
            path, dijkstra_length = chop_critical_regions(G, path, precomputed_weights, chop_critical_hamming, debug=True)
            if not path:
                pbar.update(1)
                continue
        # Check for co mplement pairs in the path and chop if necessary
        original_path = path
        path, removed_pairs_count = remove_complement_pairs_from_path(path)
    
        if path is not None:
            # Skip single node paths
            if len(path) <= 1:
                pbar.update(1)
                continue
            
            #print(f"Processing valid path with {len(path)} nodes (source: {source}, sink: {sink})")
                            
            # Calculate actual path length (not weighted)
            path_length = 0
            for i in range(len(path) - 1):
                path_length += G[path[i]][path[i+1]]['prefix_length']
            
            # Add length of last node
            path_length += G.nodes[path[-1]]['read_length']
            
            # Calculate log_prob_sum
            log_prob_sum = 0
            log_prob_avg = 0
            dijkstra_score = 0
            """for i in range(len(path) - 1):
                edge_score = G[path[i]][path[i+1]]['score']
                sigmoid_score = 1 / (1 + np.exp(-edge_score))
                log_prob_sum += np.log(sigmoid_score)
            log_prob_avg = log_prob_sum/len(path)"""
            
            #print(f"path_length: {path_length}, dijkstra_cost: {dijkstra_length}, log_prob_sum: {log_prob_sum}")
            #print(f"logs: path_length: {np.log(path_length)}, dijkstra_cost: {np.log(dijkstra_length + 0.00001)}, log_prob_sum: {log_prob_sum}")
            # Apply path selection formula based on config
            alpha = 1
            beta = 1 
            gamma = 0.01

            if config['select_walk'] == 'additive':
                # Additive formula: combines all three metrics
                path_length_score = alpha * np.log(path_length)
                pred_score = beta * log_prob_avg
                dijkstra_score =  - gamma * np.log(dijkstra_length + 0.00001)
                score =  path_length_score + pred_score + dijkstra_score
            elif config['select_walk'] == 'multiplicative':
                # Multiplicative formula: multiplies normalized metrics
                score =  np.log(dijkstra_length) * log_prob_sum / np.log(path_length)
                #score =  np.log(dijkstra_length * 1000) * log_prob_avg / np.log(path_length)
            else:
                # Default to path length for other configurations
                score = path_length
            
            # Store this candidate: (path, path_length, dijkstra_length, score, dijkstra_score, log_prob_sum)
            all_candidates.append((path, path_length, dijkstra_length, score, dijkstra_score))

    
    pbar.update(1)
    
    pbar.close()
    

    # Filter candidates and select best one
    print(f"Found {len(all_candidates)} total candidates")
    if not all_candidates:
        best_path = []
        best_length = 0
    else:
        # Find the longest path length
        max_length = max(candidate[1] for candidate in all_candidates)
        filter_candidates = 'n_longest'  # Can be False, True, or 'n_longest'
        n_longest = max(1, len(all_candidates)//2)  # Ensure at least 1 candidate is selected
        len_thr_percentage = 0.9
        
        if filter_candidates == True:
            # Filter for paths with at least 90% of the longest length
            length_threshold = len_thr_percentage * max_length
            filtered_candidates = [candidate for candidate in all_candidates if candidate[1] >= length_threshold]
            
            # Among filtered candidates, select the one with highest score
            best_candidate = max(filtered_candidates, key=lambda x: x[3])  # x[3] is score
            best_path, best_length, dijkstra_length, best_score, dijkstra_score = best_candidate
            
            print(f"Found {len(all_candidates)} total paths, {len(filtered_candidates)} paths >= 90% of max length ({max_length:,}bp)")
            print(f"Selected path with length {best_length:,}bp (score: {best_score:.2f}, dijkstra_score: {dijkstra_score:.2f}, dijkstra_length: {dijkstra_length:,})")
        elif filter_candidates == 'n_longest':
            # Sort all candidates by length (descending) and take the top n
            all_candidates_by_length = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            top_n_candidates = all_candidates_by_length[:n_longest]
            
            # Among the top n longest paths, select the one with highest score
            best_candidate = max(top_n_candidates, key=lambda x: x[3])  # x[3] is score
            best_path, best_length, dijkstra_length, best_score, dijkstra_score = best_candidate
            
            print(f"Found {len(all_candidates)} total paths, filtered to top {len(top_n_candidates)} longest paths")
            print(f"Selected path with length {best_length:,}bp (score: {best_score:.2f}, dijkstra_score: {dijkstra_score:.2f}, dijkstra_length: {dijkstra_length:,})")
        else:
            # No filtering, just select the best candidate by score
            best_candidate = max(all_candidates, key=lambda x: x[3])  # x[3] is score
            best_path, best_length, dijkstra_length, best_score, dijkstra_score = best_candidate
            
            print(f"Found {len(all_candidates)} total paths")
            print(f"Selected path with length {best_length:,}bp (score: {best_score:.2f}, dijkstra_score: {dijkstra_score:.2f}, dijkstra_length: {dijkstra_length:,})")
    
    return best_path, best_length

def remove_complement_pairs_from_path(path):
    """
    Remove complement pairs from a path and return the longest contiguous subpath without complement issues.
    
    Args:
        path: List of nodes representing the path
    
    Returns:
        tuple: (chopped_path, removed_pairs_count) where chopped_path is the longest valid subpath
               and removed_pairs_count is the number of complement pairs that were found
    """
    if len(path) <= 1:
        return path, 0
    
    # Find all complement pairs in the path
    complement_pairs = []
    for i in range(len(path)):
        for j in range(i+1, len(path)):
            if path[i] == path[j]^1:
                complement_pairs.append((i, j))
    
    if not complement_pairs:
        return path, 0
    
    # If we have complement pairs, we need to find the longest contiguous subpath
    # that doesn't contain any complement pairs
    
    # Create a set of all indices that are part of complement pairs
    problematic_indices = set()
    for i, j in complement_pairs:
        problematic_indices.add(i)
        problematic_indices.add(j)
    
    # Find all contiguous ranges that don't contain problematic indices
    valid_ranges = []
    start = 0
    
    for i in range(len(path)):
        if i in problematic_indices:
            # End current range if it exists
            if i > start:  
                valid_ranges.append((start, i-1))
            start = i + 1
    
    # Add final range if it exists
    if start < len(path):
        valid_ranges.append((start, len(path)-1))
    
    # Find the longest valid range
    if not valid_ranges:
        # All nodes are problematic, return empty path
        return [], len(complement_pairs)
    
    longest_range = max(valid_ranges, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_range
    
    # Extract the longest valid subpath
    chopped_path = path[start_idx:end_idx + 1]
    
    return chopped_path, len(complement_pairs)

def find_greedy_walk_pairs(G, hap, config):
    """
    Find source-sink pairs by sampling edges and doing greedy walks from them.
    
    Args:
        G: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary with parameters
    
    Returns:
        list: List of (source, sink) tuples where sink is reached via greedy walk from source
    """
    greedy_pairs = []
    
    # Sample edges for greedy walks
    sampled_edges = sample_edges(G, hap, config.get('sample_size', 25), sampling_by_score=config.get('sample_by_score', False))
    print(f"Sampled {len(sampled_edges)} edges for greedy walks...")
    
    if not sampled_edges:
        return greedy_pairs
    
    # Check if we need to reverse the graph (same logic as other walk methods)
    src = sampled_edges[0][0]  # example node
    if src^1 not in G.nodes():
        reversed_graph = G.reverse()
        reversed = True
    else:
        reversed = False
        reversed_graph = G
        
    for walk_idx, (src, dst) in enumerate(tqdm(sampled_edges, desc="Trying greedy walks from sampled edges")):
        # Forward greedy walk
        forward_path, forward_length, forward_wrong_haplo_len, _, _ = walk_graph(
            G, dst, hap=hap, 
            max_wrong_haplo=config.get('max_wrong_haplo', 5),
            score_attr=config.get('gt_score', 'score'),
            random_mode=False,  # Use greedy mode
            penalty_wrong_hap=config.get('penalty_wrong_hap', 0.6),
            penalty_ambiguous=config.get('penalty_ambiguous', 0.05)
        )
        
        if not forward_path:
            continue
            
        if not reversed:
            src = src^1
            
        # Backward greedy walk  
        backward_path, backward_length, backward_wrong_haplo_len, _, _ = walk_graph(
            reversed_graph, src, hap=hap, 
            max_wrong_haplo=config.get('max_wrong_haplo', 5),
            score_attr=config.get('gt_score', 'score'),
            random_mode=False,  # Use greedy mode
            penalty_wrong_hap=config.get('penalty_wrong_hap', 0.6),
            penalty_ambiguous=config.get('penalty_ambiguous', 0.05)
        )
        
        if not backward_path:
            continue
            
        # Get the endpoints from the greedy walks
        if reversed:
            start_node = backward_path[-1]  # Last node of backward path
        else:
            start_node = backward_path[-1]^1  # Last node of backward path (complemented)
        
        end_node = forward_path[-1]  # Last node of forward path
        
        # Add this pair if it's valid and not already included
        if (start_node != end_node^1 and nx.has_path(G, start_node, end_node) and 
            (start_node, end_node) not in greedy_pairs):
            greedy_pairs.append((start_node, end_node))
    
    print(f"Found {len(greedy_pairs)} greedy walk pairs")
    return greedy_pairs

def find_random_walk_pairs(G, hap, config):
    """
    Find source-sink pairs by sampling edges and doing random walks from them.
    
    Args:
        G: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary with parameters
    
    Returns:
        list: List of (source, sink) tuples where sink is reached via random walk from source
    """
    random_pairs = []
    
    # Sample edges for random walks
    sampled_edges = sample_edges(G, hap, config['sample_size'], sampling_by_score=config['sample_by_score'])
    print(f"Sampled {len(sampled_edges)} edges for random walks...")
    
    if not sampled_edges:
        return random_pairs
    
    # Check if we need to reverse the graph (same logic as other walk methods)
    src = sampled_edges[0][0]  # example node
    if src^1 not in G.nodes():
        reversed_graph = G.reverse()
        reversed = True
    else:
        reversed = False
        reversed_graph = G
        
    for walk_idx, (src, dst) in enumerate(tqdm(sampled_edges, desc="Trying random walks from sampled edges")):
        # Forward random walk
        forward_path, forward_length, forward_wrong_haplo_len, _, _ = walk_graph(
            G, dst, hap=hap, 
            max_wrong_haplo=0,
            score_attr=config.get('gt_score', 'score'),
            random_mode=True,  # Use random mode
            penalty_wrong_hap=0,
            penalty_ambiguous=0
        )
        
        if not forward_path:
            continue
            
        if not reversed:
            src = src^1
            
        # Backward random walk  
        backward_path, backward_length, backward_wrong_haplo_len, _, _ = walk_graph(
            reversed_graph, src, hap=hap, 
            max_wrong_haplo=0,
            score_attr=config.get('gt_score', 'score'),
            random_mode=True,  # Use random mode
            penalty_wrong_hap=0,
            penalty_ambiguous=0
        )
        
        if not backward_path:
            continue
            
        # Get the endpoints from the random walks
        if reversed:
            start_node = backward_path[-1]  # Last node of backward path
        else:
            start_node = backward_path[-1]^1  # Last node of backward path (complemented)
        
        end_node = forward_path[-1]  # Last node of forward path
        
        # Add this pair if it's valid and not already included
        if (start_node != end_node^1 and nx.has_path(G, start_node, end_node) and 
            (start_node, end_node) not in random_pairs):
            random_pairs.append((start_node, end_node))
    
    print(f"Found {len(random_pairs)} random walk pairs")
    return random_pairs

def beam_search_no_wh(nx_graph, start_node, hap=None, score_attr='score', penalty_ambiguous=0.3, penalty_wrong_hap=0.1, beam_width=3, wrong_hap_threshold=100000, use_best_intermediate=False, visited=None, wrong_kmer_fraction=None):
    """Find a path through the graph using beam search that terminates paths when they would be chopped due to wrong haplotype subpaths.
    
    Args:
        nx_graph: NetworkX graph
        start_node: Node to start path from
        hap: Target haplotype ('m' or 'p' or None)
        score_attr: Edge attribute to use for scoring
        penalty_ambiguous: Penalty to apply to edges leading to ambiguous nodes
        penalty_wrong_hap: Penalty to apply to edges leading to wrong haplotype nodes
        beam_width: Number of candidate paths to maintain at each step (default: 3)
        wrong_hap_threshold: Length threshold for wrong haplotype subpaths (default: 1)
        use_best_intermediate: Whether to use best intermediate walk if better than final (default: True)
        visited: Set of already visited nodes to avoid conflicts (optional)
        wrong_kmer_fraction: Precomputed wrong kmer fraction for the graph (optional)
    
    Returns:
        tuple: (path, path_length, wrong_haplo_count, visited_set) where path is list of nodes, 
               path_length is sum of prefix_lengths, and visited_set is the set of visited nodes
    """
    print("This one: 'beam_search_no_wh' is discontinued for now")
    exit()
    if len(nx_graph) == 0 or start_node not in nx_graph:
        return [], 0, 0, set()
    
    # Check initial node haplotype status
    start_yak_val = nx_graph.nodes[start_node].get(f'yak_{hap}', 0) if hap else 1
    start_is_wrong_hap = (start_yak_val <= 0) if hap else False
    start_wh_length = nx_graph.nodes[start_node]['read_length'] if start_is_wrong_hap else 0
    
    # Initialize visited set
    if visited is None:
        initial_visited = {start_node, start_node ^ 1}
    else:
        initial_visited = visited.copy()
        initial_visited.add(start_node)
        initial_visited.add(start_node ^ 1)
    
    # Initialize beam with the start node
    # Each beam item contains: (path, visited_nodes, total_score, avg_score, path_length, wrong_haplo_count, edge_count, current_wh_subpath_length)
    beam = [(
        [start_node],                      # path
        initial_visited,                   # visited nodes (including complements)
        0,                                 # total score
        0,                                 # average score (will be used for ranking)
        0,                                 # path length
        0,                                 # wrong haplo count
        0,                                 # edge count (for averaging)
        start_wh_length                    # current wrong haplotype subpath length
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
            current_best = max(beam, key=lambda x: x[3])  # x[3] is avg_score
            if current_best[3] > best_intermediate_score:
                best_intermediate = current_best
                best_intermediate_score = current_best[3]
        
        # Process each candidate path in the current beam
        for path, visited_nodes, total_score, avg_score, path_length, wrong_haplo_len, edge_count, current_wh_length in beam:
            current_node = path[-1]
            
            # Get all possible next edges
            out_edges = list(nx_graph.out_edges(current_node, data=True))
            if not out_edges:
                # If this path can't be extended, keep it as a candidate in the next beam
                next_beam.append((path, visited_nodes, total_score, avg_score, path_length, wrong_haplo_len, edge_count, current_wh_length))
                continue
            
            # Filter out edges leading to visited nodes or their complements
            valid_edges = []
            for src, dst, data in out_edges:
                if dst not in visited_nodes:
                    valid_edges.append((src, dst, data))
            
            
            if not valid_edges:
                # If no valid edges, keep this path as a candidate
                next_beam.append((path, visited_nodes, total_score, avg_score, path_length, wrong_haplo_len, edge_count, current_wh_length))
                continue
            
            # Score all valid edges and check wrong haplotype constraints
            scored_edges = []
            for src, dst, data in valid_edges:
                edge_score = compute_beam_edge_score(nx_graph, src, dst, data, hap=hap,
                                                   beam_score_shift=0.5,
                                                   edge_penalty=100,  # Default penalty
                                                   kmer_penalty_factor=1,
                                                   graphic_preds=None,
                                                   use_hifiasm_result=False,
                                                   wrong_kmer_fraction=wrong_kmer_fraction)
                scored_edges.append((src, dst, data, edge_score, current_wh_length))
            
            # If no valid edges after filtering, keep this path as terminated
            if not scored_edges:
                next_beam.append((path, visited_nodes, total_score, avg_score, path_length, wrong_haplo_len, edge_count, current_wh_length))
                continue
            
            # Sort edges by score (descending)
            scored_edges.sort(key=lambda x: x[3], reverse=True)
            
            # Take top edges to expand this path
            for i, (src, dst, data, edge_score, new_wh_length) in enumerate(scored_edges):
                if i >= beam_width:
                    break
                
                # Calculate new wrong haplo count
                new_wrong_haplo_len = wrong_haplo_len
                if hap is not None:
                    dst_yak_score = nx_graph.nodes[dst].get(f'yak_{hap}', 0)
                    if dst_yak_score == -1:  # Wrong haplotype
                        new_wrong_haplo_len += 1
                
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
                # Calculate average score (average negative log probability)
                new_avg_score = new_total_score / new_edge_count if new_edge_count > 0 else 0
                
                # Add to candidates for next beam
                next_beam.append((new_path, new_visited, new_total_score, new_avg_score, new_path_length, new_wrong_haplo_len, new_edge_count, new_wh_length))
        
        # If no paths were extended, we're done
        if next_beam == beam:
            break
        
        # Select top beam_width candidates for next iteration
        # Sort by average score (descending) instead of total score
        next_beam.sort(key=lambda x: x[3], reverse=True)
        beam = next_beam[:beam_width]
    
    # If no valid paths found
    if not beam:
        return [], 0, 0, set()
    
    # Select the best path from final beam (highest average score)
    final_best_path, final_best_visited, _, final_best_avg_score, final_best_path_length, final_best_wrong_haplo_count, _, _ = max(beam, key=lambda x: x[3])
    
    # Compare final best with best intermediate walk (if enabled)
    if use_best_intermediate and best_intermediate is not None and best_intermediate_score > final_best_avg_score:
        # Use the best intermediate walk
        best_path, best_visited, _, _, best_path_length, best_wrong_haplo_count, _, _ = best_intermediate
        print(f"Beam search no_wh: Using best intermediate walk (score: {best_intermediate_score:.3f}) over final best (score: {final_best_avg_score:.3f})")
    else:
        # Use the final best path
        best_path = final_best_path
        best_visited = final_best_visited
        best_path_length = final_best_path_length
        best_wrong_haplo_count = final_best_wrong_haplo_count
        if use_best_intermediate:
            print(f"Beam search no_wh: Using final best walk (score: {final_best_avg_score:.3f}) over intermediate best (score: {best_intermediate_score:.3f})")
    
    # Add length of last node to path length
    if best_path:
        best_path_length += nx_graph.nodes[best_path[-1]]['read_length']
    
    # print(f"Beam search completed: {edges_filtered}/{total_edges_considered} edges filtered due to visited set")
    # print(f"Final path length: {len(best_path)}, Final visited set size: {len(best_visited)}")
    
    return best_path, best_path_length, best_wrong_haplo_count, best_visited

def test_beam_search_no_wh():
    """Simple test function to verify beam_search_no_wh works correctly."""
    # This is a placeholder test function - would need actual graph data to run
    print("beam_search_no_wh function has been added successfully!")
    print("Key features:")
    print("- Incorporates wrong haplotype subpath detection directly into beam search")
    print("- Terminates paths early when they would exceed wrong_hap_threshold")
    print("- More efficient than post-processing with filter_wrong_haplotype_subpaths")
    print("- Use walk_mode='beam_search_no_wh' to activate this mode")
    print("- Configure wrong_hap_threshold in config dict (default: 1)")

def identify_wrong_haplotype_subpaths_kmer(G, path, hap, wrong_hap_threshold):
    """
    Identify contiguous subpaths that contain only wrong haplotype or neutral nodes.
    This version estimates wrong haplotype kmer hits by multiplying path length by wrong kmer scores.
    
    Args:
        G: NetworkX graph
        path: List of nodes representing the path
        hap: Target haplotype ('m' or 'p')
        wrong_hap_threshold: Threshold for considering a wrong haplotype subpath as problematic (in estimated kmer hits)
    
    Returns:
        list: List of (start_idx, end_idx, subpath_kmer_hits) tuples for problematic wrong haplotype subpaths
    """
    if hap is None or len(path) <= 1:
        return []
    
    # Set up wrong haplotype identifier for kmer counting
    not_hap = 'm' if hap == 'p' else 'p'
    
    problematic_subpaths = []
    current_subpath_start = None
    current_subpath_kmer_hits = 0
    
    for i, node in enumerate(path):
        yak_val = G.nodes[node].get(f'yak_{hap}', 0)
        
        # Check if this node is wrong haplotype (-1) or neutral (0)
        if yak_val <= 0:
            # Start a new wrong haplotype subpath if not already in one
            if current_subpath_start is None:
                current_subpath_start = i
                # Calculate kmer hits for this node: length * wrong_kmer_score
                node_length = G.nodes[node]['read_length']
                wrong_kmer_score = G.nodes[node].get(f'kmer_count_{not_hap}', 0)
                current_subpath_kmer_hits = node_length * wrong_kmer_score
            else:
                # Continue the current subpath
                if i > 0:
                    # Subtract overlap length from the edge, then add current node contribution
                    overlap_length = G[path[i-1]][path[i]]['overlap_length']
                    effective_length = G.nodes[node]['read_length'] - overlap_length
                else:
                    effective_length = G.nodes[node]['read_length']
                
                wrong_kmer_score = G.nodes[node].get(f'kmer_count_{not_hap}', 0)
                current_subpath_kmer_hits += effective_length * wrong_kmer_score
        else:
            # This node is correct haplotype, end current subpath if exists
            if current_subpath_start is not None:
                # Check if the subpath exceeds threshold
                if current_subpath_kmer_hits > wrong_hap_threshold:
                    problematic_subpaths.append((current_subpath_start, i-1, current_subpath_kmer_hits))
                
                # Reset for next potential subpath
                current_subpath_start = None
                current_subpath_kmer_hits = 0
    
    # Handle case where path ends with a wrong haplotype subpath
    if current_subpath_start is not None:
        if current_subpath_kmer_hits > wrong_hap_threshold:
            problematic_subpaths.append((current_subpath_start, len(path)-1, current_subpath_kmer_hits))
    
    return problematic_subpaths

def filter_wrong_haplotype_subpaths_kmer(G, path, hap, config):
    """
    Remove problematic wrong haplotype subpaths using kmer hit estimation and return the longest remaining contiguous subpath.
    
    Args:
        G: NetworkX graph
        path: List of nodes representing the path
        hap: Target haplotype ('m' or 'p')
        config: Configuration dictionary containing 'max_wrong_haplo' threshold
    
    Returns:
        tuple: (filtered_path, filtered_length) where filtered_path is the longest remaining subpath
    """
    wrong_hap_threshold = config['max_wrong_haplo']
    if hap is None or len(path) <= 1:
        # Calculate original path length
        path_length = 0
        for i in range(len(path)-1):
            path_length += G[path[i]][path[i+1]]['prefix_length']
        if path:
            path_length += G.nodes[path[-1]]['read_length']
        return path, path_length
    
    # Identify problematic subpaths using kmer hit estimation
    problematic_subpaths = identify_wrong_haplotype_subpaths_kmer(G, path, hap, wrong_hap_threshold)
    
    if not problematic_subpaths:
        # No problematic subpaths, return original path
        path_length = 0
        for i in range(len(path)-1):
            path_length += G[path[i]][path[i+1]]['prefix_length']
        path_length += G.nodes[path[-1]]['read_length']
        return path, path_length
    
    # Create list of valid ranges (excluding problematic subpaths)
    valid_ranges = []
    last_end = -1
    
    for start_idx, end_idx, _ in problematic_subpaths:
        # Add the range before this problematic subpath
        if start_idx > last_end + 1:
            valid_ranges.append((last_end + 1, start_idx - 1))
        last_end = end_idx
    
    # Add the final range after the last problematic subpath
    if last_end < len(path) - 1:
        valid_ranges.append((last_end + 1, len(path) - 1))
    
    # Find the longest valid range
    if not valid_ranges:
        return [], 0
    
    longest_range = max(valid_ranges, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_range
    
    # Extract the longest valid subpath
    filtered_path = path[start_idx:end_idx + 1]
    
    # Calculate length of filtered path
    filtered_length = 0
    for i in range(len(filtered_path) - 1):
        filtered_length += G[filtered_path[i]][filtered_path[i+1]]['prefix_length']
    if filtered_path:
        filtered_length += G.nodes[filtered_path[-1]]['read_length']
    
    return filtered_path, filtered_length

def source_beam(nx_graph, hap, config, graphic_preds=None):
    """Find paths through the graph using beam search starting only from source nodes.
    
    Args:
        nx_graph: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
        config: Configuration dictionary with parameters
        graphic_preds: Optional dictionary of graphic predictions for nodes
    
    Returns:
        tuple: (best_path, best_length) where best_path is list of nodes and best_length is the total length
    """
    # Find source nodes (nodes with no incoming edges)
    source_nodes = [n for n in nx_graph.nodes() if nx_graph.in_degree(n) == 0]
    
    if not source_nodes:
        print("No source nodes found in the graph")
        return [], 0
    
    print(f"Found {len(source_nodes)} source nodes")
    
    # Filter out complement nodes if their counterpart is already included
    filtered_source_nodes = []
    used_nodes = set()
    
    for node in source_nodes:
        if node not in used_nodes:
            filtered_source_nodes.append(node)
            used_nodes.add(node)
            used_nodes.add(node ^ 1)  # Also mark complement as used
    
    print(f"After filtering complements: {len(filtered_source_nodes)} source nodes")
    
    # Sample nodes if we have more than the sample size
    if len(filtered_source_nodes) > config['sample_size']:
        selected_nodes = random.sample(filtered_source_nodes, config['sample_size'])
        print(f"Sampled {len(selected_nodes)} source nodes from {len(filtered_source_nodes)} available")
    else:
        selected_nodes = filtered_source_nodes
        print(f"Using all {len(selected_nodes)} filtered source nodes")
    
    if not selected_nodes:
        return [], 0
    
    best_length = 0
    best_walk = []
    best_penalized_length = 0
    
    # Run beam search from each selected source node
    for node in tqdm(selected_nodes, desc="Trying beam search from source nodes"):
        path, path_length, visited_nodes = walk_beamsearch(
            nx_graph, node, hap=hap,
            score_attr=config['gt_score'],
            beam_width=config.get('beam_width', 10),
            beam_score_shift=config.get('beam_score_shift', 0.5),
            edge_penalty=config.get('edge_penalty', 100),
            kmer_penalty_factor=config.get('kmer_penalty_factor', 1),
            use_best_intermediate=config['beam_intermediate'],
            graphic_preds=graphic_preds,
            use_hifiasm_result=config.get('use_hifiasm_result', False)
        )
        
        if not path:
            continue
        
        # Calculate actual path length
        actual_path_length = 0
        for i in range(len(path)-1):
            actual_path_length += nx_graph[path[i]][path[i+1]]['prefix_length']
        actual_path_length += nx_graph.nodes[path[-1]]['read_length']
        
        # Apply selection criteria
        if config['select_walk'] == 'haplo_penalty':
            # Beam search doesn't track wrong haplotype length, so no penalty applied
            penalized_path_length = actual_path_length
        elif config['select_walk'] == 'longest_no_whs':
            path, actual_path_length = filter_wrong_haplotype_subpaths(nx_graph, path, hap, config)
            penalized_path_length = actual_path_length
        elif config['select_walk'] == 'longest_no_whs_kmer':
            path, actual_path_length = filter_wrong_haplotype_subpaths_kmer(nx_graph, path, hap, config)
            penalized_path_length = actual_path_length
        else:
            penalized_path_length = actual_path_length
        
        # Update best path if this one is better
        if penalized_path_length > best_penalized_length:
            best_walk = path
            best_length = actual_path_length
            best_penalized_length = penalized_path_length
    
    print(f"Best beam search path found: {len(best_walk)} nodes, length={best_length:,}bp")
    
    return best_walk, best_length

def compute_malicious_edges(nx_graph, sampled_edge, hap=None):
    """
    Compute malicious edges based on hifiasm results.
    
    Args:
        nx_graph: NetworkX graph
        sampled_edge: The initially sampled edge (src, dst)
        hap: Target haplotype ('m' or 'p' or None)
    
    Returns:
        None: Modifies the graph in-place by setting malicious attribute on edges
    """
    print(f"Computing malicious edges based on hifiasm results...")
    
    # Initialize all edges as non-malicious
    for src, dst in nx_graph.edges():
        nx_graph[src][dst]['to_cut'] = 0
    
    # Initialize all nodes as not belonging to current contig
    for node in nx_graph.nodes():
        nx_graph.nodes[node]['current_contig_nodes'] = 0
    
    # Step 1: Find all nodes connected to the sampled edge via hifiasm_result=1 edges
    initial_src, initial_dst = sampled_edge
    current_contig_nodes = set()
    
    # Use BFS to find all nodes connected via hifiasm_result=1 edges
    queue = [initial_src, initial_dst]
    visited = set()
    
    while queue:
        current_node = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)
        current_contig_nodes.add(current_node)
        
        # Check all edges from this node
        for neighbor in nx_graph.neighbors(current_node):
            if neighbor not in visited:
                # Check if edge has hifiasm_result=1
                if nx_graph[current_node][neighbor]['score'] == 1:
                    queue.append(neighbor)
    
    # Mark current contig nodes
    for node in current_contig_nodes:
        nx_graph.nodes[node]['current_contig_nodes'] = 1
    
    print(f"Found {len(current_contig_nodes)} nodes in current contig")
    
    # Step 2: Find all other contig nodes (nodes that have at least one hifiasm_result=1 edge but are not in current contig)
    other_contig_nodes = set()
    all_nodes = set(nx_graph.nodes())
    remaining_nodes = all_nodes - current_contig_nodes
    
    # Find nodes that have at least one hifiasm_result=1 edge
    for node in remaining_nodes:
        has_hifiasm_edge = False
        # Check all edges from this node
        for neighbor in nx_graph.neighbors(node):
            # Check if edge has hifiasm_result=1
            if nx_graph[node][neighbor]['score'] == 1:
                has_hifiasm_edge = True
                break
        
        # Only add if this node has at least one hifiasm_result=1 edge
        if has_hifiasm_edge:
            other_contig_nodes.add(node)
    
    print(f"Found {len(other_contig_nodes)} nodes in other contigs")
    
    # Step 3: Mark malicious edges
    malicious_count = 0
    for src, dst in nx_graph.edges():
        # Check if this edge is not a hifiasm_result edge
        if nx_graph[src][dst]['score'] != 1:
            # Check if it connects to a node in other contig paths
            if dst in other_contig_nodes or src in other_contig_nodes:
                nx_graph[src][dst]['to_cut'] = 1
                malicious_count += 1
    
    print(f"Marked {malicious_count} edges as malicious")

def compute_malicious_edges_2(nx_graph):
    """
    Compute malicious edges based on hifiasm results - optimized version.
    Called once at the start instead of for every sampled edge.
    
    Args:
        nx_graph: NetworkX graph
        hap: Target haplotype ('m' or 'p' or None)
    
    Returns:
        None: Modifies the graph in-place by setting to_cut attribute on edges
    """
    print(f"Computing malicious edges (optimized version)...")
    
    # Initialize all edges as non-malicious
    for src, dst in nx_graph.edges():
        nx_graph[src][dst]['to_cut'] = 0
    
    # Step 1: Find all contig nodes and group them by contig
    contig_nodes = set()
    node_to_contig = {}  # Maps node to contig ID
    contig_counter = 0
    
    # Find all nodes that have at least one edge with score=1 (hifiasm edges)
    for node in nx_graph.nodes():
        has_hifiasm_edge = False
        # Check outgoing edges
        for neighbor in nx_graph.successors(node):
            if nx_graph[node][neighbor].get('score', 0) == 1:
                has_hifiasm_edge = True
                break
        
        # Check incoming edges if not found yet
        if not has_hifiasm_edge:
            for neighbor in nx_graph.predecessors(node):
                if nx_graph[neighbor][node].get('score', 0) == 1:
                    has_hifiasm_edge = True
                    break
        
        if has_hifiasm_edge:
            contig_nodes.add(node)
    
    print(f"Found {len(contig_nodes)} contig nodes")
    
    # Step 2: Group contig nodes into contigs using connected components of score=1 edges
    visited = set()
    
    for node in contig_nodes:
        if node in visited:
            continue
            
        # BFS to find all nodes in this contig (connected via score=1 edges)
        current_contig = set()
        queue = [node]
        
        while queue:
            current_node = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)
            current_contig.add(current_node)
            
            # Check all neighbors connected via score=1 edges (both directions)
            # Check outgoing edges (successors)
            for neighbor in nx_graph.successors(current_node):
                if (neighbor not in visited and 
                    neighbor in contig_nodes and
                    nx_graph[current_node][neighbor].get('score', 0) == 1):
                    queue.append(neighbor)
            
            # Check incoming edges (predecessors) 
            for neighbor in nx_graph.predecessors(current_node):
                if (neighbor not in visited and 
                    neighbor in contig_nodes and
                    nx_graph[neighbor][current_node].get('score', 0) == 1):
                    queue.append(neighbor)
        
        # Assign contig ID to all nodes in this contig
        for contig_node in current_contig:
            node_to_contig[contig_node] = contig_counter
        
        contig_counter += 1
    
    print(f"Found {contig_counter} distinct contigs")
    
    # Step 3: Check edges with score=0 for malicious behavior
    malicious_count = 0
    
    for src, dst in nx_graph.edges():
        edge_score = nx_graph[src][dst].get('score', 0)
        
        # Only check edges with score=0
        if edge_score == 0:
            # Check if dst is a contig node
            if dst in contig_nodes:
                dst_contig = node_to_contig[dst]
                
                # BFS from src node with maximum depth 3
                is_malicious = True
                visited_bfs = set()
                current_level = {src}
                
                for depth in range(1, 4):  # depths 1, 2, 3
                    next_level = set()
                    
                    # Explore all nodes at current level
                    for node in current_level:
                        if node in visited_bfs:
                            continue
                        visited_bfs.add(node)
                        
                        # Get all neighbors of this node
                        for neighbor in nx_graph.neighbors(node):
                            if neighbor not in visited_bfs:
                                next_level.add(neighbor)
                    
                    # Check if any nodes at this depth are in the same contig as dst
                    if len(next_level) > 1:  # Multiple nodes found at this step
                        for node in next_level:
                            if (node in contig_nodes and 
                                node_to_contig[node] == dst_contig):
                                # Found a node from the correct contig
                                is_malicious = False
                                break
                        
                        if not is_malicious:
                            break
                    
                    # Move to next level
                    current_level = next_level
                    
                    # If no more nodes to explore, break
                    if not current_level:
                        break
                
                # Mark edge as malicious if no path to correct contig was found
                if is_malicious:
                    nx_graph[src][dst]['to_cut'] = 1
                    malicious_count += 1
    
    print(f"Marked {malicious_count} edges as malicious (score=0 edges)")

def compute_malicious_edges_3(nx_graph):
    """
    Compute malicious edges based on expanded contig sets with 1-hop neighborhoods.
    For each contig, create an expanded set that includes the contig nodes and their
    1-hop neighbors, except nodes that are 1-hop neighbors of other contigs.
    Mark edges as malicious if they don't connect nodes within the same expanded contig set.
    
    Args:
        nx_graph: NetworkX graph
    
    Returns:
        None: Modifies the graph in-place by setting to_cut attribute on edges
    """
    print(f"Computing malicious edges with expanded contig sets (version 3)...")
    
    # Initialize all edges as non-malicious
    for src, dst in nx_graph.edges():
        nx_graph[src][dst]['to_cut'] = 0
    
    # Step 1: Find all contigs (connected components of score=1 edges)
    contig_nodes = set()
    contigs = []  # List of sets, each containing nodes in one contig
    
    # Find all nodes that have at least one edge with score=1 (hifiasm edges)
    for node in nx_graph.nodes():
        has_hifiasm_edge = False
        for neighbor in nx_graph.neighbors(node):
            if nx_graph[node][neighbor]['score'] == 1:
                has_hifiasm_edge = True
                break
        
        if has_hifiasm_edge:
            contig_nodes.add(node)
    
    print(f"Found {len(contig_nodes)} contig nodes")
    
    # Group contig nodes into contigs using connected components of score=1 edges
    visited = set()
    
    for node in contig_nodes:
        if node in visited:
            continue
            
        # BFS to find all nodes in this contig (connected via score=1 edges)
        current_contig = set()
        queue = [node]
        
        while queue:
            current_node = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)
            current_contig.add(current_node)
            
            # Check all neighbors connected via score=1 edges (both directions)
            # Check outgoing edges (successors)
            for neighbor in nx_graph.successors(current_node):
                if (neighbor not in visited and 
                    neighbor in contig_nodes and
                    nx_graph[current_node][neighbor]['score'] == 1):
                    queue.append(neighbor)
            
            # Check incoming edges (predecessors) 
            for neighbor in nx_graph.predecessors(current_node):
                if (neighbor not in visited and 
                    neighbor in contig_nodes and
                    nx_graph[neighbor][current_node]['score'] == 1):
                    queue.append(neighbor)
        
        if current_contig:  # Only add non-empty contigs
            contigs.append(current_contig)
    
    print(f"Found {len(contigs)} distinct contigs")
    
    # Step 2: For each contig, find its 1-hop neighborhood
    contig_neighborhoods = []  # List of sets, each containing 1-hop neighbors of one contig
    
    for i, contig in enumerate(contigs):
        neighborhood = set()
        
        # Find all 1-hop neighbors of nodes in this contig
        for node in contig:
            # Get both successors (outgoing edges) and predecessors (incoming edges)
            for neighbor in nx_graph.successors(node):
                if neighbor not in contig:  # Don't include nodes already in the contig
                    neighborhood.add(neighbor)
            for neighbor in nx_graph.predecessors(node):
                if neighbor not in contig:  # Don't include nodes already in the contig
                    neighborhood.add(neighbor)
        
        # Filter neighborhood to only include nodes that have both predecessors and successors in the contig
        filtered_neighborhood = set()
        for neighbor in neighborhood:
            has_predecessor_in_contig = any(pred in contig for pred in nx_graph.predecessors(neighbor))
            has_successor_in_contig = any(succ in contig for succ in nx_graph.successors(neighbor))
            
            if has_predecessor_in_contig and has_successor_in_contig:
                filtered_neighborhood.add(neighbor)
        
        neighborhood = filtered_neighborhood
        
        contig_neighborhoods.append(neighborhood)
        #print(f"Contig {i}: {len(contig)} nodes, {len(neighborhood)} 1-hop neighbors")
    
    # Step 3: Resolve conflicts - remove nodes that are 1-hop neighbors of multiple contigs
    # Create a mapping of node -> list of contig indices it's a neighbor of
    node_to_contig_neighbors = {}
    
    for i, neighborhood in enumerate(contig_neighborhoods):
        for node in neighborhood:
            if node not in node_to_contig_neighbors:
                node_to_contig_neighbors[node] = []
            node_to_contig_neighbors[node].append(i)
    
    # Remove conflicted nodes from neighborhoods
    conflicted_nodes = 0
    for node, contig_indices in node_to_contig_neighbors.items():
        if len(contig_indices) > 1:  # Node is neighbor of multiple contigs
            conflicted_nodes += 1
            # Remove this node from all neighborhoods
            for contig_idx in contig_indices:
                contig_neighborhoods[contig_idx].discard(node)
    
    print(f"Removed {conflicted_nodes} conflicted nodes from neighborhoods")
    
    # Step 4: Create expanded contig sets (original contig + filtered 1-hop neighbors)
    expanded_contig_sets = []
    node_to_contig_set = {}  # Maps node to contig set index
    
    for i, (contig, neighborhood) in enumerate(zip(contigs, contig_neighborhoods)):
        expanded_set = contig.union(neighborhood)
        expanded_contig_sets.append(expanded_set)
        
        # Map each node to its contig set
        for node in expanded_set:
            node_to_contig_set[node] = i
        
        #print(f"Expanded contig set {i}: {len(expanded_set)} nodes ({len(contig)} original + {len(neighborhood)} neighbors)")
    
    # Step 5: Mark edges as malicious if they don't connect nodes within the same expanded contig set
    malicious_count = 0
    total_edges = 0
    
    for src, dst in nx_graph.edges():
        total_edges += 1
        
        # Check if both nodes belong to the same expanded contig set
        src_contig = node_to_contig_set.get(src, None)
        dst_contig = node_to_contig_set.get(dst, None)
        
        # Edge is malicious if:
        # 1. At least one node doesn't belong to any contig set, OR
        # 2. Nodes belong to different contig sets
        if src_contig is None and dst_contig is not None:
            #if src_contig is None or dst_contig is None or src_contig != dst_contig:

            nx_graph[src][dst]['to_cut'] = 1
            malicious_count += 1
        else:
            nx_graph[src][dst]['to_cut'] = 0
    
    print(f"Marked {malicious_count} out of {total_edges} edges as malicious")
    print(f"Non-malicious edges: {total_edges - malicious_count}")

def compute_classification_metrics(nx_graph, prediction_attr='to_cut', ground_truth_attr='malicious'):
    """
    Compute binary classification metrics between predicted and ground truth edge attributes.
    
    Args:
        nx_graph: NetworkX graph with edge attributes
        prediction_attr: Name of edge attribute containing predictions (default: 'to_cut')
        ground_truth_attr: Name of edge attribute containing ground truth labels (default: 'malicious')
    
    Returns:
        dict: Dictionary containing classification metrics:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN) 
            - f1_score: 2 * precision * recall / (precision + recall)
            - accuracy: (TP + TN) / (TP + TN + FP + FN)
            - specificity: TN / (TN + FP)
            - confusion_matrix: dict with TP, TN, FP, FN counts
            - total_edges: total number of edges evaluated
    """
    
    # Extract predictions and ground truth from all edges
    predictions = []
    ground_truth = []
    
    for src, dst, data in nx_graph.edges(data=True):
        pred = data.get(prediction_attr, 0)  # Default to 0 if attribute missing
        truth = data.get(ground_truth_attr, 0)  # Default to 0 if attribute missing
        
        predictions.append(pred)
        ground_truth.append(truth)
    
    if not predictions:
        print("No edges found in graph")
        return {}
    
    # Convert to binary values (in case they're not already 0/1)
    predictions = [1 if p > 0 else 0 for p in predictions]
    ground_truth = [1 if t > 0 else 0 for t in ground_truth]
    
    # Compute confusion matrix components
    tp = sum(1 for p, t in zip(predictions, ground_truth) if p == 1 and t == 1)  # True Positives
    tn = sum(1 for p, t in zip(predictions, ground_truth) if p == 0 and t == 0)  # True Negatives
    fp = sum(1 for p, t in zip(predictions, ground_truth) if p == 1 and t == 0)  # False Positives
    fn = sum(1 for p, t in zip(predictions, ground_truth) if p == 0 and t == 1)  # False Negatives
    
    total = tp + tn + fp + fn
    
    # Compute metrics with safe division (avoid division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Prepare results
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'specificity': specificity,
        'confusion_matrix': {
            'TP': tp,
            'TN': tn, 
            'FP': fp,
            'FN': fn
        },
        'total_edges': total
    }
    
    # Print summary
    print(f"\n=== Classification Metrics ===")
    print(f"Total edges evaluated: {total}")
    print(f"Confusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"")
    print(f"Metrics:")
    print(f"  Precision:    {precision:.4f} (TP / (TP + FP))")
    print(f"  Recall:       {recall:.4f} (TP / (TP + FN))")
    print(f"  F1-Score:     {f1_score:.4f}")
    print(f"  Accuracy:     {accuracy:.4f} ((TP + TN) / Total)")
    print(f"  Specificity:  {specificity:.4f} (TN / (TN + FP))")
    
    # Additional interpretation
    positive_rate = (tp + fn) / total if total > 0 else 0
    predicted_positive_rate = (tp + fp) / total if total > 0 else 0
    print(f"")
    print(f"Data Distribution:")
    print(f"  Actual positive rate:    {positive_rate:.4f} ({tp + fn}/{total})")
    print(f"  Predicted positive rate: {predicted_positive_rate:.4f} ({tp + fp}/{total})")
    print(f"===============================\n")
    
    return results

def evaluate_edge_predictions(nx_graph, prediction_attr='to_cut', ground_truth_attr='malicious', 
                            detailed=False):
    """
    Wrapper function to evaluate edge predictions with optional detailed analysis.
    
    Args:
        nx_graph: NetworkX graph with edge attributes
        prediction_attr: Name of edge attribute containing predictions (default: 'to_cut')
        ground_truth_attr: Name of edge attribute containing ground truth labels (default: 'malicious')
        detailed: If True, show per-edge analysis for misclassified edges (default: False)
    
    Returns:
        dict: Classification metrics from compute_classification_metrics
    """
    
    # Check if required attributes exist
    has_pred_attr = any(prediction_attr in data for _, _, data in nx_graph.edges(data=True))
    has_truth_attr = any(ground_truth_attr in data for _, _, data in nx_graph.edges(data=True))
    
    if not has_pred_attr:
        print(f"Warning: Prediction attribute '{prediction_attr}' not found in any edges")
    if not has_truth_attr:
        print(f"Warning: Ground truth attribute '{ground_truth_attr}' not found in any edges")
    
    # Compute basic metrics
    results = compute_classification_metrics(nx_graph, prediction_attr, ground_truth_attr)
    
    if detailed and results:
        print(f"\n=== Detailed Edge Analysis ===")
        
        # Show examples of misclassified edges
        fp_edges = []
        fn_edges = []
        
        for src, dst, data in nx_graph.edges(data=True):
            pred = 1 if data.get(prediction_attr, 0) > 0 else 0
            truth = 1 if data.get(ground_truth_attr, 0) > 0 else 0
            
            if pred == 1 and truth == 0:  # False Positive
                fp_edges.append((src, dst, data))
            elif pred == 0 and truth == 1:  # False Negative
                fn_edges.append((src, dst, data))
        
        # Show first 5 false positives
        if fp_edges:
            print(f"\nFirst 5 False Positives (predicted=1, actual=0):")
            for i, (src, dst, data) in enumerate(fp_edges[:5]):
                print(f"  {i+1}. Edge {src}->{dst}: pred={data.get(prediction_attr, 0)}, truth={data.get(ground_truth_attr, 0)}")
        
        # Show first 5 false negatives  
        if fn_edges:
            print(f"\nFirst 5 False Negatives (predicted=0, actual=1):")
            for i, (src, dst, data) in enumerate(fn_edges[:5]):
                print(f"  {i+1}. Edge {src}->{dst}: pred={data.get(prediction_attr, 0)}, truth={data.get(ground_truth_attr, 0)}")
        
        print(f"===============================\n")
    
    return results

def test_wrong_kmer_fraction():
    """
    Test function to verify the wrong kmer fraction computation.
    """
    import networkx as nx
    
    # Create a simple test graph
    G = nx.DiGraph()
    
    # Add nodes with different wrong kmer counts and read lengths
    G.add_node(0, read_length=1000, kmer_count_m=10, kmer_count_p=5)
    G.add_node(1, read_length=2000, kmer_count_m=20, kmer_count_p=10)
    G.add_node(2, read_length=1500, kmer_count_m=15, kmer_count_p=7)
    
    # Add edges
    G.add_edge(0, 1, score=0.8, overlap_length=100, prefix_length=900)
    G.add_edge(1, 2, score=0.9, overlap_length=150, prefix_length=1350)
    
    # Test wrong kmer fraction computation for maternal haplotype
    wrong_kmer_fraction_m = compute_graph_wrong_kmer_fraction(G, 'm')
    print(f"Wrong kmer fraction for maternal haplotype: {wrong_kmer_fraction_m:.6f}")
    
    # Test wrong kmer fraction computation for paternal haplotype
    wrong_kmer_fraction_p = compute_graph_wrong_kmer_fraction(G, 'p')
    print(f"Wrong kmer fraction for paternal haplotype: {wrong_kmer_fraction_p:.6f}")
    
    # Expected values:
    # For maternal: total wrong kmers = 5+10+7 = 22 (kmer_count_p values), total length = 1000+2000+1500 = 4500
    # Expected fraction = 22/4500 = 0.004889
    # For paternal: total wrong kmers = 10+20+15 = 45 (kmer_count_m values), total length = 4500
    # Expected fraction = 45/4500 = 0.01
    
    expected_m = 22 / 4500  # Wrong kmers for maternal = paternal kmer counts
    expected_p = 45 / 4500  # Wrong kmers for paternal = maternal kmer counts
    
    print(f"Expected maternal fraction: {expected_m:.6f}")
    print(f"Expected paternal fraction: {expected_p:.6f}")
    
    # Test edge scoring with wrong kmer fraction
    edge_data = G[0][1]
    score_with_fraction = compute_beam_edge_score(G, 0, 1, edge_data, hap='m', 
                                                wrong_kmer_fraction=wrong_kmer_fraction_m)
    score_without_fraction = compute_beam_edge_score(G, 0, 1, edge_data, hap='m', 
                                                   wrong_kmer_fraction=None)
    
    print(f"Edge score with wrong kmer fraction: {score_with_fraction:.6f}")
    print(f"Edge score without wrong kmer fraction: {score_without_fraction:.6f}")
    
    return True


if __name__ == "__main__":
    test_wrong_kmer_fraction()



