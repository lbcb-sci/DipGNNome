import numpy as np 
import networkx as nx
from tqdm import tqdm
#import split_chrom

def create_probabilistic_graph(G, prob_threshold=0.9, score_attr='score', diploid=False, higher_means_correct=True):
    """
    Create a graph where outgoing edges are filtered based on their cumulative softmax probability.
    
    Args:
        G: Input graph
        prob_threshold: Probability threshold for keeping edges
        score_attr: Edge attribute containing the score
        diploid: Whether to consider diploid mode (complementary edges)
        higher_means_correct: If True, higher scores mean better edges.
                             If False (default), lower scores mean better edges.
    """
    def softmax(scores):
        """Compute softmax probabilities for a list of scores."""
        # If higher_means_correct is True, use scores directly
        # Otherwise negate scores since lower scores should get higher probabilities
        if higher_means_correct:
            scores_for_softmax = np.array(scores)
        else:
            scores_for_softmax = -np.array(scores)  # Negate to invert relationship
            
        exp_scores = np.exp(scores_for_softmax - np.max(scores_for_softmax))  # Subtract max for numerical stability
        return exp_scores / exp_scores.sum()
    
    # Create a new graph with the same nodes
    filtered_graph = nx.DiGraph()
    filtered_graph.add_nodes_from(G.nodes(data=True))
    
    # For each node, compute probabilities and filter edges
    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        if not out_edges:  # Skip nodes without outgoing edges
            continue
            
        # Get scores and compute softmax probabilities
        scores = np.array([edge[2][score_attr] for edge in out_edges])
        probs = softmax(scores)  # Now higher probs correspond to better scores

        # Sort edges by probability (descending)
        sorted_indices = np.argsort(probs)[::-1]  # Sort descending to get best edges first
        sorted_edges = [out_edges[i] for i in sorted_indices]
        sorted_probs = probs[sorted_indices]
        
        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Find how many edges to keep based on probability threshold (top-p sampling)
        # We want the smallest set of edges whose cumulative probability >= prob_threshold
        indices_above_threshold = np.where(cumulative_probs >= prob_threshold)[0]
        if len(indices_above_threshold) == 0:
            # If no single edge reaches the threshold, keep all edges
            edges_to_keep = list(range(len(sorted_edges)))
        else:
            # Keep edges up to and including the first one that reaches the threshold
            last_idx = indices_above_threshold[0]
            edges_to_keep = list(range(last_idx + 1))
        
        if diploid:
            # Get yak_m values of all edges we're keeping
            kept_yak_values = [G.nodes[sorted_edges[i][1]]['yak_m'] for i in edges_to_keep]
            best_edge_target_yak = G.nodes[sorted_edges[0][1]]['yak_m']
            
            # Only look for complementary edge if:
            # 1. Best edge has definite haplotype (1 or -1)
            # 2. No kept edge has complementary or neutral haplotype
            needs_complement = (best_edge_target_yak in [1, -1] and 
                             not any((yak <= 0 if best_edge_target_yak == 1 else yak >= 0) 
                                   for yak in kept_yak_values[1:]))  # Skip first edge when checking
            
            if needs_complement:
                # Look for the best edge with complementary yak_m value
                best_complement_idx = None
                if higher_means_correct:
                    best_complement_score = float('-inf')  # Initialize with worst possible score
                else:
                    best_complement_score = float('inf')  # Initialize with worst possible score
                
                for i, edge in enumerate(sorted_edges):
                    if i not in edges_to_keep:  # Only look at edges we haven't already decided to keep
                        target_yak = G.nodes[edge[1]]['yak_m']
                        
                        # If best edge was yak_m=1, accept yak_m=0 or -1
                        # If best edge was yak_m=-1, accept yak_m=0 or 1
                        if ((best_edge_target_yak == 1 and target_yak <= 0) or 
                            (best_edge_target_yak == -1 and target_yak >= 0)):
                            # Check if this is the best complementary edge we've found so far
                            edge_score = edge[2][score_attr]
                            if (higher_means_correct and edge_score > best_complement_score) or \
                               (not higher_means_correct and edge_score < best_complement_score):
                                best_complement_score = edge_score
                                best_complement_idx = i
                
                # Add only the best complementary edge if one was found
                if best_complement_idx is not None:
                    edges_to_keep.append(best_complement_idx)
        
        # Add selected edges to the filtered graph
        for idx in edges_to_keep:
            edge = sorted_edges[idx]
            filtered_graph.add_edge(edge[0], edge[1], **edge[2])
    
    return filtered_graph


def cut_with_threshold(nx_graph, score_attr='score', prob_threshold=0.9, higher_means_correct=True):
    """
    Create a new graph containing only edges with scores above the given threshold.
    This keeps edges with high scores (removing low-scoring edges).
    
    Args:
        nx_graph: Input NetworkX graph
        score_attr: Edge attribute containing the score
        prob_threshold: Minimum score value to keep an edge
        higher_means_correct: If True, keep edges with scores >= threshold.
                             If False, keep edges with scores <= threshold.
        
    Returns:
        NetworkX graph with filtered edges
    """
    # Create a new graph with the same nodes
    filtered_graph = nx.DiGraph()
    filtered_graph.add_nodes_from(nx_graph.nodes(data=True))
    
    # Add edges with scores above/below the threshold based on higher_means_correct
    for u, v, data in nx_graph.edges(data=True):
        if higher_means_correct:
            # Higher scores are better, keep edges >= threshold
            if data[score_attr] >= prob_threshold:
                filtered_graph.add_edge(u, v, **data)
        else:
            # Lower scores are better, keep edges <= threshold
            if data[score_attr] <= prob_threshold:
                filtered_graph.add_edge(u, v, **data)
    
    return filtered_graph


def create_greedy_graph(G, score_attr='score', diploid=False):
    """
    Create a graph where only the highest-scored outgoing edge is kept for each node.
    In diploid mode, may keep two edges if they have different yak_m values.
    """
    # Create a new graph with the same nodes
    greedy_graph = nx.DiGraph()
    greedy_graph.add_nodes_from(G.nodes(data=True))
    
    # For each node, find and keep the best edge(s)
    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        if not out_edges:  # Skip nodes without outgoing edges
            continue
            
        # Sort edges by score in descending order (higher is better)
        sorted_edges = sorted(out_edges, key=lambda x: x[2][score_attr], reverse=True)
        
        if not diploid:
            # Non-diploid case: just keep the best edge
            best_edge = sorted_edges[0]
            greedy_graph.add_edge(best_edge[0], best_edge[1], **best_edge[2])
        else:
            # Diploid case: consider yak_m values
            best_edge = sorted_edges[0]
            best_edge_target_yak = G.nodes[best_edge[1]]['yak_m']
            
            # Always add the best edge
            greedy_graph.add_edge(best_edge[0], best_edge[1], **best_edge[2])
            
            # Only look for complementary edge if best edge has definite haplotype (1 or -1)
            if best_edge_target_yak in [1, -1]:
                # Look for next best edge with opposite or neutral yak_m
                for edge in sorted_edges[1:]:
                    target_yak = G.nodes[edge[1]]['yak_m']
                    
                    # If best edge was yak_m=1, accept yak_m=0 or -1
                    # If best edge was yak_m=-1, accept yak_m=0 or 1
                    if ((best_edge_target_yak == 1 and target_yak <= 0) or 
                        (best_edge_target_yak == -1 and target_yak >= 0)):
                        greedy_graph.add_edge(edge[0], edge[1], **edge[2])
                        break
    return greedy_graph

def is_comparable(edge_a, edge_b, edge_c, eps, overlap_lengths, read_lengths):
    # Modified to match DGL implementation parameter order
    # edge_a: n -> target
    # edge_b: target -> target2
    # edge_c: n -> target2

    node_a = edge_a[0]
    node_b = edge_b[0]
    
    length_a = read_lengths[node_a] - overlap_lengths[edge_a]
    length_b = read_lengths[node_b] - overlap_lengths[edge_b]
    length_c = read_lengths[node_a] - overlap_lengths[edge_c]
    a = length_a + length_b
    b = length_c

    return (b * (1 - eps) <= a <= b * (1 + eps)) or (a * (1 - eps) <= b <= a * (1 + eps))

def remove_transitives(g, eps=0.12, return_edges=False):
    """Remove transitive edges from a NetworkX graph"""
    candidates = {}
    marked_edges = []
    
    # Pre-fetch attributes for better performance
    overlap_lengths = nx.get_edge_attributes(g, "overlap_length")
    read_lengths = nx.get_node_attributes(g, "read_length")
    
    # Add progress bar for nodes
    for n in tqdm(g.nodes(), desc="Processing nodes"):
        # Get all successors (out edges) for current node
        successors = list(g.successors(n))
        for target in successors:
            candidates[target] = (n, target)
            
        for target in successors:
            # Get all successors of the target node
            target_successors = g.successors(target)
            for target2 in target_successors:
                if (target2 in candidates and
                    is_comparable(candidates[target2], # edge_a
                                (n, target), # edge_b
                                (target, target2), # edge_c
                                eps,
                                overlap_lengths,
                                read_lengths)):
                    marked_edges.append(candidates[target2])
                    
        candidates.clear()
    if return_edges:
        return marked_edges
    else:
        g.remove_edges_from(marked_edges)
        return g

def cut_chrom(nx_graph, reduction_param):
    print("implementation of cut_chrom depricated")
    exit()
    no_trans_graph = remove_transitives(nx_graph, eps=1.0)
    #edges_to_remove = split_chrom.find_inter_partition_edges(no_trans_graph, threshold=1e-30, resolution=1e-30)
    nx_graph.remove_edges_from(edges_to_remove)

    return nx_graph

def cut_fdl(nx_graph, cut_percentage):
    # Calculate edge lengths
    """print("create spring layout (force directed layout)...")
    pos = nx.spring_layout(nx_graph)
    print("..Done creating spring layout.")"""
    for layer, nodes in enumerate(nx.topological_generations(nx_graph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            nx_graph.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(nx_graph, subset_key="layer")

    edge_lengths = {}
    for u, v in nx_graph.edges():
        length = np.sqrt(sum((pos[u][i] - pos[v][i])**2 for i in range(len(pos[u]))))
        edge_lengths[(u, v)] = length

    # Sort edges by length (descending)
    sorted_edges = sorted(edge_lengths.items(), key=lambda x: x[1], reverse=True)

    # Calculate how many edges to remove (0.1% of total)
    num_edges_to_remove = max(1, int(cut_percentage * len(sorted_edges)))
    print(f"Removing {num_edges_to_remove} edges out of {len(sorted_edges)} total edges")

    # Remove the longest edges
    edges_to_remove = [edge for edge, _ in sorted_edges[:num_edges_to_remove]]
    
    """# Check strand_change attribute of removed edges
    strand_change_count = 0
    for u, v in edges_to_remove:
        if 'strand_change' in nx_graph[u][v] and nx_graph[u][v]['strand_change']:
            strand_change_count += 1
    
    print(f"Strand change edges among removed: {strand_change_count} out of {len(edges_to_remove)} ({strand_change_count/len(edges_to_remove)*100:.2f}%)")
    """
    nx_graph.remove_edges_from(edges_to_remove)

    print(f"Edges removed: {edges_to_remove}")
    print(f"Remaining edges: {list(nx_graph.edges())}")
    

    return nx_graph

def reduction(nx_graph, config, diploid, symmetry=True, reduce_complements=True):
    strategy = config['reduction']
    reduction_param = config['reduction_param']
    score_attr = config['reduction_score']
    cut_higher_than = config['cut_higher_than']
    
    if strategy == 'none':
        return nx_graph
    if strategy == "transitives" and reduction_param == 0:
        return nx_graph
    
    print(f"Reduce graph with strategy {strategy}")
    print(f"Before, graph: {nx_graph.number_of_nodes():,} nodes, {nx_graph.number_of_edges():,} edges")
    
    # Store the original edge count
    edges_before = nx_graph.number_of_edges()
    
    if strategy == "transitives":
        nx_graph = remove_transitives(nx_graph, eps=reduction_param)
    elif strategy == "cut_and_top_p":
        print("Cut and top p")
        print(f"Cutting with attribute {reduction_param} and top p with score {score_attr}")
        print(f"Make sure in {reduction_param} higher means wrong")
        nx_graph = cut_with_threshold(nx_graph, prob_threshold=cut_higher_than, score_attr='to_cut', higher_means_correct=False)
        nx_graph = create_probabilistic_graph(nx_graph, prob_threshold=reduction_param, diploid=diploid, score_attr='score')
    elif strategy == "cut_lower_threshold":
        nx_graph = cut_with_threshold(nx_graph, prob_threshold=reduction_param, score_attr=score_attr, higher_means_correct=True)
    elif strategy == "cut_higher_threshold":
        nx_graph = cut_with_threshold(nx_graph, prob_threshold=cut_higher_than, score_attr=score_attr, higher_means_correct=False)
    elif strategy == "top_p_graph":
        nx_graph = create_probabilistic_graph(nx_graph, prob_threshold=reduction_param, diploid=diploid, score_attr='score')
    elif strategy == "greedy_graph":
        nx_graph = create_greedy_graph(nx_graph, diploid=diploid, score_attr=score_attr)
    elif strategy == "cut_chrom":
        nx_graph = cut_chrom(nx_graph, reduction_param)
    elif strategy == 'fdl':
        nx_graph = cut_fdl(nx_graph, reduction_param)
    else: 
        raise ValueError(f"Invalid reduction strategy: {strategy}")
    

    if symmetry:
        # Count edges after reduction but before handling complements
        edges_after_reduction = nx_graph.number_of_edges()
        edges_removed = edges_before - edges_after_reduction
        
        # Count edges without complements after filtering
        edges_after = set(nx_graph.edges())
        edges_without_complement = [(src, dst) for (src, dst) in edges_after
                                    if (dst^1, src^1) not in edges_after]
        
        if reduce_complements:
            # Remove edges without complements
            print(f"\nRemoving edges without complements...")
            nx_graph.remove_edges_from(edges_without_complement)
            print(f"Removed {len(edges_without_complement):,} edges without complements")
        else:
            # Add missing complement edges
            print(f"\nReintroducing complement edges...")
            reintroduced_edges_count = 0
            for src, dst in edges_without_complement:
                comp_src = dst ^ 1
                comp_dst = src ^ 1
                # Copy edge data from original edge
                edge_data = nx_graph[src][dst].copy()
                nx_graph.add_edge(comp_src, comp_dst, **edge_data)
                reintroduced_edges_count += 1
            
            print(f"Reintroduced {reintroduced_edges_count:,} complement edges")

        print(f"- {nx_graph.number_of_nodes():,} total nodes")
        print(f"- {nx_graph.number_of_edges():,} total edges")
        
        # Print summary of edge changes
        print(f"Summary:")
        print(f"- Edges before reduction: {edges_before:,}")
        print(f"- Edges removed during reduction: {edges_removed:,}")
    #print(f"- Edges reintroduced as complements: {reintroduced_edges_count:,}")
    print(f"- Edges after all operations: {nx_graph.number_of_edges():,}")

    return nx_graph