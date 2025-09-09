import numpy as np 
import networkx as nx
from tqdm import tqdm

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


def reduction(nx_graph, config, diploid, symmetry=True, reduce_complements=True):
    score_attr = config['reduction_score']
    cut_higher_than = config['cut_higher_than']
    
    print(f"Reduce graph with strategy {strategy}")
    print(f"Before, graph: {nx_graph.number_of_nodes():,} nodes, {nx_graph.number_of_edges():,} edges")
    
    # Store the original edge count
    edges_before = nx_graph.number_of_edges()
    nx_graph = cut_with_threshold(nx_graph, prob_threshold=cut_higher_than, score_attr=score_attr, higher_means_correct=False)

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