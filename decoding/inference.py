import networkx as nx
import reduce_and_cut
import graph_walk
import mark_as_visited

def find_paths(nx_graph, hap, config, symmetry=True, graphic_preds=None):
    
    working_graph = nx_graph.copy()

    walks = []
    while working_graph.number_of_nodes() > config['min_path_length']:

        # Beam search is only walk strategy (greedy is beam_width=1)
        walk, walk_length = graph_walk.get_walk(working_graph, hap, config, walk_mode="beamsearch", graphic_preds=graphic_preds)
        # Adaptive minimum length check - be more lenient for short graphs or when no long paths exist
        min_length_threshold = config['min_walk_length']
        
        if walk_length < min_length_threshold:
            print(f"Path not added: length {walk_length:,} is less than {min_length_threshold:,.0f}")
            return walks

        # Only add path if it meets haplotype requirements
        if walk is None or len(walk) == 0:
            continue

        walks.append(walk)
        mark_as_visited_strategy = config['mark_as_visited']
        if mark_as_visited_strategy == 'shared_neighbors':
            visited_nodes = mark_as_visited.get_shared_neighbors(working_graph, walk)
        elif mark_as_visited_strategy == '1-hop':
            visited_nodes = mark_as_visited.get_1_hop_neighbors(working_graph, walk)
        elif mark_as_visited_strategy == '2-hop':
            visited_nodes = mark_as_visited.get_2_hop_neighbors(working_graph, walk)
        elif mark_as_visited_strategy == 'full_component':
            return walks
        else:
            raise ValueError(f"Invalid mark as visited strategy: {mark_as_visited_strategy}")
        working_graph.remove_nodes_from(visited_nodes)
    return walks

def find_components(nx_graph, min_component_size=1):
    print("Analyzing graph components...")
    components = list(nx.weakly_connected_components(nx_graph))
    original_component_count = len(components)
    print(f"   - Total components: {original_component_count:,}")

    components = [comp for comp in components if len(comp) >= min_component_size]
    
    print(f"   - Total components: {original_component_count:,}")
    if len(components) != original_component_count:
        removed_count = original_component_count - len(components)
        print(f"   - Removed {removed_count:,} components smaller than {min_component_size} nodes")
        print(f"   - Remaining components: {len(components):,}")
    
    # Component size distribution
    sizes = [len(comp) for comp in components]
    if sizes:
        print(f"   - Component size stats:")
        print(f"     * Minimum: {min(sizes):,} nodes")
        print(f"     * Maximum: {max(sizes):,} nodes")
        print(f"     * Average: {sum(sizes)/len(sizes):,.1f} nodes")
    
    # Check for complement components
    complement_components = []
    unique_components = []
    seen_sizes = {}  # Dictionary to track component sizes and their nodes
    # Sort components by size before processing
    for comp in components:
        comp_size = len(comp)
        comp_set = set(comp)
        
        # Check if we've seen a component of this size before
        if comp_size in seen_sizes:
            is_complement = False
            for prev_comp in seen_sizes[comp_size]:
                # Create set of complements for all nodes in current component
                comp_complements = {node ^ 1 for node in comp_set}
                # Check if prev_comp contains exactly these complements
                if comp_complements == prev_comp:
                    complement_components.append((prev_comp, comp_set))
                    is_complement = True
                    break
            if not is_complement:
                unique_components.append(comp_set)
                seen_sizes[comp_size].append(comp_set)
        else:
            # First component of this size
            unique_components.append(comp_set)
            seen_sizes[comp_size] = [comp_set]
    
    # Convert unique components back to list of sets
    components = [set(comp) for comp in unique_components]
    print(f"\nPartition Statistics:")
    print()
    print(f"Final components: {len(components)} (removed {len(complement_components)} complement pairs)")
    print("Component sizes:")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {len(comp)} nodes")
    print()

    return components

def get_walks(nx_graph, config, diploid=False, symmetry=True, graphic_scores=False):
    """
    Start the inference process using the provided NetworkX graph and reads.
    
    Args:
        nx_graph: NetworkX graph with scores
        config: Configuration dictionary
        diploid: Boolean indicating if diploid mode should be used
    """
    
    # Print all node and edge attribute names
    print("Node attributes:")
    if nx_graph.number_of_nodes() > 0:
        # Get a sample node to see what attributes are available
        sample_node = next(iter(nx_graph.nodes()))
        node_attrs = nx_graph.nodes[sample_node].keys()
        for attr in node_attrs:
            print(f"  - {attr}")
    else:
        print("  - No nodes in graph")
    
    print("\nEdge attributes:")
    if nx_graph.number_of_edges() > 0:
        # Get a sample edge to see what attributes are available
        sample_edge = next(iter(nx_graph.edges()))
        edge_attrs = nx_graph.edges[sample_edge].keys()
        for attr in edge_attrs:
            print(f"  - {attr}")
    else:
        print("  - No edges in graph")
    
    print(f"Starting inference with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
    
    initial_edges = nx_graph.number_of_edges()
    
    # reduce from translocation score using c_t threshold
    nx_graph = reduce_and_cut.reduction(nx_graph, config, diploid, symmetry, score_attr='to_cut', threshold=config['c_t'])
    # reduce from edge score using c_b threshold
    nx_graph = reduce_and_cut.reduction(nx_graph, config, diploid, symmetry, score_attr='score', threshold=config['c_b'], higher_means_correct=False)

    after_reduction = nx_graph.number_of_edges()
    print(f"Removed { initial_edges - after_reduction} edges during reduction ({initial_edges} -> {after_reduction})")
    
    components = find_components(nx_graph, min_component_size=config['min_component_size'])
    #components = [nx_graph]
    # Initialize walk lists
    if diploid:
        mat_walks = []
        pat_walks = []
    else:
        walks = []
    
    total_components = len(components)
    print(f"\nFinding paths in {total_components} components:")
    print(f"\nProcessing {total_components} components:")
    print("----------------------------------------")
    components_processed = 0


    for component in components:

        components_processed += 1
        print(f"\nComponent {components_processed}/{total_components} ({len(component)} nodes)")
        # Create subgraph for this component
        subG = nx_graph.subgraph(component).copy()
        subG = nx.DiGraph(subG)

        if graphic_scores:
            graphic_preds = nx.get_node_attributes(subG, 'graphic_score')
            graphic_preds_flipped = {k: -v for k, v in graphic_preds.items()}
        else:
            graphic_preds = None
            graphic_preds_flipped = None
            
        if diploid:
            print()
            print("----------------------------------------")
            print("Finding paths for paternal haplotype")
            comp_paths_p = find_paths(subG, hap='p', config=config, symmetry=symmetry, graphic_preds=graphic_preds)
            print()
            print("----------------------------------------")
            print("Finding paths for maternal haplotype")
            comp_paths_m = find_paths(subG, hap='m', config=config, symmetry=symmetry, graphic_preds=graphic_preds_flipped)
            mat_walks.extend(comp_paths_m)
            pat_walks.extend(comp_paths_p)
        else:
            comp_paths = find_paths(subG, hap=None, config=config, symmetry=symmetry)
            walks.extend(comp_paths)
        #break
    if diploid:
        return (mat_walks, pat_walks)
    else:
        return walks