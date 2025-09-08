import networkx as nx

import reduce_and_cut
import to_dag
import graph_walk
import improve_walk
import mark_as_visited
import dijkstra

def find_paths(nx_graph, hap, config, symmetry=True, graphic_preds=None):
    
    working_graph = nx_graph.copy()
    """yak_hap = f'yak_{hap}'
    yak_hap_attr = nx.get_node_attributes(working_graph, yak_hap)
    
    # Remove nodes with yak hap score < 0
    nodes_to_remove = [node for node, score in yak_hap_attr.items() if score < 0]
    working_graph.remove_nodes_from(nodes_to_remove)
    print(f"Removed {len(nodes_to_remove)} nodes with {yak_hap} score < 0")
    """

    walks = []
    while working_graph.number_of_nodes() > config['min_path_length']:

        if config['mark_tips']:
            visited_nodes = mark_as_visited.mark_tips(working_graph, hap, graphic_preds, threshold=0.1)
            working_graph.remove_nodes_from(visited_nodes)
            print(f"Removed {len(visited_nodes)} nodes with tips")
            # Remove small components
            components = list(nx.weakly_connected_components(working_graph))
            nodes_to_remove = set()
            for component in components:
                if len(component) < config['min_component_size']:
                    nodes_to_remove.update(component)
            working_graph.remove_nodes_from(nodes_to_remove)
            print(f"Removed {len(nodes_to_remove)} nodes from small components")
            print("--------------------------------")
    
        walk_strategy = config['walk']
        if walk_strategy == 'greedy':
            walk, walk_length = graph_walk.get_walk(working_graph, hap, config, walk_mode="greedy")
        elif walk_strategy == 'random':
            walk, walk_length = graph_walk.get_walk(working_graph, hap, config, walk_mode="random")
        elif walk_strategy == 'beam_search':
            walk, walk_length = graph_walk.get_walk(working_graph, hap, config, walk_mode="beamsearch", graphic_preds=graphic_preds)
        elif walk_strategy == 'source_beam':
            walk, walk_length = graph_walk.source_beam(working_graph, hap, config)
        elif walk_strategy == 'beam_search_no_wh':
            walk, walk_length = graph_walk.get_walk(working_graph, hap, config, walk_mode="beam_search_no_wh")
        elif walk_strategy == 'dag_longest':
            if len(working_graph.nodes) == 0 or len(working_graph.edges) == 0:
                return walks
            walk, walk_length = graph_walk.dag_longest_walk(working_graph, hap, config)
        elif walk_strategy == 'dag_longest_single':
            if len(working_graph.nodes) == 0 or len(working_graph.edges) == 0:
                return walks
            walk, walk_length = graph_walk.dag_longest_walk_single_strand(working_graph, hap, config)
        elif walk_strategy == 'source_sink':
            walk, walk_length = graph_walk.source_sink_walk(working_graph, hap, config, graphic_preds)
        elif walk_strategy == 'ss2':
            if graphic_preds is not None:
                walk, walk_length = graph_walk.source_sink_walk_2(working_graph, hap=None, config=config, graphic_preds=graphic_preds)
            else:
                walk, walk_length = graph_walk.source_sink_walk_2(working_graph, hap, config)
        elif walk_strategy == 'dijkstra_greedy':
            walk, walk_length = graph_walk.get_dijkstra_walk(working_graph, hap, config, graphic_preds)
        else:
            raise ValueError(f"Invalid walk strategy: {walk_strategy}")
    
        # Adaptive minimum length check - be more lenient for short graphs or when no long paths exist
        min_length_threshold = config['min_walk_length']
        
        # If graph is small, reduce the threshold proportionally
        """total_graph_length = sum(working_graph.nodes[n]['read_length'] for n in working_graph.nodes())
        if total_graph_length > 0:
            # If the entire graph is shorter than the threshold, use a fraction of total length
            adaptive_threshold = min(min_length_threshold, total_graph_length * 0.1)  # Use 10% of total graph length
            if adaptive_threshold < min_length_threshold:
                print(f"Reducing minimum length threshold from {min_length_threshold:,} to {adaptive_threshold:,.0f} for small graph")
                min_length_threshold = adaptive_threshold"""
        
        if walk_length < min_length_threshold:
            print(f"Path not added: length {walk_length:,} is less than {min_length_threshold:,.0f}")
            return walks
        
        """# Check for duplicate nodes or complement nodes in walk
        seen_nodes = set()
        for node in walk:
            # Check if node or its complement already seen
            if node in seen_nodes or (node ^ 1) in seen_nodes:
                raise ValueError(f"Invalid walk: Node {node} or its complement {node^1} appears multiple times")
            seen_nodes.add(node)"""
        

        improve_walk_strategy = config['improve_walk']

        if improve_walk_strategy == 'dijkstra':
            walk = improve_walk.min_wrong_haplo_dijkstra(working_graph, walk, hap, config)
        elif improve_walk_strategy == 'bellman_ford':
            walk = improve_walk.min_wrong_haplo_bellman_ford(working_graph, walk, hap, config)
        elif improve_walk_strategy == 'exploration':
            walk = improve_walk.improve_greedy_walk_by_exploration(working_graph, walk, hap, config)
        elif improve_walk_strategy == 'double_dfs':
            walk = improve_walk.improve_double_dfs(working_graph, walk, hap, config)
        elif improve_walk_strategy == 'bubble_dijkstra':
            walk = improve_walk.bubble_dijkstra(working_graph, walk, hap, config)
        elif improve_walk_strategy == 'none':
            pass
        else:
            raise ValueError(f"Invalid improve walk strategy: {improve_walk_strategy}")

        # Only add path if it meets haplotype requirements
        if walk is None or len(walk) == 0:
            continue
        """if not config['mark_tips']:
            yak_scores = nx.get_node_attributes(working_graph, f'yak_{hap}')
            if hap is not None and config['max_wrong_haplo_fraction'] > 0:
                # Calculate fraction of wrong haplotype nodes
                yak_scores = [yak_scores[n] for n in walk]
                wrong_haplo_count = sum(1 for score in yak_scores if score == -1)
                wrong_haplo_fraction = wrong_haplo_count / len(walk)
                if wrong_haplo_fraction > config['max_wrong_haplo_fraction']:
                    print(f"Path not added: {wrong_haplo_fraction*100:.1f}% wrong haplotype nodes exceeds {config['max_wrong_haplo_fraction']*100:.1f}% threshold")
                else:
                    walks.append(walk)
            else:
                # If no haplotype specified, just add the path
                walks.append(walk)
        else:
            walks.append(walk)
        """
        walks.append(walk)
        mark_as_visited_strategy = config['mark_as_visited']
        if mark_as_visited_strategy == 'shared_neighbors':
            visited_nodes = mark_as_visited.get_shared_neighbors(working_graph, walk)
        elif mark_as_visited_strategy == '1-hop':
            visited_nodes = mark_as_visited.get_1_hop_neighbors(working_graph, walk)
        elif mark_as_visited_strategy == '2-hop':
            visited_nodes = mark_as_visited.get_2_hop_neighbors(working_graph, walk)
        elif mark_as_visited_strategy == 'double_dijkstra':
            visited_nodes = mark_as_visited.double_dijkstra(working_graph, walk)
        elif mark_as_visited_strategy in ['dual_walks_1_hop', 'dual_walks_2_hop', 'dual_walks_shared']:
            source = walk[0]
            sink = walk[-1]
            if graphic_preds is not None:
                neg_graphic_preds = {k: -v for k, v in graphic_preds.items()}
                alt_walk, _ = dijkstra.graphic_dijkstra(working_graph, source, sink, neg_graphic_preds)
            else:
                alt_hap = 'm' if hap == 'p' else 'p'
                alt_walk, _ = dijkstra.dijkstra(working_graph, source, sink, alt_hap)
            if mark_as_visited_strategy == 'dual_walks_1_hop':
                visited_nodes = mark_as_visited.get_dual_walk_1_hop(working_graph, walk, alt_walk)
            elif mark_as_visited_strategy == 'dual_walks_2_hop':
                visited_nodes = mark_as_visited.get_dual_walk_2_hop(working_graph, walk, alt_walk)
            elif mark_as_visited_strategy == 'dual_walks_shared':
                visited_nodes = mark_as_visited.get_dual_walk_shared(working_graph, walk, alt_walk)
        elif mark_as_visited_strategy == 'full_component':
            return walks
        else:
            raise ValueError(f"Invalid mark as visited strategy: {mark_as_visited_strategy}")

        #break
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
    
    #nx_graph = reduce_and_cut.reduction(nx_graph, 'transitives', config['remove_transitives'], diploid, symmetry)
    #after_transitives = nx_graph.number_of_edges()
    #print(f"Removed {initial_edges - after_transitives} transitive edges ({initial_edges} -> {after_transitives})")

    nx_graph = reduce_and_cut.reduction(nx_graph, config, diploid, symmetry)
    after_reduction = nx_graph.number_of_edges()
    print(f"Removed { initial_edges - after_reduction} edges during {config['reduction']} reduction ({initial_edges} -> {after_reduction})")
    
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

        subG = to_dag.reduction(subG, config['to_dag'])

        """# Save component subgraph for debugging
        import pickle
        import os
        
        # Create debug directory if it doesn't exist
        debug_dir = "/home/schmitzmf/scratch/dags"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        # Save the subgraph with a unique name based on component number
        debug_file = os.path.join(debug_dir, f"symsyn_chr23.pkl")
        with open(debug_file, "wb") as f:
            pickle.dump(subG, f)
        exit()
        """
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