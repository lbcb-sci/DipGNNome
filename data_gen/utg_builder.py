import networkx as nx
from tqdm import tqdm
from Bio import SeqIO
import gzip
import pickle
import numpy as np


def is_comparable(g, edge_c, edge_a, edge_b, eps, prefix_lengths):
    
    a = prefix_lengths[edge_a] + prefix_lengths[edge_b]
    b = prefix_lengths[edge_c]

    return (b * (1 - eps) <= a <= b * (1 + eps)) or (a * (1 - eps) <= b <= a * (1 + eps))

def remove_transitive_edges(g, eps=0.12):
    """Remove transitive edges from a NetworkX graph - optimized version"""
    marked_edges = set()
    
    # Pre-fetch attributes for better performance
    prefix_lengths = nx.get_edge_attributes(g, "prefix_length")
    
    print(f"Starting transitive edge removal with eps={eps}")
    print(f"  Total edges before: {g.number_of_edges()}")
    
    # Add progress bar for nodes - sort nodes for deterministic behavior
    for n in tqdm(sorted(g.nodes()), desc="Processing nodes"):
        # Build candidates mapping: target_node -> edge_tuple for all outgoing edges from n
        candidates = {}
        for target in g.successors(n):
            candidates[target] = (n, target)
        
        # Check all paths of length 2 starting from n
        for intermediate in g.successors(n):
            # For each successor of the intermediate node
            for final_target in g.successors(intermediate):
                # Check if there's a direct edge from n to final_target (stored in candidates)
                if final_target in candidates:
                    # We have: n->intermediate->final_target AND n->final_target
                    # The direct edge n->final_target is potentially transitive
                    
                    edge_c = candidates[final_target]      # Direct edge (potentially transitive)
                    edge_a = (n, intermediate)             # First part of path
                    edge_b = (intermediate, final_target)  # Second part of path
                    
                    if is_comparable(g, edge_c, edge_a, edge_b, eps, prefix_lengths):
                        # Mark the direct edge as transitive
                        marked_edges.add(edge_c)

    # Before returning, filter out edges without complements
    marked_edges = {edge for edge in marked_edges if (edge[1]^1, edge[0]^1) in marked_edges}

    print(f"  Marked {len(marked_edges)} edges as transitive")
    print(f"  After complement filtering: {len(marked_edges)} edges")

    return marked_edges

def load_fasta(file_path, as_dict=True):
    """Load sequences from a FASTA/FASTQ file, handling both compressed and uncompressed files.
    Returns a dictionary mapping ids to sequences if as_dict=True, otherwise a list of SeqRecord objects."""
    filetype = "fasta" if file_path.endswith(('.fasta', '.fa', '.fasta.gz', '.fa.gz')) else "fastq"
    
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as handle:
            records = SeqIO.parse(handle, filetype)
            if as_dict:
                return {record.id: str(record.seq) for record in records}
            return list(records)
    else:
        records = SeqIO.parse(file_path, filetype)
        if as_dict:
            return {record.id: str(record.seq) for record in records}
        return list(records)

def save_fasta(sequences, output_path):
    """Save sequences to a FASTA file.
    
    Args:
        sequences: Dictionary mapping read IDs to sequence strings
        output_path: Path to save the FASTA file
    """
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    
    seq_records = []
    for read_id, sequence in sequences.items():
        seq_record = SeqRecord(Seq(sequence), id=str(read_id), description="")
        seq_records.append(seq_record)
    
    SeqIO.write(seq_records, output_path, "fasta")
    print(f"Saved {len(seq_records)} sequences to {output_path}")

def remove_self_and_complement_edges(g):
    """
    Remove self-loops (n->n) and edges to complements (n->n^1) from the graph.
    
    Args:
        g: NetworkX graph to clean
        
    Returns:
        Number of edges removed
    """
    edges_to_remove = []
    
    for u, v in g.edges():
        # Remove self-loops (n->n)
        if u == v:
            edges_to_remove.append((u, v))
            continue
            
        # Remove edges to complements (n->n^1)
        if u == (v ^ 1):
            edges_to_remove.append((u, v))
            continue
    
    if edges_to_remove:
        print(f"Removing {len(edges_to_remove)} self-loops and complement edges")
        g.remove_edges_from(edges_to_remove)
    
    return len(edges_to_remove)

def condense_unbranching_paths(g, read_seqs):
    """
    Fast merging: Find and merge all linear chains at once
    Also handles complement chains to maintain graph symmetry
    """
    simplified_g = g.copy()
    original_nodes = set(simplified_g.nodes())
    
    # Remove self-loops and complement edges before merging
    removed_edges = remove_self_and_complement_edges(simplified_g)
    if removed_edges > 0:
        print(f"Removed {removed_edges} problematic edges before chain merging")
    
    max_node_id = max(simplified_g.nodes())
    initial_nodes = simplified_g.number_of_nodes()
    
    # Pre-compute degrees once for efficiency
    in_degrees = dict(simplified_g.in_degree())
    out_degrees = dict(simplified_g.out_degree())
    
    # Find ALL chains at once instead of one by one
    all_chains = _find_all_linear_chains(simplified_g, in_degrees, out_degrees)
    
    if not all_chains:
        print("No chains found to merge")
        return simplified_g, 0, read_seqs, {}
    
    print(f"Found {len(all_chains)} chains to merge")
    
    # Filter out chains that would conflict with their complements
    valid_chains = _filter_valid_chain_pairs(simplified_g, all_chains)
    
    print(f"After filtering: {len(valid_chains)} valid chains to merge")
    
    if not valid_chains:
        return simplified_g, 0, read_seqs, {}
    
    # Track which old nodes are merged into which new nodes
    utg_to_old_nodes = {}
    
    # Merge all chains with progress bar
    nodes_removed = 0
    with tqdm(total=len(valid_chains), desc="Merging chains", unit="chain") as pbar:
        for i, chain in enumerate(valid_chains):
            # Get complement chain
            complement_chain = [n ^ 1 for n in reversed(chain)]
            
            # Track the mappings (keep original node ID creation logic)
            utg_to_old_nodes[max_node_id + 1 + i*2] = chain
            utg_to_old_nodes[max_node_id + 2 + i*2] = complement_chain
            
            # Merge both chains
            chain_nodes_removed = _merge_chain_fast(simplified_g, chain, read_seqs, max_node_id + 1 + i*2)
            complement_nodes_removed = _merge_chain_fast(simplified_g, complement_chain, read_seqs, max_node_id + 2 + i*2)
            
            nodes_removed += chain_nodes_removed + complement_nodes_removed
            
            # Update progress
            current_nodes = simplified_g.number_of_nodes()
            #pbar.set_description(f"Merging chains (nodes: {initial_nodes}→{current_nodes}, removed: {nodes_removed})")
            pbar.update(1)
    
    print(f"Chain merging completed: merged {len(valid_chains)} chain pairs, removed {nodes_removed} nodes")
    
    # Add nodes that weren't part of any chain to the mapping
    merged_nodes = set()
    for old_nodes in utg_to_old_nodes.values():
        merged_nodes.update(old_nodes)
    
    # Add single nodes that weren't merged (only from original graph)
    for node in simplified_g.nodes():
        if node in original_nodes and node not in merged_nodes and node not in utg_to_old_nodes:
            utg_to_old_nodes[node] = [node]
    
    return simplified_g, nodes_removed, read_seqs, utg_to_old_nodes


def _find_all_linear_chains(g, in_degrees, out_degrees):
    """
    Find ALL linear chains in the graph at once.
    A linear chain is a maximal path where all internal nodes have in_degree=1 and out_degree=1.
    """
    visited = set()
    chains = []
    
    # Get total nodes for progress tracking
    total_nodes = len(g.nodes())
    processed_nodes = 0
    
    print(f"Finding linear chains in graph with {total_nodes} nodes...")
    
    # Look for chain starts (nodes with in_degree != 1 but out_degree = 1)
    # or nodes that can start a chain
    for i, node in enumerate(g.nodes()):
        if node in visited:
            continue
            
        # Progress update every 1000 nodes
        if i % 1000 == 0:
            print(f"  Progress: {i}/{total_nodes} nodes processed, found {len(chains)} chains so far")
            
        # Skip if complement doesn't exist
        if (node ^ 1) not in g:
            continue
            
        # Try to build a chain starting from this node
        chain = _build_maximal_chain(g, node, in_degrees, out_degrees, visited)
        
        if chain and len(chain) > 1:
            # Mark all nodes in chain as visited
            visited.update(chain)
            chains.append(chain)
            processed_nodes += len(chain)
            
            # Debug info for very long chains
            if len(chain) > 100:
                print(f"  Found long chain: {len(chain)} nodes starting from {node}")
    
    print(f"Chain finding completed: {len(chains)} chains found, {processed_nodes} nodes in chains")
    return chains


def _build_maximal_chain(g, start_node, in_degrees, out_degrees, visited):
    """
    Build a maximal linear chain starting from start_node.
    Returns None if no valid chain can be built.
    
    Optimized version with cycle detection and safety limits.
    """
    # Safety limit to prevent infinite loops
    MAX_CHAIN_LENGTH = 10000
    
    # Track nodes visited during this chain building to detect cycles
    chain_visited = set()
    
    # Build chain backwards first (more efficient than inserting at beginning)
    backward_chain = []
    current = start_node
    
    # Extend backwards to find the true start
    while (in_degrees[current] == 1 and 
           len(backward_chain) < MAX_CHAIN_LENGTH):
        
        # Add current node to chain-specific visited set
        if current in chain_visited:
            print(f"Warning: Cycle detected during backward extension at node {current}")
            break
        chain_visited.add(current)
        backward_chain.append(current)
        
        predecessors = list(g.predecessors(current))
        if not predecessors:
            break
            
        prev_node = predecessors[0]
        
        # Check if prev_node can be part of chain
        if (out_degrees[prev_node] == 1 and 
            prev_node not in visited and 
            prev_node not in chain_visited and  # Prevent cycles within this chain
            prev_node != (current ^ 1)):  # Avoid complement edges
            current = prev_node
        else:
            break
    
    # Add the final node that couldn't be extended further
    if current not in chain_visited:
        backward_chain.append(current)
        chain_visited.add(current)
    
    # Now reverse to get forward order
    chain = list(reversed(backward_chain))
    
    # Extend forwards from the last node in the current chain
    current = chain[-1] if chain else start_node
    
    while (out_degrees[current] == 1 and 
           len(chain) < MAX_CHAIN_LENGTH):
        
        successors = list(g.successors(current))
        if not successors:
            break
            
        next_node = successors[0]
        
        # Check if next_node can be part of chain
        if (in_degrees[next_node] == 1 and 
            next_node not in visited and 
            next_node not in chain_visited and  # Prevent cycles within this chain
            next_node != (current ^ 1)):  # Avoid complement edges
            
            chain.append(next_node)
            chain_visited.add(next_node)
            current = next_node
        else:
            break
    
    # Safety check for extremely long chains
    if len(chain) >= MAX_CHAIN_LENGTH:
        print(f"Warning: Chain building hit safety limit of {MAX_CHAIN_LENGTH} nodes starting from {start_node}")
        return None
    
    # Only return chains with more than 1 node
    return chain if len(chain) > 1 else None

def _filter_valid_chain_pairs(g, all_chains):
    """
    Filter chains to ensure each chain and its complement can both be merged without conflicts.
    """
    if not all_chains:
        return []
    
    valid_chains = []
    used_nodes = set()
    
    # Pre-compute set of all nodes for faster membership testing
    all_nodes = set(g.nodes())
    
    for chain in tqdm(all_chains, desc="Filtering valid chain pairs", unit="chain"):
        # Convert chain to set once for all operations
        chain_set = set(chain)
        
        # Skip if any node in chain is already used
        if used_nodes & chain_set:  # Using & operator is faster than intersection()
            continue
           
        # Check complement chain - calculate once
        complement_chain = [n ^ 1 for n in reversed(chain)]
        complement_set = set(complement_chain)
        
        # Skip if complement nodes don't exist
        if not complement_set <= all_nodes:  # Using <= operator is faster than issubset()
            continue
            
        # Skip if any complement node is already used
        if used_nodes & complement_set:
            continue
        
        # Skip if chain and complement overlap (palindromic)
        if chain_set & complement_set:
            continue
            
        # This chain pair is valid
        valid_chains.append(chain)
        used_nodes |= chain_set          # Using |= operator is faster than update()
        used_nodes |= complement_set
    
    return valid_chains


def _merge_chain_fast(simplified_g, chain, read_seqs, new_node_id):
    """
    Merge an entire chain into the first node efficiently.
    This version only handles a single chain - call twice for chain and complement.
    """
    # Get original target node (first in chain)
    original_target = chain[0]
    chain_set = set(chain)
    
    # Debug: track sequence lengths
    original_total_length = sum(len(read_seqs[node]) for node in chain)
    
    # Create new node with merged attributes
    simplified_g.add_node(new_node_id)
    
    # Copy attributes from original target node as base
    for attr, value in simplified_g.nodes[original_target].items():
        simplified_g.nodes[new_node_id][attr] = value

    # Initialize sequence for new node 
    read_seqs[new_node_id] = read_seqs[original_target]

    # Calculate total read_length by summing all lengths minus overlaps
    total_length = simplified_g.nodes[original_target]['read_length']
    
    # Process chain - merge each subsequent node
    for i in range(1, len(chain)):
        source_node = chain[i-1]  # source of the edge
        target_node = chain[i]    # target of the edge
        
        overlap_len = simplified_g[source_node][target_node]['overlap_length']
        
        # Debug: check for suspicious overlaps
        if overlap_len >= len(read_seqs[target_node]):
            print(f"Warning: Overlap {overlap_len} >= target sequence length {len(read_seqs.get(target_node, ''))} for nodes {source_node}->{target_node}")
        
        # Update total length
        total_length += simplified_g.nodes[target_node]['read_length'] - overlap_len
        
        # Merge sequence
        _append_sequence_with_overlap(read_seqs, new_node_id, target_node, overlap_len)
    
    # Debug: check final sequence length
    final_seq_length = len(read_seqs[new_node_id])
    expected_length = original_total_length - sum(
        simplified_g[chain[i-1]][chain[i]]['overlap_length'] 
        for i in range(1, len(chain))
    )
    
    """if abs(final_seq_length - expected_length) > 10:  # Allow small differences
        print(f"Warning: Sequence length mismatch for chain {chain}")
        print(f"  Expected: {expected_length}, Got: {final_seq_length}")
        print(f"  Original total: {original_total_length}")"""
    
    # Set attributes based on analysis of entire chain
    _set_chain_attributes(simplified_g, new_node_id, chain, total_length)
    
    # Redirect incoming edges to new node
    for source in list(simplified_g.predecessors(original_target)):
        if source not in chain_set:
            edge_data = simplified_g[source][original_target]
            filtered_data = {k: v for k, v in edge_data.items() 
                           if k in ['overlap_length', 'overlap_similarity', 'prefix_length', 'hifiasm_result']}
            
            # Don't recompute prefix_length here - will be done in post-processing
            simplified_g.add_edge(source, new_node_id, **filtered_data)
    
    # Redirect outgoing edges from last node in chain
    last_node = chain[-1]
    for target in list(simplified_g.successors(last_node)):
        if target not in chain_set:
            edge_data = simplified_g[last_node][target]
            filtered_data = {k: v for k, v in edge_data.items() 
                           if k in ['overlap_length', 'overlap_similarity', 'prefix_length', 'hifiasm_result']}
            
            # Don't recompute prefix_length here - will be done in post-processing
            simplified_g.add_edge(new_node_id, target, **filtered_data)
    
    # Remove nodes from graph
    simplified_g.remove_nodes_from(chain_set)
    
    return len(chain_set)

def _set_chain_attributes(simplified_g, new_node_id, chain, total_length):
    """
    Set attributes for the merged node based on analysis of the entire chain.
    For coordinates: takes the earliest start and latest end from all reads in the chain.
    Can infer coordinates for first/last reads using overlap lengths if they're missing coordinates.
    
    Args:
        simplified_g: The graph
        new_node_id: ID of the new merged node
        chain: List of node IDs in the chain (ordered from first to last)
        total_length: Pre-calculated total read length
    """
    merge_attrs = simplified_g.nodes[new_node_id]
    
    # Set the pre-calculated read_length
    merge_attrs['read_length'] = total_length
    
    # Check if we have coordinate attributes for any node in the chain
    coordinate_attrs = ['read_start_M', 'read_start_P', 'read_end_M', 'read_end_P', 'read_chr', 'read_strand']
    has_coordinates = any(
        any(attr in simplified_g.nodes[node] for attr in coordinate_attrs)
        for node in chain
    )
    
    if not has_coordinates:
        # Real data - only process support attribute
        support_values = []
        support_weights = []
        
        for node in chain:
            node_attrs = simplified_g.nodes[node]
            if 'support' in node_attrs and node_attrs['support'] is not None:
                support_values.append(node_attrs['support'])
                node_length = node_attrs.get('read_length', 1)
                support_weights.append(node_length)
        
        # Set support attribute: weighted average by node length
        if support_values and support_weights:
            weighted_sum = sum(support * weight for support, weight in zip(support_values, support_weights))
            total_weight = sum(support_weights)
            merged_support = weighted_sum / total_weight
            merge_attrs['support'] = merged_support
        else:
            merge_attrs['support'] = None
        
        return  # Skip coordinate processing for real data
    
    # Synthetic data - process coordinates as before
    # First, try to infer missing coordinates for first/last reads using overlaps
    _infer_missing_coordinates_in_chain(simplified_g, chain)
    
    # Collect all values for coordinate attributes
    start_m_values = []
    start_p_values = []
    end_m_values = []
    end_p_values = []
    chr_values = []
    strand_values = []
    
    # Collect support values and their weights for weighted average
    support_values = []
    support_weights = []
    
    # Debug: track coordinate ranges for validation
    coordinate_debug = {
        'chain_length': len(chain),
        'chain_nodes': chain,
        'coordinates_found': {'M': False, 'P': False}
    }
    
    for node in chain:
        node_attrs = simplified_g.nodes[node]
        
        # Collect coordinate values (only non-None values)
        if 'read_start_M' in node_attrs and node_attrs['read_start_M'] is not None:
            start_m_values.append(node_attrs['read_start_M'])
            coordinate_debug['coordinates_found']['M'] = True
        if 'read_start_P' in node_attrs and node_attrs['read_start_P'] is not None:
            start_p_values.append(node_attrs['read_start_P'])
            coordinate_debug['coordinates_found']['P'] = True
        if 'read_end_M' in node_attrs and node_attrs['read_end_M'] is not None:
            end_m_values.append(node_attrs['read_end_M'])
        if 'read_end_P' in node_attrs and node_attrs['read_end_P'] is not None:
            end_p_values.append(node_attrs['read_end_P'])
            
        # Collect categorical values (only non-None values)
        if 'read_chr' in node_attrs and node_attrs['read_chr'] is not None:
            chr_values.append(node_attrs['read_chr'])
        if 'read_strand' in node_attrs and node_attrs['read_strand'] is not None:
            strand_values.append(node_attrs['read_strand'])
        
        # Collect support values and weights (only non-None values)
        if 'support' in node_attrs and node_attrs['support'] is not None:
            support_values.append(node_attrs['support'])
            # Use read_length as weight
            node_length = node_attrs.get('read_length', 1)  # Default to 1 if missing
            support_weights.append(node_length)
    
    # Set coordinate attributes: min for start (earliest), max for end (latest)
    # This ensures the merged read spans from the earliest start to the latest end
    # of all reads in the unbranching path
    
    # Maternal coordinates
    if start_m_values and end_m_values:
        earliest_start_m = min(start_m_values)
        latest_end_m = max(end_m_values)
        
        merge_attrs['read_start_M'] = earliest_start_m
        merge_attrs['read_end_M'] = latest_end_m
    else:
        merge_attrs['read_start_M'] = None
        merge_attrs['read_end_M'] = None
    
    # Paternal coordinates  
    if start_p_values and end_p_values:
        earliest_start_p = min(start_p_values)
        latest_end_p = max(end_p_values)
        
        merge_attrs['read_start_P'] = earliest_start_p
        merge_attrs['read_end_P'] = latest_end_p
    else:
        merge_attrs['read_start_P'] = None
        merge_attrs['read_end_P'] = None
    
    # Set categorical attributes: majority vote
    merge_attrs['read_chr'] = _get_majority_value(chr_values)
    merge_attrs['read_strand'] = _get_majority_value(strand_values)
    
    # Set support attribute: weighted average by node length
    if support_values and support_weights:
        weighted_sum = sum(support * weight for support, weight in zip(support_values, support_weights))
        total_weight = sum(support_weights)
        merged_support = weighted_sum / total_weight
        merge_attrs['support'] = merged_support
    else:
        merge_attrs['support'] = None

def _infer_missing_coordinates_in_chain(simplified_g, chain):
    """
    Infer coordinates for first/last reads in the chain if they're missing coordinates
    but other reads in the chain have coordinates, using overlap lengths and read lengths.
    
    Args:
        simplified_g: The graph
        chain: List of node IDs in the chain (ordered from first to last)
    """
    if len(chain) < 2:
        return
    
    # Try to infer coordinates for first read if missing
    _infer_coordinates_for_first_read(simplified_g, chain)
    
    # Try to infer coordinates for last read if missing  
    _infer_coordinates_for_last_read(simplified_g, chain)

def _infer_coordinates_for_first_read(simplified_g, chain):
    """
    Infer coordinates for the first read in the chain if it's missing coordinates
    but some later read has coordinates.
    """
    first_node = chain[0]
    first_attrs = simplified_g.nodes[first_node]
    
    # Check if first read is missing coordinates
    has_maternal = (first_attrs.get('read_start_M') is not None and 
                   first_attrs.get('read_end_M') is not None)
    has_paternal = (first_attrs.get('read_start_P') is not None and 
                   first_attrs.get('read_end_P') is not None)
    
    if has_maternal and has_paternal:
        return  # First read already has coordinates
    
    # Find the first read in the chain that has coordinates
    for i in range(1, len(chain)):
        ref_node = chain[i]
        ref_attrs = simplified_g.nodes[ref_node]
        
        ref_has_maternal = (ref_attrs.get('read_start_M') is not None and 
                           ref_attrs.get('read_end_M') is not None)
        ref_has_paternal = (ref_attrs.get('read_start_P') is not None and 
                           ref_attrs.get('read_end_P') is not None)
        
        if ref_has_maternal or ref_has_paternal:
            # Calculate accumulated sequence length from first read to this reference read
            accumulated_length = _calculate_accumulated_length(simplified_g, chain, 0, i)
            
            # Infer coordinates for first read
            if ref_has_maternal and not has_maternal:
                ref_start_m = ref_attrs['read_start_M'] 
                ref_end_m = ref_attrs['read_end_M']
                
                # First read starts 'accumulated_length' bases before the reference read
                inferred_start_m = ref_start_m - accumulated_length
                inferred_end_m = inferred_start_m + first_attrs['read_length']
                
                first_attrs['read_start_M'] = inferred_start_m
                first_attrs['read_end_M'] = inferred_end_m
                
                print(f"Inferred maternal coordinates for first read {first_node}: "
                      f"{inferred_start_m}-{inferred_end_m} (offset -{accumulated_length} from read {ref_node})")
            
            if ref_has_paternal and not has_paternal:
                ref_start_p = ref_attrs['read_start_P']
                ref_end_p = ref_attrs['read_end_P']
                
                # First read starts 'accumulated_length' bases before the reference read
                inferred_start_p = ref_start_p - accumulated_length
                inferred_end_p = inferred_start_p + first_attrs['read_length']
                
                first_attrs['read_start_P'] = inferred_start_p
                first_attrs['read_end_P'] = inferred_end_p
                
                print(f"Inferred paternal coordinates for first read {first_node}: "
                      f"{inferred_start_p}-{inferred_end_p} (offset -{accumulated_length} from read {ref_node})")
            
            break  # We found a reference read, no need to continue

def _infer_coordinates_for_last_read(simplified_g, chain):
    """
    Infer coordinates for the last read in the chain if it's missing coordinates
    but some earlier read has coordinates.
    """
    last_node = chain[-1]
    last_attrs = simplified_g.nodes[last_node]
    
    # Check if last read is missing coordinates
    has_maternal = (last_attrs.get('read_start_M') is not None and 
                   last_attrs.get('read_end_M') is not None)
    has_paternal = (last_attrs.get('read_start_P') is not None and 
                   last_attrs.get('read_end_P') is not None)
    
    if has_maternal and has_paternal:
        return  # Last read already has coordinates
    
    # Find the last read in the chain that has coordinates (searching backwards)
    for i in range(len(chain) - 2, -1, -1):
        ref_node = chain[i]
        ref_attrs = simplified_g.nodes[ref_node]
        
        ref_has_maternal = (ref_attrs['read_start_M'] is not None and 
                           ref_attrs['read_end_M'] is not None)
        ref_has_paternal = (ref_attrs['read_start_P'] is not None and 
                           ref_attrs['read_end_P'] is not None)
        
        if ref_has_maternal or ref_has_paternal:
            # Calculate accumulated sequence length from reference read to last read
            accumulated_length = _calculate_accumulated_length(simplified_g, chain, i, len(chain) - 1)
            
            # Infer coordinates for last read
            if ref_has_maternal and not has_maternal:
                ref_start_m = ref_attrs['read_start_M']
                ref_end_m = ref_attrs['read_end_M']
                
                # Last read starts 'accumulated_length' bases after the reference read
                inferred_start_m = ref_start_m + accumulated_length
                inferred_end_m = inferred_start_m + last_attrs['read_length']
                
                last_attrs['read_start_M'] = inferred_start_m
                last_attrs['read_end_M'] = inferred_end_m
                
                print(f"Inferred maternal coordinates for last read {last_node}: "
                      f"{inferred_start_m}-{inferred_end_m} (offset +{accumulated_length} from read {ref_node})")
            
            if ref_has_paternal and not has_paternal:
                ref_start_p = ref_attrs['read_start_P']
                ref_end_p = ref_attrs['read_end_P']
                
                # Last read starts 'accumulated_length' bases after the reference read
                inferred_start_p = ref_start_p + accumulated_length
                inferred_end_p = inferred_start_p + last_attrs.get('read_length', 0)
                
                last_attrs['read_start_P'] = inferred_start_p
                last_attrs['read_end_P'] = inferred_end_p
                
                print(f"Inferred paternal coordinates for last read {last_node}: "
                      f"{inferred_start_p}-{inferred_end_p} (offset +{accumulated_length} from read {ref_node})")
            
            break  # We found a reference read, no need to continue

def _calculate_accumulated_length(simplified_g, chain, start_idx, end_idx):
    """
    Calculate the accumulated sequence length from start_idx to end_idx in the chain.
    This is the sum of read lengths minus the sum of overlaps between consecutive reads.
    
    Args:
        simplified_g: The graph
        chain: List of node IDs in the chain
        start_idx: Starting index in the chain (inclusive)
        end_idx: Ending index in the chain (inclusive)
        
    Returns:
        int: Accumulated sequence length in bases
    """
    if start_idx >= end_idx:
        return 0
    
    accumulated = 0
    
    # Add lengths of all reads from start_idx to end_idx
    for i in range(start_idx, end_idx + 1):
        node = chain[i]
        read_length = simplified_g.nodes[node]['read_length']
        accumulated += read_length
    
    # Subtract overlaps between consecutive reads
    for i in range(start_idx, end_idx):
        source_node = chain[i]
        target_node = chain[i + 1]
        
        if simplified_g.has_edge(source_node, target_node):
            overlap_length = simplified_g[source_node][target_node]['overlap_length']
            accumulated -= overlap_length
        else:
            print(f"Warning: No edge found between consecutive chain nodes {source_node} -> {target_node}")
    
    return accumulated

def _get_majority_value(values):
    """
    Get the majority value from a list of values.
    Returns None if no values or no clear majority.
    """
    if not values:
        return None
    
    # Count occurrences
    from collections import Counter
    counts = Counter(values)
    
    # Get the most common value
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else None

def _append_sequence_with_overlap(read_seqs, accumulated_node, target_node, overlap_len):
    """
    Append target sequence to accumulated sequence, accounting for overlap.
    """
    
    accumulated_seq = read_seqs[accumulated_node]
    target_seq = read_seqs[target_node]
    
    #print(f"Merging: accumulated_len={len(accumulated_seq)}, target_len={len(target_seq)}, overlap={overlap_len}")
    
    # Append the non-overlapping part of target sequence to accumulated sequence
    if overlap_len < len(target_seq):
        merged_seq = accumulated_seq + target_seq[overlap_len:]
    else:
        # If overlap is >= target length, just keep accumulated sequence
        merged_seq = accumulated_seq
    
    #print(f"Result: merged_len={len(merged_seq)}")
    read_seqs[accumulated_node] = merged_seq

def analyze_existing_node_attributes(graph):
    """
    Analyze what node attributes actually exist in the graph.
    
    Returns:
        dict: Summary of attribute coverage
    """
    print("Analyzing existing node attributes...")
    
    if len(graph.nodes()) == 0:
        print("No nodes to analyze")
        return {}
    
    # Collect all attributes across all nodes
    all_attributes = set()
    attr_coverage = {}
    
    for node in graph.nodes():
        node_attrs = set(graph.nodes[node].keys())
        all_attributes.update(node_attrs)
    
    # Check coverage for each attribute
    total_nodes = len(graph.nodes())
    for attr in all_attributes:
        nodes_with_attr = sum(1 for node in graph.nodes() if attr in graph.nodes[node])
        coverage_pct = (nodes_with_attr / total_nodes) * 100
        attr_coverage[attr] = {
            'count': nodes_with_attr,
            'coverage': coverage_pct
        }
    
    return attr_coverage

def validate_node_attributes(graph, expected_attributes=None, require_full_coverage=False):
    """
    Check node attributes with flexible validation.
    
    Args:
        graph: NetworkX graph to validate
        expected_attributes: List of attribute names to check
        require_full_coverage: If True, all nodes must have all attributes.
                              If False, only check that attributes exist where expected.
    """
    print("Validating node attributes...")
    
    if len(graph.nodes()) == 0:
        print("✅ No nodes to validate")
        return True
    
    # First analyze what actually exists
    attr_coverage = analyze_existing_node_attributes(graph)
    
    if expected_attributes is None:
        # Only validate attributes that have reasonable coverage (>50%)
        expected_attributes = [attr for attr, info in attr_coverage.items() 
                             if info['coverage'] > 50.0]
        print(f"Auto-selected attributes with >50% coverage: {expected_attributes}")
    
    if not expected_attributes:
        print("✅ No attributes to validate")
        return True
    
    # Separate coordinate attributes from core attributes
    coordinate_attrs = {'read_start_M', 'read_start_P', 'read_end_M', 'read_end_P', 'read_chr', 'read_strand'}
    core_attrs = [attr for attr in expected_attributes if attr not in coordinate_attrs]
    coord_attrs_expected = [attr for attr in expected_attributes if attr in coordinate_attrs]
    
    if require_full_coverage:
        # Strict validation - all nodes must have all attributes
        missing_attrs_by_node = {}
        for node in graph.nodes():
            node_attrs = set(graph.nodes[node].keys())
            missing_attrs = [attr for attr in expected_attributes if attr not in node_attrs]
            if missing_attrs:
                missing_attrs_by_node[node] = missing_attrs
        
        if missing_attrs_by_node:
            print(f"❌ Strict validation failed: {len(missing_attrs_by_node)} nodes missing attributes")
            return False
        else:
            print(f"✅ Strict validation passed: All nodes have {expected_attributes}")
            return True
    else:
        # Flexible validation - just warn about inconsistencies
        # Be more strict about core attributes, more lenient about coordinate attributes
        for attr in expected_attributes:
            if attr in attr_coverage:
                coverage = attr_coverage[attr]['coverage']
                
                if attr in core_attrs:
                    # Core attributes should have high coverage
                    if coverage < 95.0:
                        print(f"⚠️  Core attribute '{attr}' coverage: {coverage:.1f}% ({attr_coverage[attr]['count']} nodes)")
                    else:
                        print(f"✅ Core attribute '{attr}' coverage: {coverage:.1f}%")
                else:
                    # Coordinate attributes can have lower coverage (due to inference not always working)
                    if coverage < 70.0:
                        print(f"⚠️  Coordinate attribute '{attr}' coverage: {coverage:.1f}% ({attr_coverage[attr]['count']} nodes)")
                    else:
                        print(f"✅ Coordinate attribute '{attr}' coverage: {coverage:.1f}%")
            else:
                print(f"❌ '{attr}' not found in any nodes")
        
        return True  # Flexible validation always passes, just reports issues

def validate_complement_symmetry(graph):
    """
    Validate that all nodes have their complements and the graph maintains symmetry.
    
    Args:
        graph: NetworkX graph to validate
        
    Returns:
        bool: True if valid, False if issues found
    """
    print("Validating complement symmetry...")
    
    missing_complements = []
    asymmetric_edges = []
    
    # Check all nodes have complements
    for node in graph.nodes():
        complement = node ^ 1
        if complement not in graph:
            missing_complements.append(node)
    
    # Check edge symmetry
    for u, v, data in graph.edges(data=True):
        # Get complement nodes
        u_comp = u ^ 1
        v_comp = v ^ 1
        
        # Check if complement edge exists
        if not graph.has_edge(v_comp, u_comp):
            asymmetric_edges.append((u, v))
    
    if missing_complements or asymmetric_edges:
        if missing_complements:
            print(f"❌ Found {len(missing_complements)} nodes missing complements:")
            for node in missing_complements[:10]:  # Show first 10
                print(f"  Node {node} missing complement {node ^ 1}")
            if len(missing_complements) > 10:
                print(f"  ... and {len(missing_complements) - 10} more")
                
        if asymmetric_edges:
            print(f"❌ Found {len(asymmetric_edges)} asymmetric edges:")
            for u, v in asymmetric_edges[:10]:  # Show first 10
                print(f"  Edge ({u}, {v}) missing complement ({v^1}, {u^1})")
            if len(asymmetric_edges) > 10:
                print(f"  ... and {len(asymmetric_edges) - 10} more")
        return False
    
    print("✅ Complement symmetry validated: all nodes have complements and edges are symmetric")
    return True

def validate_graph_integrity(graph, expected_node_attributes=None, strict_node_validation=False):
    """
    Comprehensive validation of graph integrity.
    
    Args:
        graph: NetworkX graph to validate
        expected_node_attributes: List of expected node attributes
        strict_node_validation: If True, require all nodes to have all attributes
        
    Returns:
        bool: True if valid, False if issues found
    """
    print("Validating graph integrity...")
    
    # Check edge integrity (always strict)
    all_nodes = set(graph.nodes())
    invalid_edges = []
    
    for u, v, data in graph.edges(data=True):
        if u not in all_nodes:
            invalid_edges.append((u, v, f"Source node {u} does not exist"))
        if v not in all_nodes:
            invalid_edges.append((u, v, f"Target node {v} does not exist"))
    
    edge_integrity_valid = True
    if invalid_edges:
        print(f"❌ Found {len(invalid_edges)} invalid edges:")
        for u, v, reason in invalid_edges[:10]:  # Show first 10
            print(f"  Edge ({u}, {v}): {reason}")
        if len(invalid_edges) > 10:
            print(f"  ... and {len(invalid_edges) - 10} more")
        edge_integrity_valid = False
    else:
        print(f"✅ Edge integrity validated: {len(graph.edges())} edges all point to existing nodes")
    
    # Check node attributes
    #node_attrs_valid = validate_node_attributes(graph, expected_node_attributes, strict_node_validation)
    node_attrs_valid = True

    # Check complement symmetry
    complement_symmetry_valid = validate_complement_symmetry(graph)
    
    return edge_integrity_valid and node_attrs_valid and complement_symmetry_valid

def analyze_edge_statistics(graph):
    """
    Analyze and print statistics about edge attributes in the graph.
    
    Args:
        graph: NetworkX graph to analyze
    """
    print("\n" + "="*50)
    print("EDGE STATISTICS ANALYSIS")
    print("="*50)
    
    if graph.number_of_edges() == 0:
        print("No edges in graph to analyze")
        return
    
    # Get edge attributes
    overlap_lengths = list(nx.get_edge_attributes(graph, "overlap_length").values())
    prefix_lengths = list(nx.get_edge_attributes(graph, "prefix_length").values())
    similarities = list(nx.get_edge_attributes(graph, "overlap_similarity").values())
    supports = list(nx.get_edge_attributes(graph, "support").values())
    
    def print_stats(values, name):
        if not values:
            print(f"{name}: No data available")
            return
            
        values = np.array(values)
        print(f"\n{name}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean:  {np.mean(values):.2f}")
        print(f"  Std:   {np.std(values):.2f}")
        print(f"  Min:   {np.min(values):.2f}")
        print(f"  Max:   {np.max(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  25th percentile: {np.percentile(values, 25):.2f}")
        print(f"  75th percentile: {np.percentile(values, 75):.2f}")
    
    # Print statistics for each attribute
    print_stats(overlap_lengths, "Overlap Lengths")
    print_stats(prefix_lengths, "Prefix Lengths")
    print_stats(similarities, "Similarities")
    
    # Additional analysis
    if overlap_lengths and prefix_lengths:
        print(f"\nOverlap vs Prefix Length Analysis:")
        overlap_array = np.array(overlap_lengths)
        prefix_array = np.array(prefix_lengths)
        
        # Find cases where overlap is very large compared to prefix
        large_overlap_ratio = overlap_array / (prefix_array + 1e-6)  # Add small value to avoid division by zero
        suspicious_overlaps = np.sum(large_overlap_ratio > 0.9)
        
        print(f"  Edges with overlap > 90% of prefix length: {suspicious_overlaps}")
        print(f"  Mean overlap/prefix ratio: {np.mean(large_overlap_ratio):.3f}")
        
        if suspicious_overlaps > 0:
            print(f"  Warning: {suspicious_overlaps} edges have very large overlaps relative to prefix length")
    
    print("="*50)

def recompute_all_prefix_lengths(graph):
    """
    Recompute prefix_length for all edges in the graph.
    prefix_length = source_node_length - overlap_length
    
    Args:
        graph: NetworkX graph with edges that have overlap_length attribute
    """
    print("Recomputing all prefix lengths...")
    
    edges_updated = 0
    for u, v, data in graph.edges(data=True):
        if 'overlap_length' in data:
            source_node_length = graph.nodes[u].get('read_length', 0)
            overlap_length = data['overlap_length']
            
            # Recompute prefix length
            new_prefix_length = source_node_length - overlap_length
            
            # Update the edge attribute
            graph[u][v]['prefix_length'] = new_prefix_length
            edges_updated += 1
        else:
            print(f"Warning: Edge ({u}, {v}) missing overlap_length attribute")
    
    print(f"Updated prefix_length for {edges_updated} edges")

def build_simplified_graph_iterative(graph_nx, read_seqs):
    """
    Iteratively remove transitive edges and condense paths until convergence
    """
    current_graph = graph_nx.copy()
    
    # Calculate initial sequence statistics
    initial_total_sequence = sum(len(seq) for seq in read_seqs.values())
    
    print(f"Initial graph statistics:")
    print(f"  Nodes: {current_graph.number_of_nodes()}")
    print(f"  Edges: {current_graph.number_of_edges()}")
    print(f"  Total sequence length: {initial_total_sequence:,} bp")
    
    # Analyze initial edge statistics
    print("\nINITIAL GRAPH EDGE STATISTICS:")
    analyze_edge_statistics(current_graph)
    
    # Detect data type and determine expected attributes
    synthetic_data = is_synthetic_data(current_graph)
    expected_attrs = ['read_length', 'support']
    if synthetic_data:
        expected_attrs.extend(['read_start_M', 'read_start_P', 'read_end_M', 'read_end_P', 'read_chr', 'read_strand'])
    
    # Clean up graph attributes - only keep specified attributes
    _clean_graph_attributes(current_graph)
    
    # Validate initial graph with dynamically determined attributes
    if not validate_graph_integrity(current_graph, expected_attrs):
        print("❌ Initial graph is invalid! Stopping.")
        return current_graph, read_seqs

    initial_nodes = current_graph.number_of_nodes()
    initial_edges = current_graph.number_of_edges()
    
    # Step 1: Remove transitive edges
    marked_edges = remove_transitive_edges(current_graph)
    current_graph.remove_edges_from(marked_edges)
    
    # Validate after transitive edge removal
    if not validate_complement_symmetry(current_graph):
        print("❌ Graph lost complement symmetry after transitive edge removal!")
        exit()
        
    nodes_after_transitive = current_graph.number_of_nodes()
    edges_after_transitive = current_graph.number_of_edges()
    
    print(f"Removed {len(marked_edges)} transitive edges")
    print(f"  Nodes after transitive removal: {nodes_after_transitive}")
    print(f"  Edges after transitive removal: {edges_after_transitive}")
    
    # Analyze edge statistics after transitive removal
    print("\nAFTER TRANSITIVE EDGE REMOVAL:")
    analyze_edge_statistics(current_graph)
    
    # Step 2: Condense unbranching paths  
    current_graph, nodes_removed, read_seqs, utg_to_old_nodes = condense_unbranching_paths(current_graph, read_seqs)
    
    # Step 3: Recompute all prefix lengths after merging
    recompute_all_prefix_lengths(current_graph)
    
    # Validate after condensing
    if not validate_graph_integrity(current_graph, expected_attrs):
        print("❌ Graph invalid after condensing paths!")
        exit()
        
    nodes_after_condensing = current_graph.number_of_nodes()
    edges_after_condensing = current_graph.number_of_edges()
    
    print(f"Condensed {nodes_removed} nodes")
    print(f"  Nodes after condensing: {nodes_after_condensing}")
    print(f"  Edges after condensing: {edges_after_condensing}")
    
    # Analyze final edge statistics
    print("\nFINAL GRAPH EDGE STATISTICS:")
    analyze_edge_statistics(current_graph)
    
    # Summary for this iteration
    total_nodes_removed = initial_nodes - nodes_after_condensing
    total_edges_removed = initial_edges - edges_after_condensing
    print(f"\nSUMMARY:")
    print(f"  Total nodes removed: {total_nodes_removed}")
    print(f"  Total edges removed: {total_edges_removed}")
    print(f"  Final nodes: {nodes_after_condensing}")
    print(f"  Final edges: {edges_after_condensing}")
    
    # Calculate final sequence statistics
    final_total_sequence = sum(len(seq) for seq in read_seqs.values())
    print(f"  Total sequence length: {final_total_sequence:,} bp")
    print(f"  Sequence loss: {initial_total_sequence - final_total_sequence:,} bp ({((initial_total_sequence - final_total_sequence) / initial_total_sequence * 100):.1f}%)")
    
    # Final validation
    if not validate_graph_integrity(current_graph, expected_attrs):
        print("❌ Final graph is invalid!")
    
    return current_graph, read_seqs, utg_to_old_nodes

def reindex_nodes(graph, read_seqs, utg_to_old_nodes=None):
    """
    Reindex nodes in the graph starting from 0 while preserving complement relationships.
    Each node i and its complement i^1 should map to new indices j and j^1.
    
    Args:
        graph: NetworkX graph with nodes to reindex
        read_seqs: Optional dictionary of read sequences to update
        utg_to_old_nodes: Optional dictionary mapping current UTG IDs to old node IDs
        
    Returns:
        new_graph: Graph with reindexed nodes
        new_read_seqs: Updated read sequences with new indices
        new_utg_to_old_nodes: Updated mapping with new UTG IDs
    """
    # Get all nodes
    nodes = sorted(list(graph.nodes()))
    
    # Create a mapping for node pairs (ensuring that i → j means i^1 → j^1)
    node_mapping = {}
    new_index = 0
    
    for node in nodes:
        if node not in node_mapping:
            # Assign new indices to this node and its complement
            node_mapping[node] = new_index
            node_mapping[node ^ 1] = new_index ^ 1
            new_index += 2
    
    # Create a new graph with the reindexed nodes
    new_graph = nx.DiGraph()
    
    # Copy all nodes with their attributes
    for old_id, new_id in node_mapping.items():
        if old_id in graph:
            new_graph.add_node(new_id)
            # Copy node attributes
            for key, value in graph.nodes[old_id].items():
                new_graph.nodes[new_id][key] = value
    
    # Copy all edges with their attributes
    for u, v, data in graph.edges(data=True):
        new_u, new_v = node_mapping[u], node_mapping[v]
        new_graph.add_edge(new_u, new_v, **data)
    
    # Update read sequences
    new_read_seqs = {}
    for old_id, seq in read_seqs.items():
        if int(old_id) in node_mapping:
            new_read_seqs[node_mapping[int(old_id)]] = seq
    
    # Update the utg_to_old_nodes mapping with new node IDs
    new_utg_to_old_nodes = {}
    if utg_to_old_nodes is not None:
        for current_utg_id, old_node_list in utg_to_old_nodes.items():
            if current_utg_id in node_mapping:
                new_utg_id = node_mapping[current_utg_id]
                new_utg_to_old_nodes[new_utg_id] = old_node_list
    
    return new_graph, new_read_seqs, new_utg_to_old_nodes

def load_and_simplify_graph(graph_path, reads_fasta_path=None, utgnode_to_node_path=None, output_fasta_path=None, output_graph_path=None, output_gfa_path=None):
    """
    Load a graph from a file and simplify it.
    
    Args:
        graph_path: Path to the NetworkX graph file
        reads_fasta_path: Path to the FASTA file with read sequences OR pickle file with dict {node_id: sequence}
        output_fasta_path: Path to save the updated sequences
        output_graph_path: Path to save the simplified graph
        
    Returns:
        The simplified graph, updated sequences, and node mapping
    """
    # Load the graph
    print(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        graph_nx = pickle.load(f)

    original_input_nodes = set(graph_nx.nodes())

    # Load read sequences - handle both FASTA and pickle formats
    print(f"Loading read sequences from {reads_fasta_path}")
    
    if reads_fasta_path.endswith('.pkl'):
        # Load pickle file containing dictionary
        with open(reads_fasta_path, 'rb') as f:
            read_seqs_raw = pickle.load(f)
        print(f"Loaded pickle file with {len(read_seqs_raw)} sequences")
        
        # Convert keys to integers if needed
        read_seqs = {}
        for read_id, sequence in read_seqs_raw.items():
            try:
                # Handle both string and integer keys
                if isinstance(read_id, str):
                    read_seqs[int(read_id)] = sequence
                else:
                    read_seqs[int(read_id)] = sequence
            except (ValueError, TypeError):
                print(f"Warning: Could not convert read ID '{read_id}' to integer, skipping")
    else:
        # Load FASTA file (existing logic)
        read_seqs_raw = load_fasta(reads_fasta_path)
        
        # Convert string keys to integers to match graph node IDs
        read_seqs = {}
        for read_id, sequence in read_seqs_raw.items():
            try:
                read_seqs[int(read_id)] = sequence
            except ValueError:
                print(f"Warning: Could not convert read ID '{read_id}' to integer, skipping")
    
    print(f"Loaded {len(read_seqs)} read sequences")
    
    # Simplify the graph
    final_graph, updated_read_seqs, utg_to_old_nodes = build_simplified_graph_iterative(graph_nx, read_seqs)
    
    # Relabel nodes to start from 0 and update read sequences accordingly
    print("Relabeling nodes to start from 0...")
    final_graph, updated_read_seqs, updated_utg_to_old_nodes = reindex_nodes(final_graph, updated_read_seqs, utg_to_old_nodes)

    # Filter mapping values to ensure they reference only original input nodes and available reads
    allowed_value_ids = set()
    allowed_value_ids.update(original_input_nodes)
    allowed_value_ids &= set(read_seqs.keys())  # only keep ids that exist in the loaded reads
    
    # Save the UTG to old nodes mapping if path is provided
    if utgnode_to_node_path:
        print(f"Saving UTG to old nodes mapping to {utgnode_to_node_path}")
        with open(utgnode_to_node_path, 'wb') as f:
            pickle.dump(updated_utg_to_old_nodes, f)
        print(f"Saved mapping for {len(updated_utg_to_old_nodes)} UTGs")
    
    # Save the updated sequences
    print(f"Number of updated read sequences: {len(updated_read_seqs)}")
    print(f"Saving updated sequences to {output_fasta_path}")
    save_as_gfa(final_graph, updated_read_seqs, output_gfa_path)
    save_fasta(updated_read_seqs, output_fasta_path)

    # Save the simplified graph
    if output_graph_path:
        print(f"Saving simplified graph to {output_graph_path}")
        with open(output_graph_path, 'wb') as f:
            pickle.dump(final_graph, f)

def _clean_graph_attributes(graph):
    """Remove all attributes except the specified ones"""
    # Detect if this is synthetic or real data
    synthetic_data = is_synthetic_data(graph)
    
    # Base allowed node attributes (always kept)
    allowed_node_attrs = {'read_length', 'support'}
    
    # Add coordinate attributes only for synthetic data
    if synthetic_data:
        coordinate_attrs = {'read_start_M', 'read_start_P', 'read_end_M', 'read_end_P', 'read_chr', 'read_strand'}
        allowed_node_attrs.update(coordinate_attrs)
        print("Keeping coordinate attributes for synthetic data")
    else:
        print("Excluding coordinate attributes for real data")
    
    # Allowed edge attributes  
    allowed_edge_attrs = {'overlap_length', 'overlap_similarity', 'prefix_length', 'hifiasm_result'}
    
    # Clean node attributes
    for node in graph.nodes():
        attrs_to_remove = [attr for attr in graph.nodes[node].keys() if attr not in allowed_node_attrs]
        for attr in attrs_to_remove:
            del graph.nodes[node][attr]
    
    # Clean edge attributes
    for u, v in graph.edges():
        attrs_to_remove = [attr for attr in graph[u][v].keys() if attr not in allowed_edge_attrs]
        for attr in attrs_to_remove:
            del graph[u][v][attr]

def save_as_gfa(nx_graph, sequence_dict, path):
    """
    Save a NetworkX graph as a GFA format file similar to Hifiasm output.
    
    Args:
        nx_graph: NetworkX DiGraph representing the assembly graph
        sequence_dict: Dictionary mapping node IDs to their sequences
        path: Output file path for the GFA file
    """
    with open(path, 'w') as gfa_file:
        # Write header
        gfa_file.write("H\tVN:Z:1.0\n")
        
        # Get node attributes
        read_lengths = nx.get_node_attributes(nx_graph, 'read_length')
        node_support = nx.get_node_attributes(nx_graph, 'support')
        overlap_lengths = nx.get_edge_attributes(nx_graph, 'overlap_length')
        
        # Write S lines (segments) - only for real nodes (even numbered)
        # The graph stores both forward and reverse complement nodes
        # We only write segments for the forward nodes (even numbered)
        for node in sorted(nx_graph.nodes()):
            if node % 2 == 0:  # Real node only (forward orientation)
                segment_id = f"utg{node//2:06d}"
                sequence = sequence_dict.get(node, "*")
                length = read_lengths.get(node, len(sequence) if sequence != "*" else 0)
                
                # Convert normalized support back to coverage
                support = node_support[node]
                coverage = max(1, int(support * 100))
                
                gfa_file.write(f"S\t{segment_id}\t{sequence}\tLN:i:{length}\trd:i:{coverage}\n")
        
        # Write L lines (links)
        # Need to avoid writing duplicate edges since the graph stores both
        # forward edges and their reverse complements explicitly
        written_edges = set()
        
        for edge in nx_graph.edges():
            src, dst = edge
            
            # Convert node IDs to segment IDs and orientations
            src_segment = src // 2
            dst_segment = dst // 2
            src_orient = '+' if src % 2 == 0 else '-'
            dst_orient = '+' if dst % 2 == 0 else '-'
            
            # Create canonical representation to avoid writing both an edge and its reverse complement
            # The reverse complement of (src_segment, src_orient, dst_segment, dst_orient)
            # would be (dst_segment, flip(dst_orient), src_segment, flip(src_orient))
            forward = (src_segment, src_orient, dst_segment, dst_orient)
            reverse = (dst_segment, '-' if dst_orient == '+' else '+', 
                      src_segment, '-' if src_orient == '+' else '+')
            
            # Use the lexicographically smaller one as canonical to ensure consistency
            canonical = min(forward, reverse)
            
            if canonical in written_edges:
                continue
                
            written_edges.add(canonical)
            
            # Write the L line using the original edge direction
            src_id = f"utg{src_segment:06d}"
            dst_id = f"utg{dst_segment:06d}"
            
            # Get overlap length and format as CIGAR
            overlap_len = overlap_lengths.get(edge, 0)
            cigar = f"{overlap_len}M"
            
            gfa_file.write(f"L\t{src_id}\t{src_orient}\t{dst_id}\t{dst_orient}\t{cigar}\n")

    print(f"Saved GFA file to {path}")

def is_synthetic_data(graph):
    """
    Detect if the graph contains synthetic data (with coordinate attributes) or real data.
    For synthetic data, all coordinate attributes should be present.
    For real data, none of them should be present.
    
    Args:
        graph: NetworkX graph to check
        
    Returns:
        bool: True if synthetic data (has coordinates), False if real data (no coordinates)
    """
    if graph.number_of_nodes() == 0:
        return False
    
    coordinate_attrs = ['read_start_M', 'read_start_P', 'read_end_M', 'read_end_P', 'read_chr', 'read_strand']
    
    # Check first few nodes to determine data type
    sample_nodes = list(graph.nodes())[:min(10, len(graph.nodes()))]
    
    nodes_with_coords = 0
    for node in sample_nodes:
        node_attrs = graph.nodes[node]
        has_all_coords = all(attr in node_attrs for attr in coordinate_attrs)
        if has_all_coords:
            nodes_with_coords += 1
    
    # If majority of sampled nodes have coordinates, assume synthetic data
    is_synthetic = nodes_with_coords > len(sample_nodes) // 2
    
    if is_synthetic:
        print("Detected synthetic data (coordinate attributes present)")
    else:
        print("Detected real data (no coordinate attributes)")
    
    return is_synthetic
