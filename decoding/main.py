import os
import torch
import dgl
import pickle
import yaml
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import networkx as nx
import re
import time
from datetime import datetime

import eval
import inference 
import hifiasm_res_based_inference


# Add parent directory to path before importing utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.utils import set_seed, STDS_AND_MEANS
from training.SymGatedGCN import SymGatedGCNModel, SymGatedGCNModelDoubleHead

stds_and_means = STDS_AND_MEANS

# Support-aware sequence assembly:
# When --support_aware flag is used, the walk_to_sequence_support_aware function
# will choose the sequence from the node with higher support value for overlapping regions.
# This can improve assembly quality by preferring sequences from more reliable nodes.

def get_timestamp():
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(message):
    """Print a message with a timestamp."""
    print(f"[{get_timestamp()}] {message}")

def preprocess_graph(g, x_attr, gt=False):
    """Preprocess graph features for model input."""
    # Edge features - match train_bce.py preprocessing
    ol_len = g.edata['overlap_length'].float()
    ol_len /= 10000  # Simple division instead of z-score normalization
    #ol_sim = g.edata['overlap_similarity'].float()
    #e = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
    e = ol_len.unsqueeze(-1)

    # Node degree features - no normalization like in train_bce.py
    pe_in = g.in_degrees().float().unsqueeze(1)
    pe_out = g.out_degrees().float().unsqueeze(1)
    
    # Additional node features to match train_bce.py
    support = g.ndata['support'].float()
    read_length = g.ndata['read_length'].float() / 100000
    
    # Combine all node features like in train_bce.py
    x = torch.cat((pe_in, pe_out, support.unsqueeze(1), read_length.unsqueeze(1)), dim=1)
    
    return x, e

def compute_scores(dgl_path, model_path, config, output_path, device='cpu', double_model=False):
    """
    Compute edge scores for a graph using a trained model and save them to a file.
    
    Args:
        dgl_path: Path to the DGL graph
        model_path: Path to the trained model
        config: Configuration dictionary
        output_path: Path to save the computed scores
        device: Device to use for computation
        double_model: Whether to use double head model
    
    Returns:
        Dictionary containing the computed scores
    """
    print(f"Loading graph from {dgl_path}")
    g = dgl.load_graphs(dgl_path)[0][0].int()
    g = g.to(device)
    
    # Preprocess graph
    x, e = preprocess_graph(g, 'h')
    x = x.to(device)
    e = e.to(device)
    
    # Load model configuration
    train_config = config['training']
    
    # Initialize model based on double_model flag
    print(f"Loading {'double head' if double_model else 'single head'} model...")
    if double_model:
        model = SymGatedGCNModelDoubleHead(
            train_config['node_features'],
            train_config['edge_features'],
            train_config['hidden_features'],
            train_config['hidden_edge_features'],
            train_config['num_gnn_layers'],
            train_config['hidden_edge_scores'],
            train_config['nb_pos_enc'],
            train_config['nr_classes'],
            dropout=train_config['dropout'],
            pred_dropout=train_config.get('pred_dropout', 0),
            norm=train_config['norm']  # Default to 'layer' if not specified
        )
    else:
        model = SymGatedGCNModel(
            train_config['node_features'],
            train_config['edge_features'],
            train_config['hidden_features'],
            train_config['hidden_edge_features'],
            train_config['num_gnn_layers'],
            train_config['hidden_edge_scores'],
            train_config['nb_pos_enc'],
            train_config['nr_classes'],
            dropout=train_config['dropout'],
            norm=train_config['norm']  # Default to 'layer' if not specified
        )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    
    # Compute scores
    print("Computing edge scores...")
    with torch.no_grad():
        score_logits, cut_logits = model(g, x, e)
    
    # Create dictionary to save
    save_dict = {'score_logits': score_logits.detach().cpu()}
    if cut_logits is not None:
        save_dict['cut_logits'] = cut_logits.detach().cpu()
    
    # Create edge mapping dictionary for easier loading
    edge_scores = {(src.item(), dst.item()): torch.sigmoid(score_logits[i]).item()
                  for i, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1]))}
    save_dict['edge_scores'] = edge_scores
    
    # For double models, also create cut scores (to_cut)
    if double_model and cut_logits is not None:
        cut_scores = {(src.item(), dst.item()): torch.sigmoid(cut_logits[i]).item()
                     for i, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1]))}
        save_dict['cut_scores'] = cut_scores
    
    print(f"Initial edge scores: {len(edge_scores)} edges")
    
    # Average scores for complement edge pairs
    complement_pairs_found = 0
    for (src, dst), score in list(edge_scores.items()):
        # Get complement edge (dst^1, src^1)
        comp_src = dst ^ 1
        comp_dst = src ^ 1
        comp_key = (comp_src, comp_dst)
        
        if comp_key in edge_scores:
            # Average the scores between complement pairs
            avg_score = (score + edge_scores[comp_key]) / 2
            edge_scores[(src, dst)] = avg_score
            edge_scores[comp_key] = avg_score
            complement_pairs_found += 1
    
    # Also average cut_scores if they exist
    if double_model and 'cut_scores' in save_dict:
        cut_scores = save_dict['cut_scores']
        for (src, dst), score in list(cut_scores.items()):
            comp_src = dst ^ 1
            comp_dst = src ^ 1
            comp_key = (comp_src, comp_dst)
            
            if comp_key in cut_scores:
                avg_score = (score + cut_scores[comp_key]) / 2
                cut_scores[(src, dst)] = avg_score
                cut_scores[comp_key] = avg_score
        save_dict['cut_scores'] = cut_scores
    
    print(f"Found and averaged {complement_pairs_found} complement edge pairs")
    print(f"Final edge scores: {len(edge_scores)} edges")
    save_dict['edge_scores'] = edge_scores
    
    return save_dict


def load_reads(reads_path):
    """
    Load reads from a FASTA/FASTQ file.
    
    Args:
        reads_path: Path to the FASTA/FASTQ file
    
    Returns:
        Dictionary mapping read IDs to sequences
    """
    print(f"Loading reads from {reads_path}")
    sequences = {}
    
    # Try original path first, then try with .gz if original doesn't exist
    if os.path.exists(reads_path):
        handle = open(reads_path, "rt") if not reads_path.endswith('.gz') else gzip.open(reads_path, "rt")
    elif os.path.exists(reads_path + '.gz'):
        handle = gzip.open(reads_path + '.gz', "rt")
    else:
        raise FileNotFoundError(f"Could not find reads file at {reads_path} or {reads_path}.gz")
    
    # Determine file format from extension
    file_format = "fasta"
    if reads_path.lower().endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz')):
        file_format = "fastq"
    
    for record in SeqIO.parse(handle, file_format):
        sequences[record.id] = str(record.seq)
    handle.close()
    
    print(f"Loaded {len(sequences)} reads")
    return sequences

def walk_to_sequence(walks, graph, n2s):
    """Convert walks to sequences using NetworkX graph attributes."""
    contigs = []
    prefix_length = nx.get_edge_attributes(graph, 'prefix_length')
    for i, walk in enumerate(walks):
        prefixes = []
        for src, dst in zip(walk[:-1], walk[1:]):
            if (src, dst) not in prefix_length:
                print(f"Warning: Edge ({src}, {dst}) not found in graph")
                continue
            prefixes.append((src, prefix_length[src, dst]))
        res = []
        #print(n2s.keys())
        for (src, prefix) in prefixes:
            seq = str(n2s[str(src)]) # did cast int, works?
            res.append(seq[:prefix])

        # Fix the Seq creation here
        contig = Seq(''.join(res) + str(n2s[str(walk[-1])]))
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)

    return contigs

def walk_to_sequence_classic(walks, graph, utg_node_to_node, raw_reads, old_graph):
    """Classic reconstruction by expanding each node to its old read-id chain and stitching via old-graph overlaps.
    
    walks: list of paths in the current graph (node ids)
    utg_node_to_node: mapping current node id -> list of old read ids (ordered)
    raw_reads: dict mapping old read ids -> sequence (from reduced_reads_raw)
    old_graph: NetworkX graph on old read ids with 'overlap_length' on edges
    """

    contigs = []
    old_overlap = nx.get_edge_attributes(old_graph, 'overlap_length')

    utg_map = utg_node_to_node #normalize_utg_map(utg_node_to_node)

    def expand_to_raw_ids(node_id):
        # If this id directly corresponds to a raw read, return it
        if str(node_id) in raw_reads:
            return (node_id,)
        # If this id can be expanded via the UTG map, recursively expand
        if node_id in utg_map:
            expanded = []
            for child in utg_map[node_id]:
                expanded.extend(expand_to_raw_ids(child))
            return tuple(expanded)
        # Unknown id; return empty
        return tuple()

    for i, walk in enumerate(walks):
        if not walk:
            continue

        # Expand walk nodes into a flat sequence of old node ids
        expanded_old_nodes = []
        for node in walk:
            chain = utg_map[node] #get_utg_chain_safe(node)
            # If chain is empty, try interpreting the node as raw directly
            if not chain:
                if str(node) in raw_reads:
                    chain = [int(node)]
                else:
                    print(f"Warning: No UTG mapping and not a raw read for node {node}")
                    continue
            for old_id in chain:
                expanded_old_nodes.extend(expand_to_raw_ids(old_id))

        # Stitch sequences using old overlaps
        parts = []
        prev_old = None
        for old_id in expanded_old_nodes:
            if str(old_id) not in raw_reads:
                print(f"Warning: raw read for id {old_id} not found; skipping this id")
                prev_old = old_id
                continue
            seq = raw_reads[str(old_id)]
            if prev_old is None or str(prev_old) not in raw_reads:
                parts.append(seq)
            else:
                ol = old_overlap[(prev_old, old_id)]
                ol_int = int(ol)
                parts.append(seq[ol_int:])
            prev_old = old_id

        contig_seq = ''.join(parts)
        contig = Seq(contig_seq)
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)
    return contigs

def walk_to_sequence_classic_support_aware(walks, graph, utg_node_to_node, raw_reads, old_graph):
    """Classic reconstruction by expanding each node to its old read-id chain and stitching via old-graph overlaps.
    This version includes support-aware node filtering: removes nodes with lower support than surrounding nodes
    when the surrounding nodes have a direct connection.
    
    walks: list of paths in the current graph (node ids)
    graph: NetworkX graph with node support attributes
    utg_node_to_node: mapping current node id -> list of old read ids (ordered)
    raw_reads: dict mapping old read ids -> sequence (from reduced_reads_raw)
    old_graph: NetworkX graph on old read ids with 'overlap_length' on edges
    """

    contigs = []
    old_overlap = nx.get_edge_attributes(old_graph, 'overlap_length')
    support = nx.get_node_attributes(graph, 'support')

    utg_map = utg_node_to_node #normalize_utg_map(utg_node_to_node)

    def expand_to_raw_ids(node_id):
        # If this id directly corresponds to a raw read, return it
        if str(node_id) in raw_reads:
            return (node_id,)
        # If this id can be expanded via the UTG map, recursively expand
        if node_id in utg_map:
            expanded = []
            for child in utg_map[node_id]:
                expanded.extend(expand_to_raw_ids(child))
            return tuple(expanded)
        # Unknown id; return empty
        return tuple()

    def filter_walk_by_support(walk):
        """Filter walk by removing nodes with lower support than surrounding nodes when they have direct connection."""
        if len(walk) <= 2:
            return walk  # No filtering possible for walks of length 2 or less
        
        filtered_walk = [walk[0]]  # Always keep the first node
        
        for i in range(1, len(walk) - 1):
            prev_node = walk[i - 1]
            curr_node = walk[i]
            next_node = walk[i + 1]
            
            # Check if surrounding nodes have a direct connection
            has_direct_connection = graph.has_edge(prev_node, next_node)
            
            if has_direct_connection:
                # Get support values (default to 0 if not available)
                prev_support = support.get(prev_node, 0.0)
                curr_support = support.get(curr_node, 0.0)
                next_support = support.get(next_node, 0.0)
                
                # Remove current node if its support is lower than both surrounding nodes
                if curr_support < prev_support and curr_support < next_support:
                    print(f"Removing node {curr_node} (support: {curr_support}) from walk, keeping {prev_node}->{next_node} (supports: {prev_support}, {next_support})")
                    continue  # Skip adding this node to filtered walk
            
            # Keep the current node
            filtered_walk.append(curr_node)
        
        # Always keep the last node
        filtered_walk.append(walk[-1])
        
        return filtered_walk

    for i, walk in enumerate(walks):
        if not walk:
            continue

        # Apply support-aware filtering to the walk
        original_length = len(walk)
        filtered_walk = filter_walk_by_support(walk)
        if len(filtered_walk) < original_length:
            print(f"Walk {i}: filtered from {original_length} to {len(filtered_walk)} nodes")

        # Expand walk nodes into a flat sequence of old node ids
        expanded_old_nodes = []
        for node in filtered_walk:
            chain = utg_map[node] #get_utg_chain_safe(node)
            # If chain is empty, try interpreting the node as raw directly
            if not chain:
                if str(node) in raw_reads:
                    chain = [int(node)]
                else:
                    print(f"Warning: No UTG mapping and not a raw read for node {node}")
                    continue
            for old_id in chain:
                expanded_old_nodes.extend(expand_to_raw_ids(old_id))

        # Stitch sequences using old overlaps
        parts = []
        prev_old = None
        for old_id in expanded_old_nodes:
            if str(old_id) not in raw_reads:
                print(f"Warning: raw read for id {old_id} not found; skipping this id")
                prev_old = old_id
                continue
            seq = raw_reads[str(old_id)]
            if prev_old is None or str(prev_old) not in raw_reads:
                parts.append(seq)
            else:
                ol = old_overlap[(prev_old, old_id)]
                ol_int = int(ol)
                parts.append(seq[ol_int:])
            prev_old = old_id

        contig_seq = ''.join(parts)
        contig = Seq(contig_seq)
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)
    return contigs

def walk_to_sequence_support_aware(walks, graph, n2s):
    contigs = []
    # Use node support for ranking
    support = nx.get_node_attributes(graph, 'support')
    # Use edge overlap lengths to place nodes on a global axis per walk
    overlap_lengths = nx.get_edge_attributes(graph, 'overlap_length')



def walk_to_sequence_consensus_baselevel(walks, graph, n2s): #2
    contigs = []
    overlap_lengths = nx.get_edge_attributes(graph, 'overlap_length')
    support = nx.get_node_attributes(graph, 'support')

    for i, walk in enumerate(walks):
        if not walk:
            continue
        if len(walk) == 1:
            contig = Seq(str(n2s[str(walk[0])]))
            contig = SeqIO.SeqRecord(contig)
            contig.id = f'contig_{i+1}'
            contig.description = f'length={len(contig)}'
            contigs.append(contig)
            continue

        # Global positions
        start_positions = {}
        end_positions = {}
        node_order_index = {node_id: idx for idx, node_id in enumerate(walk)}
        node_rank_key = {node_id: (support.get(node_id, 0.0), -node_order_index[node_id]) for node_id in walk}

        first_node = walk[0]
        start_positions[first_node] = 0
        end_positions[first_node] = len(str(n2s[str(first_node)]))

        for j in range(1, len(walk)):
            prev_node = walk[j - 1]
            curr_node = walk[j]
            prev_len = len(str(n2s[str(prev_node)]))
            ol = overlap_lengths.get((prev_node, curr_node), 0)
            start_positions[curr_node] = start_positions[prev_node] + prev_len - ol
            end_positions[curr_node] = start_positions[curr_node] + len(str(n2s[str(curr_node)]))

        # Break coordinates
        break_coords_set = {0}
        for n in walk:
            break_coords_set.add(start_positions[n])
            break_coords_set.add(end_positions[n])
        last_end = max(end_positions.values()) if end_positions else 0
        break_coords_set.add(last_end)
        break_coordinates = sorted(break_coords_set)

        # Contributors per interval via sweep
        from collections import defaultdict
        from bisect import bisect_left, insort
        events_add = defaultdict(list)
        events_remove = defaultdict(list)
        for n in walk:
            events_add[start_positions[n]].append(n)
            events_remove[end_positions[n]].append(n)
        active_indices = []
        interval_contributors = []
        for b_idx in range(len(break_coordinates) - 1):
            b_start = break_coordinates[b_idx]
            for n in events_remove.get(b_start, ()):  # ends do not contribute
                idx = node_order_index[n]
                pos = bisect_left(active_indices, idx)
                if pos < len(active_indices) and active_indices[pos] == idx:
                    active_indices.pop(pos)
            for n in events_add.get(b_start, ()):  # starts do contribute
                idx = node_order_index[n]
                insort(active_indices, idx)
            interval_contributors.append([walk[idx] for idx in active_indices])

        # Build consensus sequence by intervals
        consensus_parts = []
        for b_idx in range(len(break_coordinates) - 1):
            b_start = break_coordinates[b_idx]
            b_end = break_coordinates[b_idx + 1]
            interval_len = b_end - b_start
            if interval_len <= 0:
                continue
            contributors = interval_contributors[b_idx]
            if not contributors:
                continue

            # Small-k fast paths
            if len(contributors) == 1:
                node = contributors[0]
                local_start = b_start - start_positions[node]
                local_end = local_start + interval_len
                consensus_parts.append(str(n2s[str(node)])[local_start:local_end])
                continue
            if len(contributors) == 2:
                n0, n1 = contributors[0], contributors[1]
                winner = n0 if node_rank_key[n0] >= node_rank_key[n1] else n1
                local_start = b_start - start_positions[winner]
                local_end = local_start + interval_len
                consensus_parts.append(str(n2s[str(winner)])[local_start:local_end])
                continue

            # 3+ contributors: per-base weighted voting
            interval_bases = []
            for offset in range(interval_len):
                global_pos = b_start + offset
                base_to_weight = {}
                best_node = None
                best_node_support = -1.0
                best_node_chain_idx = len(walk) + 1

                for node in contributors:
                    local_index = global_pos - start_positions[node]
                    seq = str(n2s[str(node)])
                    if local_index < 0 or local_index >= len(seq):
                        continue
                    base_char = seq[local_index]
                    w = support.get(node, 0.0)
                    base_to_weight[base_char] = base_to_weight.get(base_char, 0.0) + w

                    if (w > best_node_support) or (w == best_node_support and node_order_index[node] < best_node_chain_idx):
                        best_node = node
                        best_node_support = w
                        best_node_chain_idx = node_order_index[node]

                if not base_to_weight:
                    interval_bases.append('N')
                    continue

                max_weight = max(base_to_weight.values())
                candidate_bases = [b for b, w in base_to_weight.items() if w == max_weight]
                if len(candidate_bases) == 1:
                    chosen_base = candidate_bases[0]
                else:
                    if best_node is not None:
                        local_index = global_pos - start_positions[best_node]
                        seq_best = str(n2s[str(best_node)])
                        if 0 <= local_index < len(seq_best):
                            chosen_base = seq_best[local_index]
                        else:
                            chosen_base = candidate_bases[0]
                    else:
                        chosen_base = candidate_bases[0]

                interval_bases.append(chosen_base)

            consensus_parts.append(''.join(interval_bases))

        contig_seq = ''.join(consensus_parts)
        contig = Seq(contig_seq)
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)

    return contigs

def walk_to_sequence_wrong_kmer(walks, graph, n2s, haplotype='maternal'):
    """
    Convert walks to sequences using NetworkX graph attributes.
    For each basepair position, choose the sequence from the node with lowest wrong kmer count.
    This method uses kmer_count_m for paternal haplotypes and kmer_count_p for maternal haplotypes.
    
    Args:
        walks: List of walks (paths through the graph)
        graph: NetworkX graph with node and edge attributes
        n2s: Dictionary mapping node IDs to sequences
        haplotype: 'maternal' or 'paternal' to specify which haplotype these walks belong to
    
    Returns:
        List of SeqRecord objects representing the assembled contigs
    """
    contigs = []
    overlap_length = nx.get_edge_attributes(graph, 'overlap_length')
    kmer_count_m = nx.get_node_attributes(graph, 'kmer_count_m')
    kmer_count_p = nx.get_node_attributes(graph, 'kmer_count_p')
    node_length = nx.get_node_attributes(graph, 'read_length')
    
    # Create wrong_kmer_count dictionary based on haplotype
    if haplotype == 'maternal':
        wrong_kmer_count = kmer_count_m  # Use kmer_count_m for wrong kmers (maternal haplotype)
        print(f"Using maternal kmer counts for wrong kmer selection (maternal haplotype)")
    elif haplotype == 'paternal':
        wrong_kmer_count = kmer_count_p  # Use kmer_count_p for wrong kmers (paternal haplotype)
        print(f"Using paternal kmer counts for wrong kmer selection (paternal haplotype)")
    else:
        raise ValueError(f"Invalid haplotype: {haplotype}. Must be 'maternal' or 'paternal'")
   
    for i, walk in enumerate(walks):
        if len(walk) == 1:
            # Single node walk
            contig = Seq(str(n2s[str(walk[0])]))
            contig = SeqIO.SeqRecord(contig)
            contig.id = f'contig_{i+1}'
            contig.description = f'length={len(contig)}'
            contigs.append(contig)
            continue
            
        # For each node in the walk, determine which basepairs it contributes
        # and which node has the lowest wrong kmer count for each position
        walk_nodes = []
        for node in walk:
            node_seq = str(n2s[str(node)])
            node_wrong_kmers = wrong_kmer_count.get(node, 0)
            node_len = node_length[node]
            walk_nodes.append({
                'node': node,
                'sequence': node_seq,
                'wrong_kmers': node_wrong_kmers,
                'length': node_len
            })
        
        # Calculate the total sequence length and create a wrong kmer matrix
        # For each position, track which nodes contribute and their wrong kmer counts
        total_length = 0
        position_contributors = {}  # position -> list of (node_id, wrong_kmers, start_in_node, end_in_node)
        
        # First pass: calculate total length and identify all contributing positions
        current_pos = 0
        for j, node_info in enumerate(walk_nodes):
            node_seq = node_info['sequence']
            node_len = node_info['length']
            node_wrong_kmers = node_info['wrong_kmers']
            
            if j == 0:
                # First node contributes its full sequence
                for pos in range(len(node_seq)):
                    if current_pos + pos not in position_contributors:
                        position_contributors[current_pos + pos] = []
                    position_contributors[current_pos + pos].append((node_info['node'], node_wrong_kmers, pos, pos))
                current_pos += len(node_seq)
            else:
                # Subsequent nodes: handle overlap
                if j < len(walk_nodes):
                    prev_node = walk_nodes[j-1]
                    edge_key = (prev_node['node'], node_info['node'])
                    
                    if edge_key in overlap_length:
                        ol_len = overlap_length[edge_key]
                        
                        # The new node contributes from position ol_len onwards
                        # But it also contributes to the overlapping positions
                        for pos in range(len(node_seq)):
                            global_pos = current_pos - ol_len + pos
                            
                            if global_pos not in position_contributors:
                                position_contributors[global_pos] = []
                            
                            # Add this node's contribution to this position
                            position_contributors[global_pos].append((node_info['node'], node_wrong_kmers, pos, pos))
                        
                        # Update current position (only add the non-overlapping part)
                        current_pos += len(node_seq) - ol_len
                    else:
                        # No overlap information, treat as no overlap
                        for pos in range(len(node_seq)):
                            if current_pos + pos not in position_contributors:
                                position_contributors[current_pos + pos] = []
                            position_contributors[current_pos + pos].append((node_info['node'], node_wrong_kmers, pos, pos))
                        current_pos += len(node_seq)
        
        # Second pass: build the final sequence by choosing the contributor with lowest wrong kmer count for each position
        final_sequence = ""
        for pos in range(current_pos):
            if pos in position_contributors:
                # Find the contributor with lowest wrong kmer count for this position
                best_contributor = min(position_contributors[pos], key=lambda x: x[1])
                best_node_id, best_wrong_kmers, start_in_node, end_in_node = best_contributor
                
                # Get the sequence from the best node
                best_node_seq = str(n2s[str(best_node_id)])
                final_sequence += best_node_seq[start_in_node:end_in_node + 1]
            else:
                # This shouldn't happen, but add a placeholder if it does
                final_sequence += "N"
        
        contig = Seq(final_sequence)
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)

    return contigs

def walk_to_sequence_classic_support_and_overlap_aware(walks, graph, utg_node_to_node, raw_reads, old_graph):
    """Classic reconstruction by expanding each node to its old read-id chain and stitching via old-graph overlaps.
    This version includes both support-aware node filtering and support-aware overlap handling.
    
    walks: list of paths in the current graph (node ids)
    graph: NetworkX graph with node support attributes
    utg_node_to_node: mapping current node id -> list of old read ids (ordered)
    raw_reads: dict mapping old read ids -> sequence (from reduced_reads_raw)
    old_graph: NetworkX graph on old read ids with 'overlap_length' on edges
    """

    contigs = []
    old_overlap = nx.get_edge_attributes(old_graph, 'overlap_length')
    support = nx.get_node_attributes(graph, 'support')

    utg_map = utg_node_to_node #normalize_utg_map(utg_node_to_node)

    def expand_to_raw_ids(node_id):
        # If this id directly corresponds to a raw read, return it
        if str(node_id) in raw_reads:
            return (node_id,)
        # If this id can be expanded via the UTG map, recursively expand
        if node_id in utg_map:
            expanded = []
            for child in utg_map[node_id]:
                expanded.extend(expand_to_raw_ids(child))
            return tuple(expanded)
        # Unknown id; return empty
        return tuple()

    def filter_walk_by_support(walk):
        """Filter walk by removing nodes with lower support than surrounding nodes when they have direct connection."""
        if len(walk) <= 2:
            return walk  # No filtering possible for walks of length 2 or less
        
        filtered_walk = [walk[0]]  # Always keep the first node
        
        for i in range(1, len(walk) - 1):
            prev_node = walk[i - 1]
            curr_node = walk[i]
            next_node = walk[i + 1]
            
            # Check if surrounding nodes have a direct connection
            has_direct_connection = graph.has_edge(prev_node, next_node)
            
            if has_direct_connection:
                # Get support values (default to 0 if not available)
                prev_support = support.get(prev_node, 0.0)
                curr_support = support.get(curr_node, 0.0)
                next_support = support.get(next_node, 0.0)
                
                # Remove current node if its support is lower than both surrounding nodes
                if curr_support < prev_support and curr_support < next_support:
                    print(f"Removing node {curr_node} (support: {curr_support}) from walk, keeping {prev_node}->{next_node} (supports: {prev_support}, {next_support})")
                    continue  # Skip adding this node to filtered walk
            
            # Keep the current node
            filtered_walk.append(curr_node)
        
        # Always keep the last node
        filtered_walk.append(walk[-1])
        
        return filtered_walk

    for i, walk in enumerate(walks):
        if not walk:
            continue

        # Apply support-aware filtering to the walk
        original_length = len(walk)
        filtered_walk = filter_walk_by_support(walk)
        if len(filtered_walk) < original_length:
            print(f"Walk {i}: filtered from {original_length} to {len(filtered_walk)} nodes")

        # Expand walk nodes into a flat sequence of old node ids
        expanded_old_nodes = []
        for node in filtered_walk:
            chain = utg_map[node] #get_utg_chain_safe(node)
            # If chain is empty, try interpreting the node as raw directly
            if not chain:
                if str(node) in raw_reads:
                    chain = [int(node)]
                else:
                    print(f"Warning: No UTG mapping and not a raw read for node {node}")
                    continue
            for old_id in chain:
                expanded_old_nodes.extend(expand_to_raw_ids(old_id))

        # Stitch sequences using old overlaps with support-aware overlap handling
        parts = []
        prev_old = None
        prev_node = None
        
        for j, old_id in enumerate(expanded_old_nodes):
            if str(old_id) not in raw_reads:
                print(f"Warning: raw read for id {old_id} not found; skipping this id")
                prev_old = old_id
                continue
                
            seq = raw_reads[str(old_id)]
            
            if prev_old is None or str(prev_old) not in raw_reads:
                parts.append(seq)
            else:
                ol = old_overlap[(prev_old, old_id)]
                ol_int = int(ol)
                
                # Always use support-aware overlap handling in this method
                if prev_node is not None:
                    # Find the current node in the walk that corresponds to this old_id
                    curr_node = None
                    for node in filtered_walk:
                        if str(old_id) in raw_reads and str(old_id) == str(node):
                            curr_node = node
                            break
                        # Check if old_id is in the expanded chain for this node
                        if node in utg_map:
                            for child in utg_map[node]:
                                if str(old_id) in raw_reads and str(old_id) == str(child):
                                    curr_node = node
                                    break
                            if curr_node:
                                break
                    
                    if curr_node is not None:
                        prev_support = support.get(prev_node, 0.0)
                        curr_support = support.get(curr_node, 0.0)
                        
                        if curr_support > prev_support:
                            # Current node has better support: add full sequence, remove overlap from previous
                            if parts:
                                # Remove the last overlap_length basepairs from the previous sequence
                                parts[-1] = parts[-1][:-ol_int]
                            parts.append(seq)
                            print(f"Support-aware overlap: node {curr_node} (support: {curr_support}) preferred over {prev_node} (support: {prev_support}), using full sequence")
                        else:
                            # Previous node has better support: add sequence starting from overlap
                            parts.append(seq[ol_int:])
                            print(f"Support-aware overlap: node {prev_node} (support: {prev_support}) preferred over {curr_node} (support: {curr_support}), using sequence from overlap")
                    else:
                        # Fallback to standard behavior if we can't map old_id to a node
                        parts.append(seq[ol_int:])
                else:
                    # Standard behavior: add sequence starting from overlap
                    parts.append(seq[ol_int:])
            
            prev_old = old_id
            # Update prev_node for next iteration
            if j < len(expanded_old_nodes) - 1:
                # Find the node in the walk that corresponds to the next old_id
                next_old_id = expanded_old_nodes[j + 1]
                for node in filtered_walk:
                    if str(next_old_id) in raw_reads and str(next_old_id) == str(node):
                        prev_node = node
                        break
                    # Check if next_old_id is in the expanded chain for this node
                    if node in utg_map:
                        for child in utg_map[node]:
                            if str(next_old_id) in raw_reads and str(next_old_id) == str(child):
                                prev_node = node
                                break
                        if prev_node:
                            break

        contig_seq = ''.join(parts)
        contig = Seq(contig_seq)
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)
    return contigs

def save_walks_and_sequences(nx_graph, walks, n2s, diploid, out_path, use_support_aware=False, support_aware_method=1):
    if diploid:
        mat_walks, pat_walks = walks
        print(f"\nFound {len(mat_walks)} maternal and {len(pat_walks)} paternal paths")
        
        # Create single output directory
        os.makedirs(out_path, exist_ok=True)
        
        # Save maternal and paternal walks
        pickle.dump(mat_walks, open(os.path.join(out_path, 'walks_maternal.pkl'), 'wb'))
        pickle.dump(pat_walks, open(os.path.join(out_path, 'walks_paternal.pkl'), 'wb'))
        
        # Generate maternal and paternal contigs
        if use_support_aware:
            if support_aware_method == 1:
                mat_contigs = walk_to_sequence_support_aware(mat_walks, nx_graph, n2s)
                pat_contigs = walk_to_sequence_support_aware(pat_walks, nx_graph, n2s)
            elif support_aware_method == 2:
                mat_contigs = walk_to_sequence_support_aware_2(mat_walks, nx_graph, n2s)
                pat_contigs = walk_to_sequence_support_aware_2(pat_walks, nx_graph, n2s)
            elif support_aware_method == 3: # Added consensus voting
                mat_contigs = walk_to_sequence_consensus_baselevel(mat_walks, nx_graph, n2s)
                pat_contigs = walk_to_sequence_consensus_baselevel(pat_walks, nx_graph, n2s)
            elif support_aware_method == 4: # Added wrong kmer
                mat_contigs = walk_to_sequence_wrong_kmer(mat_walks, nx_graph, n2s, 'maternal')
                pat_contigs = walk_to_sequence_wrong_kmer(pat_walks, nx_graph, n2s, 'paternal')
            else:
                raise ValueError(f"Unsupported support_aware_method: {support_aware_method}")
        else:
            mat_contigs = walk_to_sequence(mat_walks, nx_graph, n2s)
            pat_contigs = walk_to_sequence(pat_walks, nx_graph, n2s)
        
        # Save maternal contigs as hap2.fasta (maternal)
        hap2_path = os.path.join(out_path, 'hap2.fasta')
        SeqIO.write(mat_contigs, hap2_path, 'fasta')
        
        # Save paternal contigs as hap1.fasta (paternal)
        hap1_path = os.path.join(out_path, 'hap1.fasta')
        SeqIO.write(pat_contigs, hap1_path, 'fasta')
        
        # Also save with original naming for compatibility with evaluation functions
        eval.save_assembly(mat_contigs, out_path, 0, '_maternal')
        eval.save_assembly(pat_contigs, out_path, 0, '_paternal')
        
        print(f"\nSaved diploid assemblies to: {out_path}")
        print(f"  Maternal -> hap2.fasta")
        print(f"  Paternal -> hap1.fasta")
        
    else:
        # Original non-diploid code
        hap_dir = os.path.join(out_path)
        os.makedirs(hap_dir, exist_ok=True)
        pickle.dump(walks, open(os.path.join(out_path, 'walks.pkl'), 'wb'))
        
        if use_support_aware:
            if support_aware_method == 1:
                contigs = walk_to_sequence_support_aware(walks, nx_graph, n2s)
            elif support_aware_method == 2:
                contigs = walk_to_sequence_support_aware_2(walks, nx_graph, n2s)
            elif support_aware_method == 3: # Added consensus voting
                contigs = walk_to_sequence_consensus_baselevel(walks, nx_graph, n2s)
            elif support_aware_method == 4: # Added wrong kmer
                contigs = walk_to_sequence_wrong_kmer(walks, nx_graph, n2s, 'maternal')
            else:
                raise ValueError(f"Unsupported support_aware_method: {support_aware_method}")
        else:
            contigs = walk_to_sequence(walks, nx_graph, n2s)
        eval.save_assembly(contigs, out_path, 0)

def save_walks_and_sequences_classic(nx_graph, walks, utg_node_to_node, raw_reads, old_graph, diploid, out_path):
    """Save walks and sequences using classic old-node stitching with overlaps from old graph."""
    if diploid:
        mat_walks, pat_walks = walks
        print(f"\nFound {len(mat_walks)} maternal and {len(pat_walks)} paternal paths (classic)")
        os.makedirs(out_path, exist_ok=True)
        pickle.dump(mat_walks, open(os.path.join(out_path, 'walks_maternal.pkl'), 'wb'))
        pickle.dump(pat_walks, open(os.path.join(out_path, 'walks_paternal.pkl'), 'wb'))
        mat_contigs = walk_to_sequence_classic(mat_walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        pat_contigs = walk_to_sequence_classic(pat_walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        hap2_path = os.path.join(out_path, 'hap2.fasta')
        SeqIO.write(mat_contigs, hap2_path, 'fasta')
        hap1_path = os.path.join(out_path, 'hap1.fasta')
        SeqIO.write(pat_contigs, hap1_path, 'fasta')
        eval.save_assembly(mat_contigs, out_path, 0, '_maternal')
        eval.save_assembly(pat_contigs, out_path, 0, '_paternal')
        print(f"\nSaved diploid assemblies to: {out_path} (classic)")
        print(f"  Maternal -> hap2.fasta")
        print(f"  Paternal -> hap1.fasta")
    else:
        hap_dir = os.path.join(out_path)
        os.makedirs(hap_dir, exist_ok=True)
        pickle.dump(walks, open(os.path.join(out_path, 'walks.pkl'), 'wb'))
        contigs = walk_to_sequence_classic(walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        eval.save_assembly(contigs, out_path, 0)

def save_walks_and_sequences_classic_support_aware(nx_graph, walks, utg_node_to_node, raw_reads, old_graph, diploid, out_path):
    """Save walks and sequences using classic old-node stitching with overlaps from old graph and support-aware filtering."""
    if diploid:
        mat_walks, pat_walks = walks
        print(f"\nFound {len(mat_walks)} maternal and {len(pat_walks)} paternal paths (classic with support-aware filtering)")
        os.makedirs(out_path, exist_ok=True)
        pickle.dump(mat_walks, open(os.path.join(out_path, 'walks_maternal.pkl'), 'wb'))
        pickle.dump(pat_walks, open(os.path.join(out_path, 'walks_paternal.pkl'), 'wb'))
        mat_contigs = walk_to_sequence_classic_support_aware(mat_walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        pat_contigs = walk_to_sequence_classic_support_aware(pat_walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        hap2_path = os.path.join(out_path, 'hap2.fasta')
        SeqIO.write(mat_contigs, hap2_path, 'fasta')
        hap1_path = os.path.join(out_path, 'hap1.fasta')
        SeqIO.write(pat_contigs, hap1_path, 'fasta')
        eval.save_assembly(mat_contigs, out_path, 0, '_maternal')
        eval.save_assembly(pat_contigs, out_path, 0, '_paternal')
        print(f"\nSaved diploid assemblies to: {out_path} (classic with support-aware filtering)")
        print(f"  Maternal -> hap2.fasta")
        print(f"  Paternal -> hap1.fasta")
    else:
        hap_dir = os.path.join(out_path)
        os.makedirs(hap_dir, exist_ok=True)
        pickle.dump(walks, open(os.path.join(out_path, 'walks.pkl'), 'wb'))
        contigs = walk_to_sequence_classic_support_aware(walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        eval.save_assembly(contigs, out_path, 0)

def save_walks_and_sequences_support_aware(nx_graph, walks, n2s, diploid, out_path, n2s_method='support_aware'):
    """
    Save walks and sequences using support-aware sequence assembly.
    This function always uses the support-aware version of walk_to_sequence.
    
    Args:
        n2s_method: 'support_aware' for edge-based method, 'support_baselevel' for position-based method, 'consensus_baselevel' for consensus voting, 'wrong_kmer' for wrong kmer counts
    """
    use_support_aware = True
    if n2s_method == 'support_baselevel':
        support_aware_method = 2
    elif n2s_method == 'consensus_baselevel':
        support_aware_method = 3
    elif n2s_method == 'wrong_kmer':
        support_aware_method = 4
    else:
        support_aware_method = 1
    return save_walks_and_sequences(nx_graph, walks, n2s, diploid, out_path, use_support_aware=use_support_aware, support_aware_method=support_aware_method)

def save_walks_and_sequences_classic_support_and_overlap_aware(nx_graph, walks, utg_node_to_node, raw_reads, old_graph, diploid, out_path):
    """Save walks and sequences using classic old-node stitching with overlaps from old graph, support-aware filtering, and support-aware overlap handling."""
    if diploid:
        mat_walks, pat_walks = walks
        print(f"\nFound {len(mat_walks)} maternal and {len(pat_walks)} paternal paths (classic with support-aware filtering and overlap handling)")
        os.makedirs(out_path, exist_ok=True)
        pickle.dump(mat_walks, open(os.path.join(out_path, 'walks_maternal.pkl'), 'wb'))
        pickle.dump(pat_walks, open(os.path.join(out_path, 'walks_paternal.pkl'), 'wb'))
        mat_contigs = walk_to_sequence_classic_support_and_overlap_aware(mat_walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        pat_contigs = walk_to_sequence_classic_support_and_overlap_aware(pat_walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        hap2_path = os.path.join(out_path, 'hap2.fasta')
        SeqIO.write(mat_contigs, hap2_path, 'fasta')
        hap1_path = os.path.join(out_path, 'hap1.fasta')
        SeqIO.write(pat_contigs, hap1_path, 'fasta')
        eval.save_assembly(mat_contigs, out_path, 0, '_maternal')
        eval.save_assembly(pat_contigs, out_path, 0, '_paternal')
        print(f"\nSaved diploid assemblies to: {out_path} (classic with support-aware filtering and overlap handling)")
        print(f"  Maternal -> hap2.fasta")
        print(f"  Paternal -> hap1.fasta")
    else:
        hap_dir = os.path.join(out_path)
        os.makedirs(hap_dir, exist_ok=True)
        pickle.dump(walks, open(os.path.join(out_path, 'walks.pkl'), 'wb'))
        contigs = walk_to_sequence_classic_support_and_overlap_aware(walks, nx_graph, utg_node_to_node, raw_reads, old_graph)
        eval.save_assembly(contigs, out_path, 0)

def get_refs(args, diploid, single_chrom=False, ref_key=None):
    """
    Get reference paths based on configuration and arguments.
    
    Args:
        args: Command line arguments
        ref_config: Reference configuration dictionary
        diploid: Boolean indicating if diploid mode should be used
        single_chrom: Boolean indicating if processing a single chromosome
        ref_key: Reference key for specific genome references
    
    Returns:
        Dictionary containing reference paths and related information
    """

    # Load decoding paths configuration
    with open('configs/decoding_paths.yml', 'r') as file:
        decoding_paths = yaml.safe_load(file)
    # Get reference paths based on the provided key
    ref_config = decoding_paths[ref_key]
    refs = {}
    
    if diploid:
        if single_chrom:
            chr_literal = re.findall(r'chr\d+(?:_c)?', args.filename)[0]
            if chr_literal.endswith('_c'):
                chr_refs = ref_config['cent_refs']
                chr_nocent = chr_literal.split('_')[0]
                ref_m = os.path.join(chr_refs, f'{chr_nocent}_M_c.fasta')
                ref_p = os.path.join(chr_refs, f'{chr_nocent}_P_c.fasta')
            else:
                chr_refs = ref_config['chr_refs']
                ref_m = os.path.join(chr_refs, f'{chr_literal}_M.fasta')
                ref_p = os.path.join(chr_refs, f'{chr_literal}_P.fasta')
        else:
            ref_m = ref_config['mat_ref']
            ref_p = ref_config['pat_ref']
        
        refs['ref_m'] = ref_m
        refs['ref_p'] = ref_p
        refs['mat_yak'] = ref_config['mat_yak']
        refs['pat_yak'] = ref_config['pat_yak']
        refs['idx_m'] = ref_m + '.fai'
        refs['idx_p'] = ref_p + '.fai'
    else:
        if single_chrom:
            chr_literal = re.findall(r'chr\d+(?:_[MP])?(?:_c)?', args.filename)[0]
            if chr_literal.endswith('_c'):
                chr_refs = ref_config['cent_refs']
            else:
                chr_refs = ref_config['chr_refs']
            
            if ref_key in ['arabidopsis', 'arabidopsis_col_cc']:
                ref = os.path.join(chr_refs, f'{chr_literal}.fasta')
            elif ref_key in ['hg002']:
                ref = os.path.join(chr_refs, f'{chr_literal}_MATERNAL.fasta')
            else:
                ref = os.path.join(chr_refs, f'{chr_literal}_M.fasta')
        else:
            ref = ref_config['full_ref']
        
        refs['ref'] = ref
        refs['idx'] = ref + '.fai'

    return refs

def evaluate_synthetic(graph, config, out_path, diploid):
            # Create maternal and paternal directories
    # Check if graph has gt_bin attribute
    if not nx.get_edge_attributes(graph, 'gt_bin'):
        return

    if diploid:
        walks_p = pickle.load(open(os.path.join(out_path, 'walks_paternal.pkl'), 'rb'))
        walks_m = pickle.load(open(os.path.join(out_path, 'walks_maternal.pkl'), 'rb'))
        print(f"Loaded {len(walks_p)} paternal walks and {len(walks_m)} maternal walks")
    else:
        walks = pickle.load(open(os.path.join(out_path, 'walks.pkl'), 'rb'))

    if diploid:
        print(f"Eval correctness of edges Haplo p")
        eval.synthetic_edge_correctness(graph, walks_p)
        print(f"Eval correctness of edges Haplo m")
        eval.synthetic_edge_correctness(graph, walks_m)
        node_gt = nx.get_node_attributes(graph, 'yak_p')
        print(f"Eval correctness Phasing Haplo p")
        eval.synthetic_phasing_errors(walks_p, node_gt)
        node_gt = nx.get_node_attributes(graph, 'yak_m')
        print(f"Eval correctness Phasing Haplo m")
        eval.synthetic_phasing_errors(walks_m, node_gt)

        node_length = nx.get_node_attributes(graph, 'read_length')
        overlap_length = nx.get_edge_attributes(graph, 'overlap_length')
        kmer_hits_m = nx.get_node_attributes(graph, 'kmer_count_m')
        kmer_hits_p = nx.get_node_attributes(graph, 'kmer_count_p')
        eval.synthetic_kmer_evaluation(walks_p, kmer_hits_p, kmer_hits_m, node_length, overlap_length)
        eval.synthetic_kmer_evaluation(walks_m, kmer_hits_m, kmer_hits_p, node_length, overlap_length)

    else:
        print(f"Eval correctness of edges")
        eval.synthetic_edge_correctness(graph, walks)

def evaluate_real(ref, config, out_path, diploid, threads=16):

    if diploid:
        print('\nEvaluating diploid assemblies...')

        pat_asm = os.path.join(out_path, '0_assembly_paternal.fasta')
        pat_report = os.path.join(out_path, 'minigraph_paternal.txt')
        pat_paf = os.path.join(out_path, 'asm_paternal.paf')
        pat_phs = os.path.join(out_path, 'phs_paternal.txt')

        mat_asm = os.path.join(out_path, '0_assembly_maternal.fasta')
        mat_report = os.path.join(out_path, 'minigraph_maternal.txt')
        mat_paf = os.path.join(out_path, 'asm_maternal.paf')
        mat_phs = os.path.join(out_path, 'phs_maternal.txt')

        # Evaluate paternal assembly
        p = eval.run_minigraph(ref['ref_p'], pat_asm, pat_paf, minigraph_path=config['paths']['minigraph_path'], threads=threads)
        p.wait()
        p = eval.parse_pafs(ref['idx_p'], pat_report, pat_paf, paf_path=config['paths']['paftools_path'])
        p.wait()
        eval.parse_minigraph_for_full(pat_report)

        # Evaluate maternal assembly
        p = eval.run_minigraph(ref['ref_m'], mat_asm, mat_paf, minigraph_path=config['paths']['minigraph_path'], threads=threads)
        p.wait()
        p = eval.parse_pafs(ref['idx_m'], mat_report, mat_paf, paf_path=config['paths']['paftools_path'])
        p.wait()
        eval.parse_minigraph_for_full(mat_report)

        # Run YAK for phasing evaluation sequentially
        # First run maternal YAK evaluation
        p1 = eval.run_yak(ref['mat_yak'], ref['pat_yak'], mat_asm, mat_phs, yak_path=config['paths']['yak_path'], threads=threads)
        p1.wait()
        
        # Then run paternal YAK evaluation
        p2 = eval.run_yak(ref['mat_yak'], ref['pat_yak'], pat_asm, pat_phs, yak_path=config['paths']['yak_path'], threads=threads)
        p2.wait()
        
        eval.parse_real_results(mat_report, pat_report, mat_phs, pat_phs)
        eval.get_LG90(ref['ref_p'], pat_asm)
        eval.get_LG90(ref['ref_m'], mat_asm)

    else:
        print('Evaluating...')
        asm = os.path.join(out_path, f'0_assembly.fasta')
        report = os.path.join(out_path, 'minigraph.txt')
        paf = os.path.join(out_path, f'asm.paf')
        p = eval.run_minigraph(ref['ref'], asm, paf, minigraph_path=config['paths']['minigraph_path'])
        p.wait()
        p = eval.parse_pafs(ref['idx'], report, paf, paf_path=config['paths']['paftools_path'])
        p.wait()
        #subprocess.run(['paftools.js', 'asmstat', idx, paf], check=True)
        eval.parse_minigraph_for_full(report)
        eval.get_LG90(ref['ref'], asm)


def main_entry(argv=None):
    import argparse
    import os
    import pickle
    import yaml
    import time
    import networkx as nx
    
    if argv is None:
        argv = sys.argv[1:]
    start_time = time.time()
    log("Starting inference process")
    
    parser = argparse.ArgumentParser(description='Compute edge scores for a graph')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--strategy_config', type=str, default='decoding_mod/decode_strategies.yml', help='Path to the config file')
    parser.add_argument('--config', type=str, default='configs/config.yml', help='Path to the config file')
    parser.add_argument('--strategy', type=str, default='baseline', help='Strategy for graph reduction')
    
    parser.add_argument('--threads', type=int, default=16, help='Number of threads for minigraph')

    # Add dataset and filename arguments similar to decode_nx.py
    parser.add_argument('--dataset', type=str, help='Dataset directory containing dgl_graphs, nx_graphs, etc.')
    parser.add_argument('--filename', type=str, help='Base filename without extension')
    parser.add_argument('--label', action='store_true', default=False, help='Use label information instead of computing scores')
    parser.add_argument('--ref', type=str, help='Reference key for specific genome references')
    parser.add_argument('--single_chrom', action='store_true', default=False, help='Process a single chromosome')
    parser.add_argument('--skip_real_eval', action='store_true', default=False, help='Skip real evaluation')
    parser.add_argument('--skip_synthetic_eval', action='store_true', default=False, help='Skip synthetic evaluation')
    parser.add_argument('--skip_decode', action='store_true', default=False, help='Skip decoding')
    parser.add_argument('--ass_out_dir', type=str, default=None, help='Output directory for assemblies')
    parser.add_argument('--load_node_scores', type=str, default=None, help='Path to pickled dictionary of node scores')
    parser.add_argument('--hifiasm_res_based', action='store_true', default=False, help='Use hifiasm_result as base score')
    parser.add_argument('--n2s', type=str, default='default', choices=['default', 'support_aware', 'support_baselevel', 'consensus_baselevel', 'wrong_kmer', 'classic_nodes', 'classic_nodes_support_aware', 'classic_nodes_support_and_overlap_aware'], help='Sequence assembly method (default: standard, support_aware: edge-based support-aware, support_baselevel: position-based support-aware, consensus_baselevel: consensus voting, wrong_kmer: wrong kmer counts, classic_nodes: build node->sequence from classic reads, classic_nodes_support_aware: classic nodes with support-aware filtering, classic_nodes_support_and_overlap_aware: classic nodes with support-aware filtering and overlap handling)')
    # Note: double_model is automatically determined based on strategy reduction type ('cut_and_top_p')

    args = parser.parse_args(argv)
    log(f"Arguments parsed: {vars(args)}")
    
    # Model is required if not using labels
    if not args.model and not args.label:
        parser.error("--model is required when not using --label")
    if not args.ass_out_dir:
        parser.error("--ass_out_dir is required")
    os.makedirs(args.ass_out_dir, exist_ok=True)
    log(f"Created output directory: {args.ass_out_dir}")

    # Extract model name from the model path for use in the output filename
    model_name = os.path.splitext(os.path.basename(args.model))[0] if args.model else "label_based"
    log(f"Using model: {model_name}")
    
    # Handle dataset and filename arguments
    if args.dataset and args.filename:
        dgl_path = os.path.join(args.dataset, 'dgl_graphs', args.filename + '.dgl')
        nx_path = os.path.join(args.dataset, 'nx_utg_graphs', args.filename + '.pkl')
        reads_path = os.path.join(args.dataset, 'reduced_reads', args.filename + '.fasta')
        log(f"Processing dataset: {args.dataset}")
        log(f"File: {args.filename}")
        log(f"DGL graph path: {dgl_path}")
        log(f"NetworkX graph path: {nx_path}")
        log(f"Reads path: {reads_path}")
        
        # Create model_scores directory in the dataset folder
        model_scores_dir = os.path.join(args.dataset, 'model_scores')
        os.makedirs(model_scores_dir, exist_ok=True)
        # Output filename combines model name and data sample name
        out_score_path = os.path.join(model_scores_dir, f"{model_name}_{args.filename}.pt")
        log(f"Output scores will be saved to: {out_score_path}")
    else:
        raise ValueError("Please set 'dataset' and 'filename' arguments")
    
    # Load config
    log(f"Loading configuration from {args.config}")
    with open(args.config) as file:
        config = yaml.safe_load(file)
    with open(args.strategy_config) as file:
        strategies = yaml.safe_load(file)
    
    # Automatically determine if double model should be used based on strategy
    strategy_config = strategies[args.strategy]
    double_model = strategy_config['double_model'] #strategy_config.get('reduction') == 'cut_and_top_p'

    log(f"Strategy '{args.strategy}' uses reduction: '{strategy_config.get('reduction')}' -> double_model: {double_model}")
    
    # Load NetworkX graph
    log(f"Loading NetworkX graph from {nx_path}")
    load_start = time.time()
    with open(nx_path, 'rb') as file:
        nx_graph = pickle.load(file)
    log(f"Graph loaded in {time.time() - load_start:.2f} seconds")
    log(f"Graph has {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
    
    # Load node scores if provided
    graphic_scores = False
    if args.load_node_scores:
        log(f"Loading node scores from {args.load_node_scores}")

        with open(args.load_node_scores, 'rb') as file:
            node_scores_raw = pickle.load(file)
        log(f"Loaded node scores for {len(node_scores_raw)} nodes")
        
        # Transform dictionary: each key k corresponds to nodes 2*k and 2*k+1 in nx graph
        node_scores = {}
        for k, score in node_scores_raw.items():
            node_scores[2 * k] = score      # node 2*k gets the score
            node_scores[2 * k + 1] = score  # node 2*k+1 gets the same score
        
        log(f"Transformed node scores to cover {len(node_scores)} nodes")
        
        # Set graphic_score attribute for nodes
        nx.set_node_attributes(nx_graph, node_scores, 'graphic_score')
        graphic_scores = True
        log("Node scores added to graph as 'graphic_score' attribute")

    
    # Compute scores if not using labels
    if args.label:
        log("Using ground truth labels instead of computing scores")
        gt_bin_scores = nx.get_edge_attributes(nx_graph, 'hifiasm_result')
        print(sum(gt_bin_scores.values()))
        #exit()

        if double_model:
            # For double models with labels, use gt_bin as 'score' and malicious as 'to_cut'
            log("Using double model with ground truth labels: gt_bin -> score, malicious -> to_cut")
            
            # Get gt_bin attributes for score
            scores = nx.get_edge_attributes(nx_graph, 'hifiasm_result')
            #unknown = nx.get_edge_attributes(nx_graph, 'unknown')
            #scores = {edge: 0.5 if unknown.get(edge, False) else gt_bin_scores[edge] for edge in nx_graph.edges()}
            for edge in nx_graph.edges():
                if edge not in scores:
                    scores[edge] = 0
            # Apply score flipping if configured
            if strategy_config.get('flip_score', False):
                print("Flipping scores: 1 - score")
                scores = {edge: 1 - score for edge, score in scores.items()}
            
            nx.set_edge_attributes(nx_graph, scores, 'score')
            
            # Get malicious attributes for to_cut
            malicious_scores = nx.get_edge_attributes(nx_graph, 'malicious')
            # Handle missing malicious attributes gracefully
            to_cut_scores = {edge: malicious_scores[edge] for edge in nx_graph.edges()}
            
            nx.set_edge_attributes(nx_graph, to_cut_scores, 'to_cut')
            
            log(f"Set {len(scores)} edges with 'score' from gt_bin")
            log(f"Set {len(to_cut_scores)} edges with 'to_cut' from malicious")
            
        else:
            # Original single model label handling
            if strategies[args.strategy]['gt_score'] == 'hifiasm_result':
                scores = nx.get_edge_attributes(nx_graph, 'hifiasm_result')
                # Fill edges that don't have hifiasm_result attribute with 0
                for edge in nx_graph.edges():
                    if edge not in scores:
                        scores[edge] = 0
                #unknown = nx.get_edge_attributes(nx_graph, 'unknown')
                #scores = {edge: 0.5 if unknown.get(edge, False) else scores_[edge] for edge in nx_graph.edges()}
                #scores_1 = nx.get_edge_attributes(nx_graph, 'strand_change')
                #scores_0 = nx.get_edge_attributes(nx_graph, 'cross_chr')
                #scores = {edge: 0 if scores_1[edge] or scores_0[edge] else 1 for edge in nx_graph.edges()}
            elif strategies[args.strategy]['gt_score'] == 'uniform':
                scores = {edge: 1 for edge in nx_graph.edges()}
            else:
                scores = nx.get_edge_attributes(nx_graph, strategies[args.strategy]['gt_score'])
            
            # Apply score flipping if configured
            if strategy_config.get('flip_score', False):
                print("Flipping scores: 1 - score")
                scores = {edge: 1 - score for edge, score in scores.items()}
            
            nx.set_edge_attributes(nx_graph, scores, 'score')
            nx.set_edge_attributes(nx_graph, scores, 'hifiasm_result')

    else:
        save_dict = compute_scores(dgl_path, args.model, config, out_score_path, device='cpu', double_model=double_model)
        # Add scores to NetworkX graph
        print("Adding scores to NetworkX graph")
        edge_scores = save_dict['edge_scores']
        
        # Print debug information
        print(f"DGL graph provided scores for {len(edge_scores)} edges")
        print(f"NetworkX graph has {nx_graph.number_of_edges()} edges")
        
        # Check if all edges have scores
        missing_edges = []
        for edge in nx_graph.edges():
            if edge not in edge_scores:
                missing_edges.append(edge)
        
        if missing_edges:
            print(f"Warning: {len(missing_edges)} edges missing scores")
            print(f"First few missing edges: {missing_edges[:5]}")
            
            # Check if missing edges have the required attributes
            sample_missing_edges = missing_edges[:10]  # Check first 10 missing edges
            print("\nChecking attributes of missing edges:")
            for edge in sample_missing_edges:
                src, dst = edge
                edge_data = nx_graph[src][dst]
                has_overlap_length = 'overlap_length' in edge_data
                has_overlap_similarity = 'overlap_similarity' in edge_data
                has_prefix_length = 'prefix_length' in edge_data
                print(f"Edge {edge}: overlap_length={has_overlap_length}, overlap_similarity={has_overlap_similarity}, prefix_length={has_prefix_length}")
            
            # Assign default score to missing edges
            print(f"Assigning default score of 0.5 to {len(missing_edges)} missing edges")
            for edge in missing_edges:
                edge_scores[edge] = 0.5  # Default neutral score
                # Also assign default cut score for double models
                if double_model and 'cut_scores' in save_dict:
                    save_dict['cut_scores'][edge] = 0.5
        
        # Apply score flipping if configured
        if strategy_config.get('flip_score', False):
            print("Flipping edge scores: 1 - score")
            edge_scores = {edge: 1 - score for edge, score in edge_scores.items()}

        # Add scores to graph
        nx.set_edge_attributes(nx_graph, edge_scores, 'score')
        
        # For double models, also add cut scores as 'to_cut'
        if double_model and 'cut_scores' in save_dict:
            cut_scores = save_dict['cut_scores']
            
            nx.set_edge_attributes(nx_graph, cut_scores, 'to_cut')
            print(f"Added 'to_cut' scores from malicious head for double model")
    
    diploid = bool(nx.get_node_attributes(nx_graph, 'yak_region'))  # True if dict is not empty, False otherwise
    print(f"Diploid mode: {diploid}")
    
    if args.n2s == 'support_aware':
        log("Using support-aware sequence assembly for overlaps")
        log("Using original support-aware method (edge-based)")
    elif args.n2s == 'support_baselevel':
        log("Using support-aware sequence assembly for overlaps")
        log("Using position-based support-aware method (basepair-level)")
    elif args.n2s == 'consensus_baselevel':
        log("Using support-aware sequence assembly for overlaps")
        log("Using consensus voting basepair-level method")
    elif args.n2s == 'wrong_kmer':
        log("Using support-aware sequence assembly for overlaps")
        log("Using wrong kmer counts basepair-level method")
    elif args.n2s == 'classic_nodes':
        log("Using classic node-based sequence assembly (prefix stitching on classic reads)")
    elif args.n2s == 'classic_nodes_support_aware':
        log("Using classic node-based sequence assembly with support-aware filtering")
    elif args.n2s == 'classic_nodes_support_and_overlap_aware':
        log("Using classic node-based sequence assembly with support-aware filtering and overlap handling")
    
    if not args.skip_decode:
        # Start inference with diploid parameter and graphic_scores
        if args.hifiasm_res_based:
            walks = hifiasm_res_based_inference.get_walks(nx_graph, strategies[args.strategy], diploid, graphic_scores=graphic_scores)
        else:
            walks = inference.get_walks(nx_graph, strategies[args.strategy], diploid, graphic_scores=graphic_scores)
 
        if args.n2s == 'classic_nodes':
            # Build node->sequence by expanding to old read ids using UTG mapping, then assemble using old overlaps
            reads_path_raw = os.path.join(args.dataset, 'reduced_reads_raw', args.filename + '.fasta')
            utg_map_path = os.path.join(args.dataset, 'utg_node_to_node', args.filename + '.pkl')
            old_nx_path = os.path.join(args.dataset, 'nx_full_graphs', args.filename + '.pkl')
            log(f"Classic reads (raw) path: {reads_path_raw}")
            log(f"UTG node-to-node map path: {utg_map_path}")
            log(f"Old NX graph path: {old_nx_path}")
            classic_reads = load_reads(reads_path_raw)
            with open(utg_map_path, 'rb') as f:
                utg_node_to_node = pickle.load(f)
            with open(old_nx_path, 'rb') as f:
                old_nx_graph = pickle.load(f)
 
            save_walks_and_sequences_classic(nx_graph, walks, utg_node_to_node, classic_reads, old_nx_graph, diploid, args.ass_out_dir)
        elif args.n2s == 'classic_nodes_support_aware':
            # Build node->sequence by expanding to old read ids using UTG mapping, then assemble using old overlaps with support-aware filtering
            reads_path_raw = os.path.join(args.dataset, 'reduced_reads_raw', args.filename + '.fasta')
            utg_map_path = os.path.join(args.dataset, 'utg_node_to_node', args.filename + '.pkl')
            old_nx_path = os.path.join(args.dataset, 'nx_full_graphs', args.filename + '.pkl')
            log(f"Classic reads (raw) path: {reads_path_raw}")
            log(f"UTG node-to-node map path: {utg_map_path}")
            log(f"Old NX graph path: {old_nx_path}")
            classic_reads = load_reads(reads_path_raw)
            with open(utg_map_path, 'rb') as f:
                utg_node_to_node = pickle.load(f)
            with open(old_nx_path, 'rb') as f:
                old_nx_graph = pickle.load(f)
 
            save_walks_and_sequences_classic_support_aware(nx_graph, walks, utg_node_to_node, classic_reads, old_nx_graph, diploid, args.ass_out_dir)
        elif args.n2s == 'classic_nodes_support_and_overlap_aware':
            # Build node->sequence by expanding to old read ids using UTG mapping, then assemble using old overlaps with support-aware filtering and overlap handling
            reads_path_raw = os.path.join(args.dataset, 'reduced_reads_raw', args.filename + '.fasta')
            utg_map_path = os.path.join(args.dataset, 'utg_node_to_node', args.filename + '.pkl')
            old_nx_path = os.path.join(args.dataset, 'nx_full_graphs', args.filename + '.pkl')
            log(f"Classic reads (raw) path: {reads_path_raw}")
            log(f"UTG node-to-node map path: {utg_map_path}")
            log(f"Old NX graph path: {old_nx_path}")
            classic_reads = load_reads(reads_path_raw)
            with open(utg_map_path, 'rb') as f:
                utg_node_to_node = pickle.load(f)
            with open(old_nx_path, 'rb') as f:
                old_nx_graph = pickle.load(f)
 
            save_walks_and_sequences_classic_support_and_overlap_aware(nx_graph, walks, utg_node_to_node, classic_reads, old_nx_graph, diploid, args.ass_out_dir)
        else:
            reads = load_reads(reads_path)
            
            # Determine support_aware_method based on n2s argument
            use_support_aware = args.n2s in ['support_aware', 'support_baselevel', 'consensus_baselevel', 'wrong_kmer']
            if args.n2s == 'support_baselevel':
                support_aware_method = 2
            elif args.n2s == 'consensus_baselevel':
                support_aware_method = 3
            elif args.n2s == 'wrong_kmer':
                support_aware_method = 4
            else:
                support_aware_method = 1
            
            save_walks_and_sequences(nx_graph, walks, reads, diploid, args.ass_out_dir, use_support_aware=use_support_aware, support_aware_method=support_aware_method)

    if not args.skip_synthetic_eval:
        evaluate_synthetic(nx_graph, config, args.ass_out_dir, diploid)
    if not args.skip_real_eval:
        refs = get_refs(args, diploid, args.single_chrom, args.ref)
        evaluate_real(refs, config, args.ass_out_dir, diploid, threads=args.threads)

    print("Done!")

if __name__ == '__main__':
    main_entry()
