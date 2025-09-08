import pickle
import networkx as nx
from Bio import SeqIO
from Bio import Seq
import subprocess
import os
import re

def run_yak(mat_yak, pat_yak, asm, outfile, yak_path, threads=8):
    cmd = f'{yak_path} trioeval -t{threads} {pat_yak} {mat_yak} {asm} > {outfile}'.split(' ')
    with open(outfile, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def synthetic_edge_correctness(graph, walks):
    correct_edges = 0
    wrong_edges = 0
    strand_change_errors = 0
    cross_chr_errors = 0
    skip_errors = 0
    deadend_errors = 0
    
    gt_bin = nx.get_edge_attributes(graph, 'gt_bin')
    strand_changes = nx.get_edge_attributes(graph, 'cross_strand')
    cross_chr = nx.get_edge_attributes(graph, 'cross_chr')
    skip = nx.get_edge_attributes(graph, 'skip')
    dead_end = nx.get_edge_attributes(graph, 'dead_end')

    """print(len(gt_bin.keys()))
    print(len(strand_changes.keys()))
    print(len(cross_chr.keys()))
    print(len(skip.keys()))
    print(len(dead_end.keys()))
    exit()"""
    for walk in walks:
        for n in range(len(walk)-1):
            edge = (walk[n], walk[n+1])
            if edge not in gt_bin:
                print(f'Edge {edge} not in gt_bin. Exiting')
                print(f"gt bin contains {len(gt_bin.keys())} edges")
                exit()
            gt_edge = gt_bin[edge]
            
            if gt_edge == 1:
                correct_edges += 1
            elif gt_edge == 0:
                wrong_edges += 1
                
                # Count specific error types - use .get() to handle missing attributes
                if strand_changes.get(edge, 0) == 1:
                    strand_change_errors += 1
                if cross_chr.get(edge, 0) == 1:
                    cross_chr_errors += 1
                if skip.get(edge, 0) == 1:
                    skip_errors += 1
                if dead_end.get(edge, 0) == 1:
                    deadend_errors += 1
    
    print(f'Correct Edges: {correct_edges}')
    print(f'Wrong Edges: {wrong_edges}')
    print(f'  - Strand Change Errors: {strand_change_errors}')
    print(f'  - Cross Chromosome Errors: {cross_chr_errors}')
    print(f'  - Skip Errors: {skip_errors}')
    print(f'  - Deadend Errors: {deadend_errors}')
    
    if wrong_edges + correct_edges == 0:
        print('No edges to calculate error ratio')
        return 0
    
    error_ratio = wrong_edges / (wrong_edges + correct_edges)
    print(f'Error Ratio: {error_ratio:.4f}')
    
    return error_ratio

def synthetic_phasing_errors(walks, node_gt):
    total_edges = 0
    switch_errors = 0
    hamming_errors = 0
    
    for walk in walks:
        prev_gt = None
        for n in range(len(walk)-1):
            curr_node_gt = node_gt[walk[n]]
            next_node_gt = node_gt[walk[n+1]]
            total_edges += 1
            
            if curr_node_gt != 0 and next_node_gt != 0:  # Only consider non-neutral nodes for switch errors
                if prev_gt is not None and curr_node_gt != prev_gt:
                    switch_errors += 1
                prev_gt = curr_node_gt
                    
            if curr_node_gt == -1 or next_node_gt == -1:  # Count hamming errors if either node is wrong
                hamming_errors += 1
                
    print(f'Total Edges: {total_edges}')
    print(f'Switch Errors: {switch_errors}')
    print(f'Hamming Errors: {hamming_errors}')
    
    if total_edges == 0:
        return 0, 0
        
    switch_error_rate = switch_errors / total_edges
    hamming_error_rate = hamming_errors / total_edges
    
    return switch_error_rate, hamming_error_rate

def synthetic_kmer_evaluation(walks, kmer_hits, wrong_kmer_hits, node_length, overlap_length):
    """
    Evaluate kmer hits along assembly walks.
    
    For each edge, compute the approximate kmer hits in the sequence added to assembly:
    (target_node_length - overlap_length) * (kmer_hits / full_read_length)
    
    Args:
        walks: List of walks (sequences of node IDs)
        kmer_hits: Dictionary mapping node IDs to correct kmer hits
        wrong_kmer_hits: Dictionary mapping node IDs to wrong kmer hits  
        node_length: Dictionary mapping node IDs to full read lengths
        overlap_length: Dictionary mapping edges to overlap lengths
        
    Returns:
        kmer_error_rate: wrong_kmer_hits_total / (correct_kmer_hits_total + wrong_kmer_hits_total)
    """
    total_correct_kmer_hits = 0
    total_wrong_kmer_hits = 0
    total_edges = 0
    
    for walk in walks:
        for n in range(len(walk)-1):
            curr_node = walk[n]
            next_node = walk[n+1]
            edge = (curr_node, next_node)
            total_edges += 1
            
            # Calculate sequence length added to assembly (target node minus overlap)
            target_node_length = node_length[next_node]
            edge_overlap = overlap_length[edge]
            sequence_added = target_node_length - edge_overlap
            
            # Calculate fraction of the read that's added to assembly
            full_read_length = node_length[next_node]
            fraction_added = sequence_added / full_read_length if full_read_length > 0 else 0
            
            # Calculate approximate kmer hits in the added sequence
            correct_kmers_added = fraction_added * kmer_hits[next_node]
            wrong_kmers_added = fraction_added * wrong_kmer_hits[next_node]
            
            total_correct_kmer_hits += correct_kmers_added
            total_wrong_kmer_hits += wrong_kmers_added
    
    print(f'Total Edges: {total_edges}')
    print(f'Total Correct Kmer Hits: {total_correct_kmer_hits:.2f}')
    print(f'Total Wrong Kmer Hits: {total_wrong_kmer_hits:.2f}')
    
    if total_correct_kmer_hits + total_wrong_kmer_hits == 0:
        print('No kmer hits to evaluate')
        return 0
    
    kmer_error_rate = total_wrong_kmer_hits / (total_correct_kmer_hits + total_wrong_kmer_hits)
    print(f'Synthetic Hamming Error: {kmer_error_rate:.4f}')
    
    return kmer_error_rate

def parse_minigraph_result(stat_path):
    nga50 = 0
    ng50 = 0
    length = 0
    rdup = 0
    with open(stat_path) as f:
        for line in f.readlines():
            if line.startswith('NG50'):
                try:
                    ng50 = int(re.findall(r'NG50\s*(\d+)', line)[0])
                except IndexError:
                    ng50 = 0
            if line.startswith('NGA50'):
                try:
                    nga50 = int(re.findall(r'NGA50\s*(\d+)', line)[0])
                except IndexError:
                    nga50 = 0
            if line.startswith('Length'):
                try:
                    length = int(re.findall(r'Length\s*(\d+)', line)[0])
                except IndexError:
                    length = 0
            if line.startswith('Rdup'):
                try:
                    rdup = float(re.findall(r'Rdup\s*(\d+\.\d+)', line)[0])
                except IndexError:
                    rdup = 0

    return ng50, nga50, length, rdup

def parse_yak_result(yakres_path):
    """
    Yak triobinning result files have following info:
    C       F  seqName     type      startPos  endPos    count
    C       W  #switchErr  denominator  switchErrRate
    C       H  #hammingErr denominator  hammingErrRate
    C       N  #totPatKmer #totMatKmer  errRate
    """
    switch_err = None
    hamming_err = None

    with open(yakres_path, 'r') as file:
        # Read all the lines and reverse them
        lines = file.readlines()
        reversed_lines = reversed(lines)

        for line in reversed_lines:
            if line.startswith('W'):
                switch_err = float(line.split()[3])
            elif line.startswith('H'):
                hamming_err = float(line.split()[3])

            if switch_err is not None and hamming_err is not None:
                break

    return switch_err, hamming_err

def eval_synth_simple(out, nx_path):
    """Evaluate synthetic assembly results."""
    walks = pickle.load(open(os.path.join(out, 'walks.pkl'), 'rb'))
    nx_graph = pickle.load(open(nx_path, 'rb'))

    # Check if we have any walks
    if not walks:
        print("\nNo valid paths found in the graph!")
        print("\nEvaluation Statistics:")
        print("No paths to evaluate")
        return

    total_edges = 0
    correct_edges = 0
    strand_change_errors = 0
    skip_errors = 0
    both_errors = 0

    print("\nWalk    Length  Strand  Skips   Both    Total")
    print("----    ------  ------  -----   ----    -----")

    for i, walk in enumerate(walks):
        walk_edges = 0
        walk_correct = 0
        walk_strand = 0
        walk_skip = 0
        walk_both = 0

        for j in range(len(walk)-1):
            src, dst = walk[j], walk[j+1]
            if (src, dst) in nx_graph.edges():
                walk_edges += 1
                edge_data = nx_graph.edges[src, dst]
                
                if edge_data['strand_change'] and edge_data['skip']:
                    walk_both += 1
                elif edge_data['strand_change']:
                    walk_strand += 1
                elif edge_data['skip']:
                    walk_skip += 1
                else:
                    walk_correct += 1

        total_edges += walk_edges
        correct_edges += walk_correct
        strand_change_errors += walk_strand
        skip_errors += walk_skip
        both_errors += walk_both

        print(f"{i:<7d} {walk_edges:<7d} {walk_strand:<7d} {walk_skip:<7d} {walk_both:<7d} {walk_correct}")

    print("\nOverall Statistics:")
    print(f"Total edges: {total_edges}")
    print(f"Correct edges: {correct_edges}")
    print(f"Wrong edges (total): {total_edges - correct_edges}")
    print(f"  - Strand change errors: {strand_change_errors}")
    print(f"  - Skip errors: {skip_errors}")
    print(f"  - Both errors: {both_errors}")

    if total_edges > 0:
        print("\nError Ratios:")
        print(f"Overall error ratio: {(total_edges - correct_edges)/total_edges:.4f}")
        print(f"Strand change ratio: {(strand_change_errors + both_errors)/total_edges:.4f}")
        print(f"Skip ratio: {(skip_errors + both_errors)/total_edges:.4f}")
    else:
        print("\nError Ratios:")
        print("No edges to evaluate")

def eval_synth(walk_dir, nx_path, gt='gt_bin', diploid=False):
    nx_g = pickle.load(open(nx_path, 'rb'))
    ground_truth = nx.get_edge_attributes(nx_g, gt)
    read_start_dict = nx.get_node_attributes(nx_g, "read_start")
    read_end_dict = nx.get_node_attributes(nx_g, "read_end") 
    read_strand_dict = nx.get_node_attributes(nx_g, "read_strand")
    read_variant_dict = nx.get_node_attributes(nx_g, "read_variant")
    
    if diploid and gt == 'gt_m_soft':
        inference_path = os.path.join(walk_dir + "_m", f'walks.pkl')
    elif diploid and gt == 'gt_p_soft':
        inference_path = os.path.join(walk_dir + "_p", f'walks.pkl')
    else:
        inference_path = os.path.join(walk_dir, f'walks.pkl')

    # Create the dictionary
    wrong_edges_list = []
    with open(inference_path, 'rb') as file:
        walks = pickle.load(file)
    correct_edges_per_walk = []
    wrong_edges_per_walk = []
    overlap_errors = 0
    real_overlap_errors = 0
    strand_errors = 0
    
    for walk in walks:
        correct_edges = 0
        wrong_edges = 0
        for n in range(len(walk)-1):
            curr_node = walk[n]
            next_node = walk[n+1]
            
            # Check overlaps
            if read_strand_dict[curr_node] == 1:
                if read_start_dict[next_node] >= read_end_dict[curr_node] or read_start_dict[next_node] <= read_start_dict[curr_node]:
                    overlap_errors += 1
                    # Check if reads have same variant
                    if read_variant_dict[curr_node] == read_variant_dict[next_node]:
                        real_overlap_errors += 1
            else:
                if read_start_dict[curr_node] >= read_end_dict[next_node] or read_start_dict[curr_node] <= read_start_dict[next_node]:
                    overlap_errors += 1
                    # Check if reads have same variant
                    if read_variant_dict[curr_node] == read_variant_dict[next_node]:
                        real_overlap_errors += 1
                
            # Check strands match    
            if read_strand_dict[curr_node] != read_strand_dict[next_node]:
                strand_errors += 1
                
            gt_edge = ground_truth[(curr_node, next_node)]
            if gt_edge == 4:
                correct_edges += 1
            else:
                wrong_edges += 1
                wrong_edges_list.append((curr_node, next_node))

        correct_edges_per_walk.append(correct_edges)
        wrong_edges_per_walk.append(wrong_edges)
        
    correct_edges = sum(correct_edges_per_walk)
    wrong_edges = sum(wrong_edges_per_walk)
    total_edges = correct_edges + wrong_edges
    correct_edges_ratio = correct_edges / total_edges
    
    print(f"Overlap errors: {overlap_errors}")
    print(f"Real overlap errors (same variant): {real_overlap_errors}")
    print(f"Strand errors: {strand_errors}")


    print('###########################')
    print(f"Correct Edges: {'{:,}'.format(correct_edges)}")
    print(f"Wrong Edges: {wrong_edges}")
    print(f"Correct Edges Ratio: {correct_edges_ratio:.4f}")
    print('###########################')
    if diploid:
        # Initialize an empty dictionary to store the counts of each value
        value_counts = {}
        # Iterate through wrong_edges_list and populate the dictionary
        for edge in wrong_edges_list:
            value = ground_truth[edge]
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
        # Print the resulting dictionary
        print('Value counts for gt17c values in wrong_edges_list')
        for value, count in value_counts.items():
            print(f'{value}: {count}')
        print("###################")


def parse_real_results(mat_report, pat_report, mat_phs, pat_phs):
    ng50_m, nga50_m, length_m, rdup_m = parse_minigraph_result(mat_report)
    ng50_p, nga50_p, length_p, rdup_p = parse_minigraph_result(pat_report)
    switch_err_m, hamming_err_m = parse_yak_result(mat_phs)
    switch_err_p, hamming_err_p = parse_yak_result(pat_phs)
    print(f'Results:')
    print(f'Length: P:{"{:,}".format(length_p)} M:{"{:,}".format(length_m)} Avg:{"{:,}".format((length_m + length_p) // 2)}')
    print(f'Rdup: P:{rdup_p:.4f} M:{rdup_m:.4f} Avg:{(rdup_m + rdup_p) / 2:.4f}')
    print(f'NG50: P:{"{:,}".format(ng50_p)} M:{"{:,}".format(ng50_m)} Avg:{"{:,}".format((ng50_m + ng50_p) // 2)}')
    print(f'NGA50: P:{"{:,}".format(nga50_p)} M:{"{:,}".format(nga50_m)} Avg:{"{:,}".format((nga50_m + nga50_p) // 2)}')
    print(f'YAK Switch Err: P:{switch_err_p * 100:.4f}% M:{switch_err_m * 100:.4f}% Avg:{(switch_err_m + switch_err_p) / 2 * 100:.4f}%')
    print(f'YAK Hamming Err: P:{hamming_err_p * 100:.4f}% M:{hamming_err_m * 100:.4f}% Avg:{(hamming_err_m + hamming_err_p) / 2 * 100:.4f}%')
    #print(f'MERYL Switch Err: M:{mat_switch_error:.4f}% P:{pat_switch_error:.4f}% Avg:{(mat_switch_error + pat_switch_error) / 2:.4f}%')

def parse_real_results_haploid(mat_report, pat_report, mat_phs, pat_phs):
    ng50_m, nga50_m, length_m, rdup_m = parse_minigraph_result(mat_report)
    ng50_p, nga50_p, length_p, rdup_p = parse_minigraph_result(pat_report)
    switch_err_m, hamming_err_m = parse_yak_result(mat_phs)
    switch_err_p, hamming_err_p = parse_yak_result(pat_phs)
    print(f'Results:')
    print(f'Length: P:{"{:,}".format(length_p)} M:{"{:,}".format(length_m)} Avg:{"{:,}".format((length_m + length_p) // 2)}')
    print(f'Rdup: P:{rdup_p:.4f} M:{rdup_m:.4f} Avg:{(rdup_m + rdup_p) / 2:.4f}')
    print(f'NG50: P:{"{:,}".format(ng50_p)} M:{"{:,}".format(ng50_m)} Avg:{"{:,}".format((ng50_m + ng50_p) // 2)}')
    print(f'NGA50: P:{"{:,}".format(nga50_p)} M:{"{:,}".format(nga50_m)} Avg:{"{:,}".format((nga50_m + nga50_p) // 2)}')
    print(f'YAK Switch Err: P:{switch_err_p * 100:.4f}% M:{switch_err_m * 100:.4f}% Avg:{(switch_err_m + switch_err_p) / 2 * 100:.4f}%')
    print(f'YAK Hamming Err: P:{hamming_err_p * 100:.4f}% M:{hamming_err_m * 100:.4f}% Avg:{(hamming_err_m + hamming_err_p) / 2 * 100:.4f}%')
    #print(f'MERYL Switch Err: M:{mat_switch_error:.4f}% P:{pat_switch_error:.4f}% Avg:{(mat_switch_error + pat_switch_error) / 2:.4f}%')


def parse_mixed_results(mat_report, pat_report, pat_switch, mat_switch, pat_hamming, mat_hamming):
    ng50_m, nga50_m, length_m, rdup_m = parse_minigraph_result(mat_report)
    ng50_p, nga50_p, length_p, rdup_p = parse_minigraph_result(pat_report)

    print(f'Results:')
    print(f'Length: M:{"{:,}".format(length_m)} P:{"{:,}".format(length_p)} Avg:{"{:,}".format((length_m + length_p) / 2)}')
    print(f'Rdup: M:{"{:,}".format(rdup_m)} P:{"{:,}".format(rdup_p)} Avg:{"{:,}".format((rdup_m + rdup_p) / 2)}')
    print(f'NG50: M:{"{:,}".format(ng50_m)} P:{"{:,}".format(ng50_p)} Avg:{"{:,}".format((ng50_m + ng50_p) / 2)}')
    print(f'NGA50: M:{"{:,}".format(nga50_m)} P:{"{:,}".format(nga50_p)} Avg:{"{:,}".format((nga50_m + nga50_p) / 2)}')
    print(f'GT Switch Err: M:{mat_switch * 100:.4f}% P:{pat_switch * 100:.4f}% Avg:{(mat_switch + pat_switch) / 2 * 100:.4f}%')
    print(f'GT Hamming Err: M:{mat_hamming * 100:.4f}% P:{pat_hamming * 100:.4f}% Avg:{(mat_hamming + pat_hamming) / 2 * 100:.4f}%')
    #print(f'MERYL Switch Err: M:{mat_switch_error:.4f}% P:{pat_switch_error:.4f}% Avg:{(mat_switch_error + pat_switch_error) / 2:.4f}%')

def get_contig_length(walk, node_length, overlap_length, nodes=False):
    """Calculate the length of the sequence that the walk reconstructs."""
    if nodes:
        return len(walk)
    if len(walk)==0:
        return 0
    total_length = node_length[walk[0]]
    if len(walk)==1:
        return total_length
    current = walk[0]
    for next_node in walk[1:]:
        edge = (current, next_node)
        total_length += node_length[next_node] - overlap_length[edge]
        current = next_node
    return total_length

def calculate_N50(walk_lengths, chr_len=0):
    """
    Calculates N50 or NG50 of a list of walks
    :param walks: list of walks
    :return: N50
    """
    walk_lengths.sort(reverse=True)
    total_length = sum(walk_lengths)
    half_length = total_length / 2
    if chr_len!=0:
        half_length = chr_len / 2
    current_length = 0
    for length in walk_lengths:
        current_length += length
        if current_length >= half_length:
            return length
    return 0

def get_LG90(ref_fasta, asm_sta):
    # Load the reference genome and calculate the total length
    ref_records = SeqIO.parse(ref_fasta, "fasta")
    ref_length = sum(len(record.seq) for record in ref_records)

    # Load the assembly contigs and calculate their lengths
    asm_records = SeqIO.parse(asm_sta, "fasta")
    asm_lengths = sorted((len(record.seq) for record in asm_records), reverse=True)

    # Calculate the target length for LG90 (90% of the reference genome length)
    target_length = 0.9 * ref_length

    # Calculate the LG90 value (number of contigs to reach 90% of the reference genome length)
    cumulative_length = 0
    contig_count = 0
    for length in asm_lengths:
        cumulative_length += length
        contig_count += 1
        if cumulative_length >= target_length:
            break
    print(f'LG90: {contig_count}')

def walk_to_sequence(walks, graph, aux):
    edges, n2s = aux['edges_full'], aux['n2s']
    contigs = []
    for i, walk in enumerate(walks):
        prefixes = [(src, graph.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]

        res = []
        for (src, prefix) in prefixes:
            seq = str(n2s[src])
            res.append(seq[:prefix])

        contig = Seq.Seq(''.join(res) + str(n2s[walk[-1]]))  # TODO: why is this map here? Maybe I can remove it if I work with strings
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)

    return contigs

def walk_to_sequence_dgl(walks, graph, reads, edges):
    contigs = []
    for i, walk in enumerate(walks):
        prefixes = [(src, graph.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]
        sequences = [reads[src][:prefix] for (src, prefix) in prefixes]
        contig = Seq.Seq(''.join(sequences) + reads[walk[-1]])  # TODO: why is this map here? Maybe I can remove it if I work with strings
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)
    return contigs

def save_assembly(contigs, save_dir, idx, suffix=''):
    assembly_path = os.path.join(save_dir, f'{idx}_assembly{suffix}.fasta')
    SeqIO.write(contigs, assembly_path, 'fasta')

def parse_minigraph_for_full(report, save_path=None, directory=None, filename='0_minigraph.txt'):
    stat_path = report
    with open(stat_path) as f:
        report = f.read()
        print(report)
def run_minigraph(ref, asm, paf, minigraph_path=None, threads=32):
    if minigraph_path:
        cmd = f'{minigraph_path} -t{threads} -xasm -g10k -r10k --show-unmap=yes {ref} {asm}'.split(' ')
    else:
        cmd = f'minigraph -t{threads} -xasm -g10k -r10k --show-unmap=yes {ref} {asm}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def parse_pafs(idx, report, paf, paf_path=None):
    if paf_path and paf_path != 'None':
        cmd = f'k8 {paf_path} asmstat {idx} {paf}'.split()
    else:
        cmd = f'paftools.js asmstat {idx} {paf}'.split()

    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

