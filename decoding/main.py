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


# Add parent directory to path before importing utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.utils import set_seed
from training.SymGatedGCN import SymGatedGCNModel

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

def compute_scores(dgl_path, model_path, config, output_path, device='cpu'):
    """
    Compute edge scores for a graph using a trained model and save them to a file.
    
    Args:
        dgl_path: Path to the DGL graph
        model_path: Path to the trained model
        config: Configuration dictionary
        output_path: Path to save the computed scores
        device: Device to use for computation
    
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
    
    # Initialize double head model
    print("Loading double head model...")
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
        pred_dropout=train_config.get('pred_dropout', 0),
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
    
    # Create cut scores (to_cut)
    if cut_logits is not None:
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
    if 'cut_scores' in save_dict:
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

def save_walks_and_sequences(nx_graph, walks, n2s, diploid, out_path):
    if diploid:
        mat_walks, pat_walks = walks
        print(f"\nFound {len(mat_walks)} maternal and {len(pat_walks)} paternal paths")
        
        # Create single output directory
        os.makedirs(out_path, exist_ok=True)
        
        # Save maternal and paternal walks
        pickle.dump(mat_walks, open(os.path.join(out_path, 'walks_maternal.pkl'), 'wb'))
        pickle.dump(pat_walks, open(os.path.join(out_path, 'walks_paternal.pkl'), 'wb'))
        
        # Generate maternal and paternal contigs
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
        
        contigs = walk_to_sequence(walks, nx_graph, n2s)
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
    parser.add_argument('--strategy_config', type=str, default='decoding/decode_strategies.yml', help='Path to the config file')
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
    
    # Get strategy configuration
    strategy_config = strategies[args.strategy]
    log(f"Strategy '{args.strategy}' uses reduction: '{strategy_config.get('reduction')}'")
    
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
        save_dict = compute_scores(dgl_path, args.model, config, out_score_path, device='cpu')
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
                # Also assign default cut score
                if 'cut_scores' in save_dict:
                    save_dict['cut_scores'][edge] = 0.5
        
        # Apply score flipping if configured
        if strategy_config.get('flip_score', False):
            print("Flipping edge scores: 1 - score")
            edge_scores = {edge: 1 - score for edge, score in edge_scores.items()}

        # Add scores to graph
        nx.set_edge_attributes(nx_graph, edge_scores, 'score')
        
        # Add cut scores as 'to_cut'
        if 'cut_scores' in save_dict:
            cut_scores = save_dict['cut_scores']
            
            nx.set_edge_attributes(nx_graph, cut_scores, 'to_cut')
            print(f"Added 'to_cut' scores from malicious head")
    
    diploid = bool(nx.get_node_attributes(nx_graph, 'yak_region'))  # True if dict is not empty, False otherwise
    print(f"Diploid mode: {diploid}")
    
    # Using default sequence assembly method
    
    if not args.skip_decode:
        # Start inference with diploid parameter and graphic_scores
        walks = inference.get_walks(nx_graph, strategies[args.strategy], diploid, graphic_scores=graphic_scores)
 
        # Use default sequence assembly method
        reads = load_reads(reads_path)
        
        save_walks_and_sequences(nx_graph, walks, reads, diploid, args.ass_out_dir)

    if not args.skip_synthetic_eval:
        evaluate_synthetic(nx_graph, config, args.ass_out_dir, diploid)
    if not args.skip_real_eval:
        refs = get_refs(args, diploid, args.single_chrom, args.ref)
        evaluate_real(refs, config, args.ass_out_dir, diploid, threads=args.threads)

    print("Done!")

if __name__ == '__main__':
    main_entry()
