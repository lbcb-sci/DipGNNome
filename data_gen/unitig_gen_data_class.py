import os
import re
import subprocess
import gzip
from datetime import datetime
from Bio import SeqIO
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import networkx as nx
from collections import Counter, defaultdict, deque
from tqdm import tqdm
import pickle
import sys
import edlib
import numpy as np
import yaml
#import dgl
import argparse
from pyliftover import LiftOver

from data_gen_utils import create_full_dataset_dict

class DatasetCreator:
    def __init__(self, ref_path, dataset_path, data_config='dataset.yml'):
        with open(data_config) as file:
            config = yaml.safe_load(file)

        self.paths = config['paths']  # Move this line up since it's used below
        gen_config = config['gen_config']
        self.already_liftovered = False
        self.genome_str = ""
        self.genome = "hg002"
        self.gen_step_config = config['gen_steps']
        self.centromere_dict = self.load_centromere_file()
        self.load_chromosome("", "", ref_path)
        
        # Define tool paths
        self.pbsim_path = self.paths['pbsim_path']
        self.hifiasm_path = self.paths['hifiasm_path']
        self.hifiasm_dump = self.paths['hifiasm_dump']
        self.yak_path = self.paths['yak_path']

        # Configuration parameters
        self.sample_profile = self.paths['sample_profile']
        self.depth = gen_config['depth']
        self.diploid = gen_config['diploid']
        self.threads = gen_config['threads']
        self.real = gen_config['real']
        self.pbsim_model = False

        # Define root path and derived paths
        self.root_path = dataset_path
        
        # Define paths for output folders
        self.tmp_path = os.path.join(dataset_path, 'tmp')
        self.full_reads_path = os.path.join(dataset_path, "full_reads")

        self.read_descr_path = os.path.join(dataset_path, "read_descr")
        self.gfa_graphs_path = os.path.join(dataset_path, "gfa_graphs")
        self.nx_full_graphs_path = os.path.join(dataset_path, "nx_full_graphs")
        self.nx_utg_graphs_path = os.path.join(dataset_path, "nx_utg_graphs")
        self.dgl_graphs_path = os.path.join(dataset_path, "dgl_graphs")
        self.dgl_single_chrom_path = os.path.join(dataset_path, "dgl_single_chrom")
        self.pyg_graphs_path = os.path.join(dataset_path, "pyg_graphs")
        self.yak_files_path = os.path.join(dataset_path, "yak")
        self.read_to_node_path = os.path.join(dataset_path, "read_to_node")
        self.node_to_read_path = os.path.join(dataset_path, "node_to_read")
        self.utg_gfa_path = os.path.join(dataset_path, "utg_gfa")
        self.utg_node_to_node_path = os.path.join(dataset_path, "utg_node_to_node")
        # Define paths for hifiasm output
        self.hifiasm_gfa_path = os.path.join(self.root_path, 'hifiasm_gfa')
        self.hifiasm_asm_path = os.path.join(self.root_path, 'hifiasm_assemblies')
        self.reduced_reads_path = os.path.join(self.root_path, 'reduced_reads')
        self.reduced_reads_raw_path = os.path.join(self.root_path, 'reduced_reads_raw')
        self.successor_dict_path = os.path.join(self.root_path, 'successor_dict')
        self.pile_o_grams_path = os.path.join(self.root_path, 'pile_o_grams')
        self.full_reads_ec_path = os.path.join(self.root_path, 'full_reads_ec')
        self.overlaps_path = os.path.join(self.root_path, 'overlaps')
        self.gap_path = os.path.join(self.root_path, 'gap')

        # Initialize dictionaries
        self.deadends = {}
        self.gt_rescue = {}
        self.edge_info = {}

        # Create all necessary directories
        all_folders = [
            self.full_reads_path, self.gfa_graphs_path, self.nx_full_graphs_path, self.nx_utg_graphs_path, 
            self.dgl_graphs_path, self.dgl_single_chrom_path, self.pyg_graphs_path, self.read_descr_path, 
            self.tmp_path, self.yak_files_path, self.utg_gfa_path,
            self.read_to_node_path, self.node_to_read_path, self.utg_node_to_node_path,
            self.hifiasm_gfa_path, self.hifiasm_asm_path, self.reduced_reads_path, self.reduced_reads_raw_path,
            self.successor_dict_path, self.pile_o_grams_path, self.full_reads_ec_path, self.overlaps_path,
            self.gap_path
        ]
        
        for folder in all_folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Define attributes for graph construction
        self.edge_attrs = ['overlap_length', 'overlap_similarity', 'prefix_length', 'gt_bin', 'malicious', 'cross_chr', 'cross_strand', 'skip', 'gt_no_deadend', 'hifiasm_result', 'single_chrom_bad', 'cross_chr'] 
        self.node_attrs = ['read_length', 'yak_m', 'yak_p', 'support']

        self.edge_attrs_single = ['overlap_length', 'prefix_length', 'gt_bin', 'hifiasm_result', 'importance_weight'] 
        self.node_attrs_single = ['read_length', 'yak_m', 'yak_p', 'support']

    def load_chromosome(self, genome, chr_id, ref_path):
        self.genome_str = f'{genome}_{chr_id}'
        self.genome = genome
        self.chr_id = chr_id
        self.centromere_graph = '_c' in chr_id
        self.chrN = chr_id.split('_')[0]
        self.chromosomes_path = os.path.join(ref_path, 'chromosomes')
        self.centromeres_path = os.path.join(ref_path, 'centromeres')
        self.vcf_path = os.path.join(ref_path, 'vcf')
        self.chain_path = os.path.join(ref_path, 'chain')
        self.ref_path = ref_path
        #self.maternal_yak = "/mnt/sod2-project/csb4/wgs/martin/genome_references/mGorGor1_v2/mat.yak"
        #self.paternal_yak = "/mnt/sod2-project/csb4/wgs/martin/genome_references/mGorGor1_v2/pat.yak"
        
        self.maternal_yak = os.path.join(ref_path, 'mat.yak')
        self.paternal_yak = os.path.join(ref_path, 'pat.yak')



    def load_centromere_file(self):
        if 'centromere_coords' in self.paths.keys():
            centromere_coords_file = self.paths['centromere_coords']
            try:
                with open(centromere_coords_file, 'r') as file:
                    centromere_data = yaml.safe_load(file)
                return centromere_data
            except Exception as e:
                print(f"Error reading centromere coordinates file: {e}")
        exit()
    
    def _get_filetype(self, file_path):
        """Determine if file is FASTA or FASTQ format based on extension."""
        if file_path.endswith(('.gz', '')):
            base_path = file_path[:-3] if file_path.endswith('.gz') else file_path
            if base_path.endswith(('fasta', 'fna', 'fa')):
                return 'fasta'
            elif base_path.endswith(('fastq', 'fnq', 'fq')):
                return 'fastq'
        return 'fasta'  # Default to fasta if unknown
    
    def _extract_id_from_header(self, header):
        """Extract ID from real data header format.
        
        Args:
            header: Header string like 'read=4294109,m54329U_211102_230231/135006270/ccs,pos_on_original_read=0-17097'
                   or list of tuples like [('read=4294109,...', '+'), ...]
        
        Returns:
            Extracted ID like '4294109' or first ID if header is a list
        """
        if isinstance(header, list):
            # Handle unitig case where header is a list of (read_id, orientation) tuples
            if header and len(header) > 0:
                first_read_id = header[0][0]  # Get the first read ID from the first tuple
                # Apply real data parsing to the first read ID
                if self.real and first_read_id.startswith('read='):
                    return first_read_id.split(',')[0].split('=')[1]
                return first_read_id
            return header
        else:
            # Handle string case - always check for real data format when self.real is True
            if self.real:
                # Check if header has the real data format
                if header.startswith('read='):
                    return header.split(',')[0].split('=')[1]
                # For real data, we might also need to check record.description vs record.id
                # In case the header doesn't start with 'read=' but we're in real mode
                return header
            else:
                # Non-real data, return as-is
                return header

    def _parse_real_data_records(self, records, return_dict_type='id_to_description'):
        """Helper method to parse real data records and extract correct IDs.
        
        Args:
            records: Iterator of SeqRecord objects
            return_dict_type: 'id_to_description' for headers, 'id_to_sequence' for sequences
        
        Returns:
            Dictionary mapping extracted IDs to descriptions or sequences
        """
        result = {}
        for record in records:
            # Extract ID from header like 'read=4294109,m54329U_211102_230231/135006270/ccs,pos_on_original_read=0-17097'
            header = record.description if record.description else record.id
            if header.startswith('read='):
                # Extract the ID after 'read=' and before the next comma
                extracted_id = header.split(',')[0].split('=')[1]
            else:
                extracted_id = record.id
            
            if return_dict_type == 'id_to_description':
                result[extracted_id] = record.description
            elif return_dict_type == 'id_to_sequence':
                result[extracted_id] = record.seq
        return result
    
    def get_read_headers(self, reads_path):
        """Extract read headers from FASTA/FASTQ file, handling both compressed and uncompressed files.
        Returns a dict mapping read IDs to descriptions."""
        filetype = self._get_filetype(reads_path)
        if reads_path.endswith('.gz'):
            with gzip.open(reads_path, 'rt') as handle:
                records = SeqIO.parse(handle, filetype)
                if self.real:
                    return self._parse_real_data_records(records, 'id_to_description')
                else:
                    return {read.id: read.description for read in records}
        else:
            records = SeqIO.parse(reads_path, filetype)
            if self.real:
                return self._parse_real_data_records(records, 'id_to_description')
            else:
                return {read.id: read.description for read in records}
        
    def load_fasta(self, file_path, dict=False):
        """Load sequences from a FASTA/FASTQ file, handling both compressed and uncompressed files.
        Returns a list of SeqRecord objects or dict mapping ids to sequences if dict=True."""
        filetype = self._get_filetype(file_path)
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as handle:
                records = SeqIO.parse(handle, filetype)
                if dict:
                    if self.real:
                        return self._parse_real_data_records(records, 'id_to_sequence')
                    else:
                        return {record.id: record.seq for record in records}
                return list(records)
        else:
            records = SeqIO.parse(file_path, filetype)
            if dict:
                if self.real:
                    return self._parse_real_data_records(records, 'id_to_sequence')
                else:
                    return {record.id: record.seq for record in records}
            return list(records)
        
    def create_reads_fasta(self, read_seqs, chr_id):
        seq_records = []
        for read_id, sequence in read_seqs.items():
            seq_record = SeqRecord(Seq(sequence), id=str(read_id), description="")
            seq_records.append(seq_record)
        seq_record_path = os.path.join(self.reduced_reads_raw_path, f'{self.genome_str}.fasta')
        SeqIO.write(seq_records, seq_record_path, "fasta")
        
    def calculate_similarities(self, edge_ids, read_seqs, overlap_lengths):
        # Make sure that read_seqs is a dict of string, not Bio.Seq objects!
        overlap_similarities = {}
        for src, dst in tqdm(edge_ids.keys(), ncols=120):
            ol_length = overlap_lengths[(src, dst)]
            read_src = read_seqs[src]
            read_dst = read_seqs[dst]
            edit_distance = edlib.align(read_src[-ol_length:], read_dst[:ol_length])['editDistance']
            if ol_length == 0:
                overlap_similarities[(src, dst)] = 0
            else:
                overlap_similarities[(src, dst)] = 1 - edit_distance / ol_length
        return overlap_similarities
    
    def find_deadends(self, graph, read_start_dict, read_end_dict, positive=False):
        all_nodes = graph.nodes()
        graph_edges = set(graph.edges())
        gt_edges = set()
        component_start_end_nodes = []
        if positive:
            final_node = max(all_nodes, key=lambda x: read_end_dict[x])
            highest_node_reached = min(all_nodes, key=lambda x: read_end_dict[x])
        else:
            final_node = min(all_nodes, key=lambda x: read_start_dict[x])
            highest_node_reached = max(all_nodes, key=lambda x: read_start_dict[x])

        while all_nodes:
            if positive:
                start_node = min(all_nodes, key=lambda x: read_start_dict[x])
            else:
                start_node = max(all_nodes, key=lambda x: read_end_dict[x])

            # try finding a path and report the highest found node during the dfs
            current_graph = graph.subgraph(all_nodes)
            full_component = set(nx.dfs_postorder_nodes(current_graph, source=start_node))

            if positive:
                highest_node_in_component = max(full_component, key=lambda x: read_end_dict[x])
            else:
                highest_node_in_component = min(full_component, key=lambda x: read_start_dict[x])

            current_graph = graph.subgraph(full_component)
            component = set(nx.dfs_postorder_nodes(current_graph.reverse(copy=True), source=highest_node_in_component))
            current_graph = graph.subgraph(component)
            not_reached_highest = (positive and (
                    read_end_dict[highest_node_in_component] < read_end_dict[highest_node_reached])) \
                                  or (not positive and (
                    read_start_dict[highest_node_in_component] > read_start_dict[highest_node_reached]))

            if len(component) <= 2:
                all_nodes = all_nodes - full_component
                continue
            
            if not_reached_highest:
                all_nodes = all_nodes - full_component
                continue
            else:
                highest_node_reached = highest_node_in_component
            print(f"start_node: {start_node}, highest_node_in_component: {highest_node_in_component}, comp length: {len(component)}")
            component_start_end_nodes.append((start_node, highest_node_in_component))
            gt_edges = set(current_graph.edges()) | gt_edges
            if highest_node_reached == final_node:
                break
            all_nodes = all_nodes - full_component

        return gt_edges, graph_edges - gt_edges, component_start_end_nodes
    
    def is_correct_edge(self, src_start, src_end, dst_start, dst_end, positive, is_liftover=False, tolerance=0):

        if src_start == None or src_end == None or dst_start == None or dst_end == None:
            return False
        # contained:
        if (src_start <= dst_start and src_end >= dst_end) or (src_start >= dst_start and src_end <= dst_end):
            #print("contained edge")
            return False  # Contained Edges
        # overlap
        if positive:
            # For positive strand, dst_start should be between src_start and src_end
            # With tolerance for liftover cases
            if tolerance > 0:
                # Allow for some tolerance in the overlap position
                return (dst_start < src_end + tolerance and 
                        dst_start > src_start - tolerance)
            else:
                # Standard check without tolerance
                return dst_start < src_end and dst_start > src_start
        else:
            # For negative strand, src_start should be between dst_start and dst_end
            # With tolerance for liftover cases
            if tolerance > 0:
                # Allow for some tolerance in the overlap position
                return (src_start < dst_end + tolerance and 
                        src_start > dst_start - tolerance)
            else:
                # Standard check without tolerance
                return src_start < dst_end and src_start > dst_start
            #return src_end > dst_start and src_end < dst_end

    def get_correct_edges_no_hap_change(self, edges, read_start_dict, read_end_dict, read_variant_dict, positive=True):
        # Initialize sets for different edge types
        correct = set()
        cross_hap = 0

        for edge in edges:
            src, dst = edge
            
            read_start_src = read_start_dict[src]
            read_end_src = read_end_dict[src]
            read_start_dst = read_start_dict[dst]
            read_end_dst = read_end_dict[dst]
            
            # Get coordinates based on haplotype
            if read_variant_dict[dst] == read_variant_dict[src]:
                read_start_src = read_start_dict[src]
                read_end_src = read_end_dict[src]
                read_start_dst = read_start_dict[dst]
                read_end_dst = read_end_dict[dst]
            else:
                cross_hap += 1
                continue

            # Check if edge is correct
            is_correct_edge = self.is_correct_edge(read_start_src, read_end_src,
                                                   read_start_dst, read_end_dst, positive)

            if is_correct_edge:
                correct.add(edge)

        print(f"correct: {len(correct)}, incorrect: {len(edges-correct)+cross_hap}, cross_hap: {cross_hap}")
        return correct, edges-correct
    
    def create_gt(self, graph):
        
        if self.real:
            return

        read_start_dict_M = nx.get_node_attributes(graph, "read_start_M")
        read_end_dict_M = nx.get_node_attributes(graph, "read_end_M")
        read_start_dict_P = nx.get_node_attributes(graph, "read_start_P")
        read_end_dict_P = nx.get_node_attributes(graph, "read_end_P")
        hifiasm_edges_dict = nx.get_edge_attributes(graph, "hifiasm_result")
        
        # Convert hifiasm_edges dictionary to a set of edges that have value 1
        hifiasm_edges = {edge for edge, value in hifiasm_edges_dict.items() if value == 1}

        # Create a combined read_start_dict that takes the minimum of M and P values
        read_start_dict = {}
        for node in graph.nodes():
            start_M = read_start_dict_M.get(node)
            start_P = read_start_dict_P.get(node)
            
            if start_M is not None and start_P is not None:
                read_start_dict[node] = min(start_M, start_P)
            elif start_M is not None:
                read_start_dict[node] = start_M
            elif start_P is not None:
                read_start_dict[node] = start_P
            # If both are None, the node won't have an entry in read_start_dict
        
        # Similarly for read_end_dict, taking the maximum value
        read_end_dict = {}
        for node in graph.nodes():
            end_M = read_end_dict_M.get(node)
            end_P = read_end_dict_P.get(node)
            
            if end_M is not None and end_P is not None:
                read_end_dict[node] = max(end_M, end_P)
            elif end_M is not None:
                read_end_dict[node] = end_M
            elif end_P is not None:
                read_end_dict[node] = end_P
            # If both are None, the node won't have an entry in read_end_dict

        read_strand_dict = nx.get_node_attributes(graph, "read_strand")
        read_variant_dict = nx.get_node_attributes(graph, "read_variant")
        read_chr = nx.get_node_attributes(graph, 'read_chr')
        print("\nDictionary sizes:")
        print(f"read_start_dict: {len(read_start_dict):,} entries")
        print(f"read_end_dict: {len(read_end_dict):,} entries")
        print(f"read_strand_dict: {len(read_strand_dict):,} entries")
        print(f"read_variant_dict: {len(read_variant_dict):,} entries")
        print(f"read_chr: {len(read_chr):,} entries")
        print("\nRead lengths (read_end - read_start):")
        # Get connected components and analyze them
        components = list(nx.weakly_connected_components(graph))
        
        # Calculate length of each component
        component_lengths = []
        for i, component in enumerate(components):
            total_length = sum(graph.nodes[node].get('read_length', 0) for node in component)
            component_lengths.append((i, len(component), total_length))
            
        # Sort by number of nodes in descending order
        component_lengths.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nFound {len(components)} connected components:")
        print("Component analysis (sorted by node count):")
        print("Component ID | Node Count | Total Length")
        print("-" * 45)
        for comp_id, node_count, length in component_lengths:
            print(f"Comp {comp_id:6d} | {node_count:9d} | {length:12d}")

        # Add 'chr' prefix if not present and split at '_' to take first part
        read_chr = {k: f"chr{v.split('_')[0]}" if not v.split('_')[0].startswith('chr') else v.split('_')[0] for k, v in read_chr.items()}
        
        graph_edges_all, cross_chr_edges = self.check_for_cross_chromosome(graph, read_chr)
        all_chrs = list(set(read_chr.values()))
        gt_dict = {}
        self.edge_info = {edge: 0 for edge in graph.edges()}
        for edge in cross_chr_edges:
            gt_dict[edge] = 0

        # Initialize dictionaries outside the chromosome loop
        gt_bin = {edge: 1 for edge in graph.edges()}
        dead_ends = {edge: 0 for edge in graph.edges()}
        cross_chr = {edge: 0 for edge in graph.edges()}
        cross_strand = {edge: 0 for edge in graph.edges()}
        skip_wrong = {edge: 0 for edge in graph.edges()}
        forward_skip = {edge: 0 for edge in graph.edges()}
        backward_skip = {edge: 0 for edge in graph.edges()}
        unknown = {edge: 0 for edge in graph.edges()}
        mal_edges = {edge: 0 for edge in graph.edges()}
        gt_no_deadend = {edge: 0 for edge in graph.edges()}
        importance_weight = {edge: 1 for edge in graph.edges()}
        single_chrom_bad = {edge: 0 for edge in graph.edges()}

        # Initialize accumulated counters for final summary
        total_pos_forward_skips = set()
        total_pos_backward_skips = set()
        total_neg_forward_skips = set()
        total_neg_backward_skips = set()
        total_pos_unknown = set()
        total_neg_unknown = set()
        total_cross_strand_edges = set()

        for chr in all_chrs:
            # Get all graph edges of chromosome
            graph_edges = set()
            for edge in graph_edges_all:
                src, dst = edge
                if read_chr[src] == chr:
                    graph_edges.add(edge)
            # Split graph in + and - strand
            pos_edges, neg_edges, cross_strand_edges = self.split_strands(graph_edges, read_strand_dict)

            # Find nodes with read_strand == 0
            zero_strand_nodes = {node for node in graph.nodes() if read_strand_dict[node] == 0}
            # Find edges that cross through zero_strand nodes
            for node in zero_strand_nodes:
                # Add all incoming edges to that node
                for pred in graph.predecessors(node):
                    if (pred, node) in graph_edges:
                        cross_strand_edges.add((pred, node))
                        pos_edges.discard((pred, node))  # Remove from pos_edges if present
                        neg_edges.discard((pred, node))  # Remove from neg_edges if present
                # Add all outgoing edges from that node        
                for succ in graph.successors(node):
                    if (node, succ) in graph_edges:
                        cross_strand_edges.add((node, succ))
                        pos_edges.discard((node, succ))  # Remove from pos_edges if present
                        neg_edges.discard((node, succ))  # Remove from neg_edges if present
            print(f" total edges: {len(graph_edges)}, pos_edges: {len(pos_edges)}, neg_edges: {len(neg_edges)}, cross_strand_edges: {len(cross_strand_edges)}")
            # Get correct edges using new function
            #pos_correct, pos_wrong = self.get_correct_edges_no_hap_change(pos_edges, read_start_dict, read_end_dict, read_variant_dict, positive=True)
            #neg_correct, neg_wrong = self.get_correct_edges_no_hap_change(neg_edges, read_start_dict, read_end_dict, read_variant_dict, positive=False)

            pos_correct, pos_forward_skips, pos_backward_skips, pos_unknown = self.get_correct_edges_double_pos(pos_edges, read_start_dict_M, read_end_dict_M, read_start_dict_P, read_end_dict_P, read_variant_dict, positive=True)
            neg_correct, neg_forward_skips, neg_backward_skips, neg_unknown = self.get_correct_edges_double_pos(neg_edges, read_start_dict_M, read_end_dict_M, read_start_dict_P, read_end_dict_P, read_variant_dict, positive=False)

            # Combine all incorrect edges for backward compatibility
            pos_wrong = pos_forward_skips | pos_backward_skips #| pos_unknown
            neg_wrong = neg_forward_skips | neg_backward_skips #| neg_unknown

            # Accumulate counts for final summary
            total_pos_forward_skips.update(pos_forward_skips)
            total_pos_backward_skips.update(pos_backward_skips)
            total_neg_forward_skips.update(neg_forward_skips)
            total_neg_backward_skips.update(neg_backward_skips)
            total_pos_unknown.update(pos_unknown)
            total_neg_unknown.update(neg_unknown)
            total_cross_strand_edges.update(cross_strand_edges)

            # Debug output for this chromosome
            print(f"Chromosome {chr} counts: forward_skips={len(pos_forward_skips) + len(neg_forward_skips)}, backward_skips={len(pos_backward_skips) + len(neg_backward_skips)}, unknown={len(pos_unknown) + len(neg_unknown)}")
            print(f"Accumulated totals so far: forward_skips={len(total_pos_forward_skips | total_neg_forward_skips)}, backward_skips={len(total_pos_backward_skips | total_neg_backward_skips)}, unknown={len(total_pos_unknown | total_neg_unknown)}")

            pos_graph = nx.DiGraph()
            pos_graph.add_edges_from(pos_correct)   
            neg_graph = nx.DiGraph()
            neg_graph.add_edges_from(neg_correct)
            #optimal_edges_pos, deadend_edges_pos, _ = self.find_deadends(pos_graph, read_start_dict, read_end_dict, positive=True)
            #optimal_edges_neg, deadend_edges_neg, _ = self.find_deadends(neg_graph, read_start_dict, read_end_dict, positive=False)
            #print(f" deadend edges: positive: {len(deadend_edges_pos)}, negative: {len(deadend_edges_neg)}")
            #pos_correct = pos_correct - deadend_edges
            #neg_correct = neg_correct - deadend_edges_neg

            for edge in cross_chr_edges:
                cross_chr[edge] = 1
                mal_edges[edge] = 1
                gt_no_deadend[edge] = 1
                gt_bin[edge] = 0
            for edge in cross_strand_edges:
                cross_strand[edge] = 1
                mal_edges[edge] = 1
                gt_no_deadend[edge] = 1
                gt_bin[edge] = 0
                single_chrom_bad[edge] = 1
                importance_weight[edge] = 10
            for edge in pos_wrong | neg_wrong:
                skip_wrong[edge] = 1
                gt_no_deadend[edge] = 1
                gt_bin[edge] = 0
                single_chrom_bad[edge] = 1
            
            # Set specific skip type attributes
            for edge in pos_forward_skips | neg_forward_skips:
                forward_skip[edge] = 1
                #mal_edges[edge] = 1
            for edge in pos_backward_skips | neg_backward_skips:
                backward_skip[edge] = 1
                #mal_edges[edge] = 1
            for edge in pos_unknown | neg_unknown:
                unknown[edge] = 1
                single_chrom_bad[edge] = 1
                gt_bin[edge] = 0
                importance_weight[edge] = 0.1
            
            # Print information about hifiasm edges
            total_hifiasm_edges = len(hifiasm_edges)
            print()
            
            hifiasm_bad_edges = len(hifiasm_edges & (cross_strand_edges | cross_chr_edges))
            hifiasm_good_edges = total_hifiasm_edges - hifiasm_bad_edges
            print(f"\nHiFiasm Edge Analysis:")
            print(f"Total HiFiasm edges: {total_hifiasm_edges}")
            print(f"HiFiasm edges that are cross_strand or cross_chr: {hifiasm_bad_edges}")
            # Check for hifiasm edges where gt_bin is 0
            hifiasm_gt_bin_zero = len([edge for edge in hifiasm_edges if gt_bin[edge] == 0])
            print(f"HiFiasm edges with gt_bin=0: {hifiasm_gt_bin_zero}")
            print(f"Percentage of HiFiasm edges with gt_bin=0: {hifiasm_gt_bin_zero/total_hifiasm_edges*100:.2f}%" if total_hifiasm_edges > 0 else "No HiFiasm edges found")

            """for edge in hifiasm_edges:
                if edge not in cross_strand_edges | cross_chr_edges:
                    gt_bin[edge] = 1 """

            # Find trap edges after gt_bin is populated
            """trap_edges_set = self.find_trap_edges(graph, gt_bin)
            trap_edges = {edge: 0 for edge in graph.edges()}
            for edge in trap_edges_set:
                trap_edges[edge] = 1
                gt_bin[edge] = 0"""
            
            """gt_malicious = {edge: 0 for edge in graph.edges()}
            gt_malicious_r = {edge: 1 for edge in graph.edges()}
            for edge in cross_strand_edges:
                gt_malicious[edge] = 1
                gt_malicious_r[edge] = 0
            for edge in cross_chr_edges:
                gt_malicious[edge] = 1
                gt_malicious_r[edge] = 0"""
            
            #gt_wo_skip = {edge: 1 for edge in graph.edges()}

            #for edge in cross_chr_edges | cross_strand_edges | deadend_edges_pos | deadend_edges_neg:
            #    gt_wo_skip[edge] = 0

        #nx.set_edge_attributes(graph, gt_malicious, 'gt_malicious')
        #nx.set_edge_attributes(graph, gt_malicious_r, 'gt_malicious_r')

        nx.set_edge_attributes(graph, gt_bin, 'gt_bin')

        nx.set_edge_attributes(graph, dead_ends, 'dead_end')
        nx.set_edge_attributes(graph, cross_chr, 'cross_chr')
        nx.set_edge_attributes(graph, cross_strand, 'cross_strand')
        nx.set_edge_attributes(graph, skip_wrong, 'skip')
        #nx.set_edge_attributes(graph, trap_edges, 'trap_edge')
        nx.set_edge_attributes(graph, forward_skip, 'forward_skip')
        nx.set_edge_attributes(graph, backward_skip, 'backward_skip')
        nx.set_edge_attributes(graph, unknown, 'unknown')
        nx.set_edge_attributes(graph, mal_edges, 'malicious')
        nx.set_edge_attributes(graph, gt_no_deadend, 'gt_no_deadend')
        nx.set_edge_attributes(graph, importance_weight, 'importance_weight')
        nx.set_edge_attributes(graph, single_chrom_bad, 'single_chrom_bad')

        #nx.set_edge_attributes(graph, gt_wo_skip, 'gt_wo_skip')
        print(f"Total Nodes: {len(graph.nodes())}")
        print("optimal edges: ", len(graph.edges()) - len(cross_chr_edges | total_cross_strand_edges | total_pos_forward_skips | total_pos_backward_skips | total_neg_forward_skips | total_neg_backward_skips))
        print(f"Total edges: {len(graph.edges())}")
        print(f"Total cross chromosome edges: {len(cross_chr_edges)}")
        print(f"Total cross strand edges: {len(total_cross_strand_edges)}")
        #print(f"Total deadend edges: {len(deadend_edges_pos) + len(deadend_edges_neg)}")
        total_skip_edges = len(total_pos_forward_skips | total_pos_backward_skips | total_neg_forward_skips | total_neg_backward_skips)
        print(f"Total Skip edges {total_skip_edges}")
        print(f"Total Forward skip edges: {len(total_pos_forward_skips | total_neg_forward_skips)}")
        print(f"Total Backward skip edges: {len(total_pos_backward_skips | total_neg_backward_skips)}")
        print(f"Total Other incorrect edges: {len(total_pos_unknown | total_neg_unknown)}")
        
        # Verification: Ensure all incorrect edges are marked as gt_bin=0
        all_incorrect_edges = cross_chr_edges | total_cross_strand_edges | total_pos_forward_skips | total_pos_backward_skips | total_neg_forward_skips | total_neg_backward_skips | total_pos_unknown | total_neg_unknown
        incorrect_edges_with_gt_bin_1 = [edge for edge in all_incorrect_edges if gt_bin[edge] != 0]
        if incorrect_edges_with_gt_bin_1:
            print(f"WARNING: Found {len(incorrect_edges_with_gt_bin_1)} incorrect edges with gt_bin=1!")
            print("Fixing these edges to gt_bin=0...")
            for edge in incorrect_edges_with_gt_bin_1:
                gt_bin[edge] = 0
            # Update the graph attributes
            nx.set_edge_attributes(graph, gt_bin, 'gt_bin')
        else:
            print("✓ All incorrect edges properly marked as gt_bin=0")
        
        # Final verification statistics
        final_gt_bin_0 = sum(1 for val in gt_bin.values() if val == 0)
        final_gt_bin_1 = sum(1 for val in gt_bin.values() if val == 1)
        print(f"Final gt_bin distribution: {final_gt_bin_0} edges with gt_bin=0, {final_gt_bin_1} edges with gt_bin=1")
        
        # Detailed breakdown of incorrect edges
        print(f"\nDetailed breakdown of incorrect edges:")
        print(f"  Cross-chromosome edges: {len(cross_chr_edges)}")
        print(f"  Cross-strand edges: {len(total_cross_strand_edges)}")
        print(f"  Forward skip edges (positive): {len(total_pos_forward_skips)}")
        print(f"  Forward skip edges (negative): {len(total_neg_forward_skips)}")
        print(f"  Backward skip edges (positive): {len(total_pos_backward_skips)}")
        print(f"  Backward skip edges (negative): {len(total_neg_backward_skips)}")
        print(f"  Unknown edges (positive): {len(total_pos_unknown)}")
        print(f"  Unknown edges (negative): {len(total_neg_unknown)}")
        print(f"  Total incorrect edges: {len(all_incorrect_edges)}")

    def gen_yak_files(self, threads=32):
        yak_file_path = os.path.join(self.yak_files_path, f'{self.genome_str}.yak')
        reduced_fasta_path = os.path.join(self.reduced_reads_path, f'{self.genome_str}.fasta')
        cmd = f'yak triobin -t {threads} {self.paternal_yak} {self.maternal_yak} {reduced_fasta_path} > {yak_file_path}'
        print(cmd)
        subprocess.run(cmd, shell=True, cwd=self.yak_path)

    def add_trio_binning_labels(self, nx_graph):
        """
        Read a YAK trio binning file
        """
        # yak triobin output columns:
        # 1. Sequence Name: Identifier of the sequence.
        # 2. Classification: Origin of the sequence ('p' for paternal, 'm' for maternal, 'a' for ambiguous, '0' for none).
        # 3. Strong Paternal K-mers: Count of k-mers strongly associated with the paternal genome.
        # 4. Strong Maternal K-mers: Count of k-mers strongly associated with the maternal genome.
        # 5. Weak Paternal K-mers: Count of k-mers weakly associated with the paternal genome.
        # 6. Weak Maternal K-mers: Count of k-mers weakly associated with the maternal genome.
        # 7. Paternal K-mers in Stripes: Count of consecutive paternal-specific k-mers (stripes) in the sequence.
        # 8. Maternal K-mers in Stripes: Count of consecutive maternal-specific k-mers (stripes) in the sequence.
        # 9. Total K-mers: Total number of k-mers in the sequence.
        # 10. Ambiguous K-mers: Count of k-mers that cannot be confidently assigned to either parent.

        """test_labels = {edge: 1 for edge in nx_graph.edges()}
        test_labels_N = {edge: 0 for edge in nx_graph.edges()}

        nx.set_node_attributes(nx_graph, test_labels, 'yak_p')
        nx.set_node_attributes(nx_graph, test_labels_N, 'yak_m')
        nx.set_node_attributes(nx_graph, test_labels, 'yak_region')
        nx.set_node_attributes(nx_graph, test_labels, 'ambiguous')"""

        yak_file = os.path.join(self.yak_files_path, f'{self.genome_str}.yak')

        mat_label = {}
        pat_label = {}
        kmer_count_m = {}
        kmer_count_p = {}
        ambigious = {}

        # Initialize counters for 'p', 'a', 'm', '0', and 'others'
        counters = {
            'p': 0,
            'a': 0,
            'm': 0,
            '0': 0,
            'others': 0
        }
        print(yak_file)
        with open(yak_file, 'r') as yak_file:
            print(" yak file opened")
            for line in yak_file:
                entry = line.strip().split('\t')
                id = int(entry[0])
                label = entry[1]
                strong_pat_kmers = float(entry[2])/1000
                strong_mat_kmers = float(entry[3])/1000
                #print(f"strong_pat_kmers: {strong_pat_kmers}, strong_mat_kmers: {strong_mat_kmers}")

                kmer_count_p[id] = strong_pat_kmers
                kmer_count_m[id] = strong_mat_kmers

                if label == 'p':
                    pat_label[id] = 1
                    mat_label[id] = -1
                    ambigious[id] = 0
                    counters['p'] += 1
                elif label == 'm':
                    pat_label[id] = -1
                    mat_label[id] = 1
                    ambigious[id] = 0
                    counters['m'] += 1
                elif label == '0':
                    pat_label[id] = 0
                    mat_label[id] = 0
                    ambigious[id] = 0
                    counters['0'] += 1
                elif label == 'a':
                    pat_label[id] = 0
                    mat_label[id] = 0
                    ambigious[id] = 1
                    counters['a'] += 1
                else:
                    print(f"Unknown label: {label}")
                    counters['others'] += 1
                    

        # Print the counts for each label
        total_entries = sum(counters.values())
        print("\nLabel distribution:")
        for label, count in counters.items():
            percentage = (count / total_entries) * 100 if total_entries > 0 else 0
            print(f"'{label}': {count} occurrences ({percentage:.2f}%)")

        # Check the number of nodes in nx_graph and entries in pat_label
        num_nodes = nx_graph.number_of_nodes()
        num_pat_entries = len(pat_label)
        num_mat_entries = len(mat_label)
        
        print(f"Number of nodes in nx_graph: {num_nodes}")
        print(f"Number of entries in pat_label: {num_pat_entries}")
        print(f"Number of entries in mat_label: {num_mat_entries}")
        
        # Create yak_region dictionary
        yak_region = {}
        for node_id, label in pat_label.items():
            if label == 0:  # yak is '0' or 'a'
                yak_region[node_id] = 'O'
            else:  # yak is 'm' or 'p'
                yak_region[node_id] = 'E'
        
        # Set node attributes
        nx.set_node_attributes(nx_graph, pat_label, 'yak_p')
        nx.set_node_attributes(nx_graph, mat_label, 'yak_m')
        nx.set_node_attributes(nx_graph, kmer_count_p, 'kmer_count_p')
        nx.set_node_attributes(nx_graph, kmer_count_m, 'kmer_count_m')
        nx.set_node_attributes(nx_graph, yak_region, 'yak_region')
        nx.set_node_attributes(nx_graph, ambigious, 'ambigious')

    def pickle_save(self, pickle_object, path):
        # Save the graph using pickle
        file_name = os.path.join(path, f'{self.genome_str}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(pickle_object, f) 
        print(f"File saved successfully to {file_name}")

    def load_utg_graph(self):
        file_name = os.path.join(self.nx_utg_graphs_path, f'{self.genome_str}.pkl')
        with open(file_name, 'rb') as file:
            nx_graph = pickle.load(file)
        print(f"Loaded nx utg graph {self.genome_str}")
        return nx_graph
    
    def load_full_graph(self):
        file_name = os.path.join(self.nx_full_graphs_path, f'{self.genome_str}.pkl')
        with open(file_name, 'rb') as file:
            nx_graph = pickle.load(file)
        print(f"Loaded nx full graph {self.genome_str}")
        return nx_graph
    

    def save_to_dgl_and_pyg(self, nx_graph):
        """
        Save the entire graph as a single DGL file.
        For chromosome-specific splitting, use save_to_dgl_and_pyg_by_chromosome() instead.
        """
        print()
        print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
        print(f"Total edges in graph: {nx_graph.number_of_edges()}")
        
        # Ensure all expected node attributes are present for DGL conversion
        for attr in self.node_attrs:
            if len(nx.get_node_attributes(nx_graph, attr)) == 0:
                print(f"Warning: Node attribute '{attr}' not found in graph, creating empty attribute")
                # Create minimal placeholder values so attribute exists and is callable
                placeholder_values = {node: 0 for node in nx_graph.nodes()}
                nx.set_node_attributes(nx_graph, placeholder_values, attr)
        
        # Check which edge attributes are actually present and filter out missing ones
        available_edge_attrs = []
        for attr in self.edge_attrs:
            edge_attr_dict = nx.get_edge_attributes(nx_graph, attr)
            if len(edge_attr_dict) > 0:
                available_edge_attrs.append(attr)
            else:
                print(f"Warning: Edge attribute '{attr}' not found in graph, skipping it")
        
        print(f"Available edge attributes for DGL conversion: {available_edge_attrs}")
        
        # Convert to DGL using only available attributes
        graph_dgl = dgl.from_networkx(nx_graph, 
                                     node_attrs=self.node_attrs, 
                                     edge_attrs=available_edge_attrs)
        
        # Verify that attributes are accessible by name
        print("Verifying DGL graph attributes...")
        print(f"Available node data keys: {list(graph_dgl.ndata.keys())}")
        print(f"Available edge data keys: {list(graph_dgl.edata.keys())}")
        
        # Test accessibility of each expected attribute
        for attr in self.node_attrs:
            if attr in graph_dgl.ndata:
                print(f"✓ Node attribute '{attr}' is accessible")
            else:
                print(f"✗ Node attribute '{attr}' is NOT accessible")
                
        for attr in available_edge_attrs:
            if attr in graph_dgl.edata:
                print(f"✓ Edge attribute '{attr}' is accessible")
            else:
                print(f"✗ Edge attribute '{attr}' is NOT accessible")
        
        dgl.save_graphs(os.path.join(self.dgl_graphs_path, f'{self.genome_str}.dgl'), graph_dgl)
        print(f"Saved DGL graph of {self.genome_str} with:")
        print(f"- Node attributes: {self.node_attrs}")
        print(f"- Edge attributes: {available_edge_attrs}")
        print(f"All attributes should now be callable by name in the DGL graph.")
   
    def add_comp_dist(self, nx_graph):
        
        """
        Adds a node attribute 'comp_dist' representing the shortest path distance 
        between each node and its complement, and edge attributes 'ec_count' and
        'ec_count_normal' representing how often edges appear in shortest complement paths.
        """

        comp_dist = {}
        ec_count = {edge: 0 for edge in nx_graph.edges()}
        ec_count_normal = {edge: 0 for edge in nx_graph.edges()}
        
        # Initialize all nodes with -1 (indicating no path found)
        for node in nx_graph.nodes():
            comp_dist[node] = -1
        
        # Process only non-complement nodes (even numbered nodes)
        for node in tqdm(nx_graph.nodes(), desc="Computing complement distances"):
            # Skip if we already processed this node's complement
            if node % 2 == 1:  # if it's a complement node
                continue
            
            complement = node ^ 1
            try:
                # Get shortest path between node and its complement
                path = nx.shortest_path(nx_graph, node, complement)
                path_length = len(path) - 1  # Number of edges in path
                
                # Store distance for both node and its complement
                comp_dist[node] = path_length
                comp_dist[complement] = path_length
                
                # Update edge counts and normalized counts
                for i in range(len(path)-1):
                    edge = (path[i], path[i+1])
                    if edge in ec_count:
                        ec_count[edge] += 1
                        ec_count_normal[edge] += 1/path_length
                    # Also count the complement edge
                    comp_edge = (path[i+1]^1, path[i]^1)
                    if comp_edge in ec_count:
                        ec_count[comp_edge] += 1
                        ec_count_normal[comp_edge] += 1/path_length
                
            except nx.NetworkXNoPath:
                # No path exists between node and complement
                pass
        
        # Get total number of nodes for normalization
        num_nodes = nx_graph.number_of_nodes()
        
        # Normalize edge counts by number of nodes
        for edge in ec_count:
            ec_count[edge] = ec_count[edge] / num_nodes
            ec_count_normal[edge] = ec_count_normal[edge] / num_nodes
            
        # Add the attributes to the graph
        nx.set_node_attributes(nx_graph, comp_dist, 'comp_dist')
        nx.set_edge_attributes(nx_graph, ec_count, 'ec_count') 
        nx.set_edge_attributes(nx_graph, ec_count_normal, 'ec_count_normal')
        
        # Print statistics for nodes
        distances = [d for d in comp_dist.values() if d != -1]
        if distances:
            print(f"\nComplement distance statistics:")
            print(f"Average distance: {sum(distances)/len(distances):.2f}")
            print(f"Minimum distance: {min(distances)}")
            print(f"Maximum distance: {max(distances)}")
            print(f"Nodes with no path to complement: {sum(1 for x in comp_dist.values() if x == -1)}")
            print(f"Nodes with path to complement: {len(distances)}")
            
            # Print distribution of distances
            dist_counts = Counter(distances)
            print("\nDistance distribution:")
            for dist in sorted(dist_counts.keys()):
                print(f"Distance {dist}: {dist_counts[dist]} nodes")
            
            # Print statistics for edges
            ec_values = list(ec_count.values())
            ec_normal_values = list(ec_count_normal.values())
            print("\nEdge count statistics:")
            print(f"Average edge count: {sum(ec_values)/len(ec_values):.2f}")
            print(f"Maximum edge count: {max(ec_values)}")
            print(f"Average normalized edge count: {sum(ec_normal_values)/len(ec_normal_values):.2f}")
            print(f"Maximum normalized edge count: {max(ec_normal_values):.2f}")
        else:
            print("No paths between complements found in the graph")
        
    def split_strands(self, graph_edges, read_strand_dict):
        pos_edges = set()
        neg_edges = set()
        cross_strand_edges = set()
        for edge in graph_edges:
            src, dst = edge
            if read_strand_dict[src] == -1 and read_strand_dict[dst] == -1:
                neg_edges.add(edge)
            elif read_strand_dict[src] == 1 and read_strand_dict[dst] == 1:
                pos_edges.add(edge)
            else:
                cross_strand_edges.add(edge)
        return pos_edges, neg_edges, cross_strand_edges

    def check_for_cross_chromosome(self, graph, read_chr):
        graph_edges_all = graph.edges()
        cross_chr_edges = set()
        other_edges = set()

        for edge in graph_edges_all:
            src, dst = edge
            if read_chr[src] != read_chr[dst]:
                cross_chr_edges.add(edge)
            else:
                other_edges.add(edge)
        return other_edges, cross_chr_edges

    def parse_read(self, read):
        if self.real:
            id = read.id
            train_desc = ""
        else:
            description = read.description.split()
            id = description[0]
            train_desc = read.description

        seqs = (str(read.seq), str(Seq(read.seq).reverse_complement()))
        return id, seqs, train_desc 
    
    def simulate_reads(self):
        if self.diploid:
            variants = ['M', 'P']
        else:
            variants = ['M']
        out_file = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta')
        read_id = 1
        
        chr_name_split = self.chr_id.split('_')
        centromere = 'c' in chr_name_split
        multi_mode = any('multi' in part for part in chr_name_split)
        
        # Handle multi-chromosome mode
        if multi_mode:
            # Find the number after 'multi'
            for part in chr_name_split:
                if 'multi' in part:
                    num_chrs = int(chr_name_split[chr_name_split.index(part) + 1])
                    break
            if self.genome == 'mPanTro3_v2':
                available_chrs = [6, 20, 8, 21, 5, 3, 7, 2, 10, 18, 4, 1, 13, 16, 11, 12, 9, 19]
            elif self.genome == 'mGorGor1_v2':
                available_chrs = [8, 15, 11, 3, 2, 20, 14, 5, 9, 18, 6, 10, 19, 7, 21, 4, 1, 16, 12, 17, 13]
            elif self.genome == 'mPanPan1_v2':
                available_chrs = [1, 11, 20, 5, 8, 9, 6, 4, 2, 10, 12, 7, 16, 18, 21, 19, 3, 13]
            elif self.genome == 'mPonAbe1_v2':
                available_chrs = [3, 4, 5, 20, 19, 7, 10, 2, 8, 21, 6]
            elif self.genome == 'mSymSyn1_v2':
                available_chrs = [13, 5, 23, 20, 8, 12, 22, 7, 17, 16, 3, 14, 10, 2, 6, 15, 9, 24, 4, 11, 19, 1]
            elif self.genome == 'mPonPyg2_v2':
                available_chrs = [20, 3, 1, 18, 7, 2, 4, 8, 10, 19, 5, 6, 21]
            elif self.genome == 'i002c_v04':
                available_chrs = [i for i in range(1,23) if i != 14]
            elif self.genome == 'chinese_tree':
                available_chrs = [i for i in range(1,21)]
            elif self.genome == 'eucalyptus':
                available_chrs = [i for i in range(1,12)]
            else:
                available_chrs = [i for i in range(1,23)]

            sampled_chrs = np.random.choice(available_chrs, size=num_chrs, replace=False)
            print(f"Sampling chromosomes: {sampled_chrs}")
        else:
            chr_num = chr_name_split[0]
            if chr_num.startswith('chr'):
                chr_num = chr_num[3:]
            sampled_chrs = [int(chr_num)]

        # Create temporary files for each chromosome/variant combination
        temp_files = []
        temp_descr_files = []

        for chr_num in sampled_chrs:
            for var in variants:
                # Create temporary files for this iteration
                temp_fasta = os.path.join(self.tmp_path, f'temp_chr{chr_num}_{var}.fasta')
                temp_descr = os.path.join(self.tmp_path, f'temp_chr{chr_num}_{var}_descr.fasta')
                temp_files.append(temp_fasta)
                temp_descr_files.append(temp_descr)

                if centromere:
                    ref_path = os.path.join(self.centromeres_path, f'chr{chr_num}_{var}_c.fasta')
                else:
                    ref_path = os.path.join(self.chromosomes_path, f'chr{chr_num}_{var}.fasta')
                
                if self.pbsim_model:
                    subprocess.run(f'./src/pbsim --strategy wgs --method qshmm --qshmm data/QSHMM-RSII.model --depth {self.depth} --genome {ref_path}',
                                   shell=True, cwd=self.pbsim_path)
                else:
                    subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {self.depth} --genome {ref_path} --sample-profile-id {self.sample_profile}',
                                   shell=True, cwd=self.pbsim_path)
                
                reads = {r.id: r for r in SeqIO.parse(f'{self.pbsim_path}/sd_0001.fastq', 'fastq')}
                reads_list = []
                
                for align in AlignIO.parse(f'{self.pbsim_path}/sd_0001.maf', 'maf'):
                    ref, read_m = align
                    start = ref.annotations['start']
                    end = start + ref.annotations['size']
                    strand = '+' if read_m.annotations['strand'] == 1 else '-'
                    description = f'strand={strand} start={start} end={end} variant={var} chr={chr_num}'
                    reads[read_m.id].description = description
                    reads[read_m.id].id = f'{read_id}'
                    read_id += 1
                    reads_list.append(reads[read_m.id])

                # Write to temporary files
                SeqIO.write(reads_list, temp_fasta, 'fasta')
                read_descr_list = [SeqRecord(Seq(""), id=record.id, description=record.description) 
                                  for record in reads_list]
                SeqIO.write(read_descr_list, temp_descr, 'fasta')

                # Clean up PBSIM files
                subprocess.run(f'rm sd_0001.fastq sd_0001.maf sd_0001.ref', shell=True, cwd=self.pbsim_path)

        try:
            # Merge all temporary FASTA files
            subprocess.run(f'cat {" ".join(temp_files)} > {out_file}', shell=True, check=True)
            
            # Compress the merged file
            subprocess.run(f'gzip -f {out_file}', shell=True, check=True)
                        
            # Verify the compressed file
            with gzip.open(f"{out_file}.gz", 'rt') as verify_file:
                verify_count = sum(1 for _ in SeqIO.parse(verify_file, 'fasta'))
                expected_count = read_id - 1
                if verify_count != expected_count:
                    raise RuntimeError(f"File verification failed: expected {expected_count} records but read {verify_count}")
                
        except Exception as e:
            raise RuntimeError(f"Error processing files: {str(e)}")
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files + temp_descr_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def parse_sequence_header(self, header):
        match = re.match(r"read=\d+,(\d+),pos_on_original_read=(\d+)-(\d+)", header)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None

    def parse_description_header(self, header):
        match = re.match(r"(\d+) strand=(.) start=(\d+) end=(\d+) variant=(\w+) chr=(.+)", header)
        if match:
            return {
                "id": int(match.group(1)),
                "strand": match.group(2),
                "start": int(match.group(3)),
                "end": int(match.group(4)),
                "variant": match.group(5),
                "chr": match.group(6)
            }
        print(f"Failed to parse header: {header}", file=sys.stderr)
        return None

    def create_graphs(self):

        if self.real:
            full_fasta = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            full_fasta = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')

        gfa_output = os.path.join(self.gfa_graphs_path, f'{self.genome_str}.gfa')
        

        if self.diploid:
            hifiasm_asm_output_1 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap1.gfa')
            hifiasm_asm_output_2 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap2.gfa')
            print(f'./hifiasm --prt-raw -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {full_fasta}')
            #exit()
            subprocess.run(f'./hifiasm --prt-raw -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {full_fasta}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.hap1.p_ctg.gfa {hifiasm_asm_output_1}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.hap2.p_ctg.gfa {hifiasm_asm_output_2}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.raw.r_utg.gfa {gfa_output}', shell=True, cwd=self.hifiasm_path)

        else:
            hifiasm_asm_output = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.gfa')
            subprocess.run(f'./hifiasm --prt-raw -r3 -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {full_fasta}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.p_ctg.noseq.gfa {hifiasm_asm_output}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.p_ctg.gfa {hifiasm_asm_output}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.raw.r_utg.gfa {gfa_output}', shell=True, cwd=self.hifiasm_path)
        
        self.save_hifiasm_assemblies()

        subprocess.run(f'rm {self.hifiasm_dump}/tmp_asm*', shell=True, cwd=self.hifiasm_path)

    def save_hifiasm_assemblies(self):
        hic_mode = False
        if  self.diploid:
            self._convert_and_move_assembly('tmp_asm.bp.hap1.p_ctg.gfa')
            self._convert_and_move_assembly('tmp_asm.bp.hap2.p_ctg.gfa')
        elif hic_mode:
            self._convert_and_move_assembly('tmp_asm.hic.hap1.p_ctg.gfa')
            self._convert_and_move_assembly('tmp_asm.hic.hap2.p_ctg.gfa')
        else:
            self._convert_and_move_assembly('tmp_asm.bp.p_ctg.gfa')

    def _convert_and_move_assembly(self, gfa_filename):
        gfa_path = os.path.join(self.hifiasm_dump, gfa_filename)
        fa_filename = gfa_filename.replace('.gfa', '.fa')
        fa_path = os.path.join(self.hifiasm_asm_path, fa_filename)
        
        if os.path.exists(gfa_path):
            # Convert GFA to FASTA
            awk_command = "awk '/^S/{print \">\"$2;print $3}'"
            subprocess.run(f"{awk_command} {gfa_path} > {fa_path}", shell=True, check=True)
            print(f"Converted and moved {gfa_filename} to {fa_filename}")
        else:
            print(f"Warning: Expected assembly file {gfa_filename} not found.")
    

    #new_pos = liftover_object.convert_coordinate(chr, position)

        
    def parse_gfa(self):
        nx_graph, read_seqs, node_to_read, read_to_node, successor_dict = self.only_from_gfa()
        self.create_reads_fasta(read_seqs, self.chr_id)  # Add self.chr_id as an argument
        # Save data
        self.pickle_save(node_to_read, self.node_to_read_path)
        self.pickle_save(successor_dict, self.successor_dict_path)
        #self.pickle_save(read_seqs, self.reduced_reads_path)
        self.pickle_save(read_to_node, self.read_to_node_path)

        return nx_graph  

    def add_hifiasm_final_edges(self, gfa_path):
            print(f"Loading HiFiasms final gfa from {gfa_path}...")
            
            if not os.path.exists(gfa_path):
                print(f"ERROR: GFA file not found at {gfa_path}")
                return [], [], []
                
            try:
                with open(gfa_path) as f:
                    rows = f.readlines()
                    c2r = defaultdict(list)
                    for row in rows:
                        row = row.strip().split()
                        if row[0] != "A": continue
                        c2r[row[1]].append(row)

                print(f"Found {len(c2r)} contigs in GFA file")
                print(f"Generating contigs...")
                edges = []
                prefixes = []
                orientations = []
                for c_id, reads in c2r.items():
                    reads = sorted(reads, key=lambda x:int(x[2]))
                    for i in range(len(reads)-1):
                        curr_row, next_row = reads[i], reads[i+1]
                        curr_prefix = int(next_row[2])-int(curr_row[2])
                        edges.append((str(curr_row[4]), str(next_row[4])))
                        orientations.append((0 if curr_row[3] == '+' else 1, 0 if next_row[3] == '+' else 1))
                        #print(curr_row[4])
                        prefixes.append(curr_prefix)
                print(f"Generated {len(edges)} edges from contigs")
                return edges, prefixes, orientations
            except Exception as e:
                print(f"ERROR: Failed to parse GFA file: {str(e)}")
                return [], [], []

    def only_from_gfa(self, rescue_hifiasm_final_edges=True, use_liftover=True):
        training = not self.real
        gfa_path = os.path.join(self.gfa_graphs_path, f'{self.genome_str}.gfa')
        #reads_path = os.path.join(self.full_reads_ec_path, f'{self.genome_str}_ec.fa')
        if self.real:
            reads_path = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            reads_path = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')

        # OPTIMIZATION: Load both headers and sequences in a single pass
        print(f"Loading reads data (headers and sequences) in single pass...")
        read_headers, fastaq_seqs = self._load_reads_data_optimized(reads_path)
        print(f"Loaded {len(read_headers)} read headers and {len(fastaq_seqs)} sequences")

        graph_nx = nx.DiGraph()
        read_to_node, node_to_read, old_read_to_utg = {}, {}, {}  ##########
        read_to_node2 = {}
        edges_dict = {}
        node_support = {}
        read_lengths, read_seqs = {}, {}  # Obtained from the GFA
        read_idxs, read_strands, read_starts_M, read_ends_M, read_starts_P, read_ends_P, read_chrs, read_variants, variant_class = {}, {}, {}, {}, {}, {}, {}, {}, {}  # Obtained from the FASTA/Q headers
        edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}
        no_seqs_flag = False

        # Initialize liftover objects if use_liftover is True
        liftover_objects = {}
        # Initialize default suffix values to avoid UnboundLocalError when processing real data
        m_suffix = 'M'
        p_suffix = 'P'
        
        if use_liftover and self.diploid and not self.real:
            if 'multi' in self.chrN:
                # Multi-chromosome case - create liftover objects for each chromosome
                unique_chrs = set()
                for i in range(1, 24):  # chromosomes 1-23
                    chr_id = f'chr{i}'
                    unique_chrs.add(chr_id)
                
                for chr_id in unique_chrs:
                    if self.genome == 'hg002_v101':
                        chain_file = os.path.join(self.ref_path, 'v1.0.bothdirs.chain')
                        liftover_objects[chr_id] = {
                            'p_to_m': LiftOver(chain_file),
                            'm_to_p': LiftOver(chain_file)
                        }
                        m_suffix = 'MATERNAL'
                        p_suffix = 'PATERNAL'
                    else:
                        p_to_m_chain = os.path.join(self.chain_path, f'{chr_id}_P_to_M.chain')
                        m_to_p_chain = os.path.join(self.chain_path, f'{chr_id}_M_to_P.chain')
                        
                        # Check if chain files exist before creating LiftOver objects
                        if os.path.exists(p_to_m_chain) and os.path.exists(m_to_p_chain):
                            liftover_objects[chr_id] = {
                                'p_to_m': LiftOver(p_to_m_chain),
                                'm_to_p': LiftOver(m_to_p_chain)
                            }
                        else:
                            liftover_objects[chr_id] = {
                                'p_to_m': None,
                                'm_to_p': None
                            }
                        m_suffix = 'M'
                        p_suffix = 'P'
            else:
                # Single chromosome case
                if self.genome == 'hg002_v101':
                    chain_file = os.path.join(self.ref_path, 'v1.0.bothdirs.chain')
                    if os.path.exists(chain_file):
                        p_to_m_lo = LiftOver(chain_file)
                        m_to_p_lo = LiftOver(chain_file)
                        # Store single chromosome liftover objects in the same format for consistency
                        liftover_objects = {'p_to_m': p_to_m_lo, 'm_to_p': m_to_p_lo}
                    else:
                        print("HG002 liftover files not found")
                        exit()
                    m_suffix = 'MATERNAL'
                    p_suffix = 'PATERNAL'
                else:
                    p_to_m_chain = os.path.join(self.chain_path, f'{self.chrN}_P_to_M.chain')
                    m_to_p_chain = os.path.join(self.chain_path, f'{self.chrN}_M_to_P.chain')
                    
                    # Check if chain files exist before creating LiftOver objects
                    if os.path.exists(p_to_m_chain) and os.path.exists(m_to_p_chain):
                        p_to_m_lo = LiftOver(p_to_m_chain)
                        m_to_p_lo = LiftOver(m_to_p_chain)
                        # Store single chromosome liftover objects in the same format for consistency
                    else:
                        print(f"Warning: Chain files {p_to_m_chain} or {m_to_p_chain} not found, skipping liftover for {self.chrN}")
                        p_to_m_lo = None
                        m_to_p_lo = None
                    liftover_objects = {'p_to_m': p_to_m_lo, 'm_to_p': m_to_p_lo}

                    m_suffix = 'M'
                    p_suffix = 'P'
                
                # Store single chromosome liftover objects in the same format for consistency
                liftover_objects = {'p_to_m': p_to_m_lo, 'm_to_p': m_to_p_lo}
        time_start = datetime.now()
        print(f'Starting to loop over GFA')

        # OPTIMIZATION: Process GFA file streaming instead of loading all lines into memory
        node_idx = 0
        edge_idx = 0
        
        # Store lines that need to be processed together (A lines after S lines)
        pending_a_lines = []
        current_s_id = None
        
        for line in self._process_gfa_streaming(gfa_path):
            if line[0] == 'A':
                old_read_to_utg[line[4]] = line[1]
                pending_a_lines.append(line)
                continue

            if line[0] == 'S':
                # Process any pending A lines from the previous S line
                if current_s_id is not None and pending_a_lines:
                    self._process_pending_a_lines(pending_a_lines, current_s_id, node_idx - 2, 
                                                 read_to_node2, node_to_read, read_headers, 
                                                 training, use_liftover, liftover_objects, 
                                                 m_suffix, p_suffix, read_strands, read_starts_M, 
                                                 read_ends_M, read_starts_P, read_ends_P, 
                                                 read_variants, read_chrs)
                
                # Reset for new S line
                pending_a_lines = []
                
                if len(line) == 6: 
                    tag, id, sequence, length, count, support = line
                if len(line) == 5:
                    tag, id, sequence, length, support = line 
                if len(line) == 4:
                    tag, id, sequence, length = line
                if sequence == '*':
                    no_seqs_flag = True
                    sequence = '*' * int(length[5:])
                sequence = Seq(sequence)  # This sequence is already trimmed in raven!
                length = int(length[5:])

                real_idx = node_idx
                virt_idx = node_idx + 1
                read_to_node[id] = (real_idx, virt_idx)
                node_to_read[real_idx] = id
                node_to_read[virt_idx] = id

                graph_nx.add_node(real_idx)  # real node = original sequence
                graph_nx.add_node(virt_idx)  # virtual node = rev-comp sequence

                read_seqs[real_idx] = str(sequence)
                read_seqs[virt_idx] = str(sequence.reverse_complement())

                read_lengths[real_idx] = length
                read_lengths[virt_idx] = length

                support = int(line[4].split(':')[-1])
                node_support[real_idx] = support/self.depth
                node_support[virt_idx] = support/self.depth
                
                current_s_id = id
                node_idx += 2

            elif line[0] == 'L':
                # Process any remaining A lines before processing L lines
                if current_s_id is not None and pending_a_lines:
                    self._process_pending_a_lines(pending_a_lines, current_s_id, node_idx - 2, 
                                                 read_to_node2, node_to_read, read_headers, 
                                                 training, use_liftover, liftover_objects, 
                                                 m_suffix, p_suffix, read_strands, read_starts_M, 
                                                 read_ends_M, read_starts_P, read_ends_P, 
                                                 read_variants, read_chrs)
                    pending_a_lines = []
                    current_s_id = None

                if len(line) == 6:
                    # raven, normal GFA 1 standard
                    tag, id1, orient1, id2, orient2, cigar = line
                elif len(line) == 7:
                    # hifiasm GFA
                    tag, id1, orient1, id2, orient2, cigar, _ = line
                    patterns = self._compile_regex_patterns()
                    id1_match = patterns['hifiasm_id'].search(id1)
                    id2_match = patterns['hifiasm_id'].search(id2)
                    id1 = id1_match.group(1) if id1_match else id1
                    id2 = id2_match.group(1) if id2_match else id2
                elif len(line) == 8:
                    # hifiasm GFA newer
                    tag, id1, orient1, id2, orient2, cigar, _, _ = line
                else:
                    raise Exception("Unknown GFA format!")

                if orient1 == '+' and orient2 == '+':
                    src_real = read_to_node[id1][0]
                    dst_real = read_to_node[id2][0]
                    src_virt = read_to_node[id2][1]
                    dst_virt = read_to_node[id1][1]
                if orient1 == '+' and orient2 == '-':
                    src_real = read_to_node[id1][0]
                    dst_real = read_to_node[id2][1]
                    src_virt = read_to_node[id2][0]
                    dst_virt = read_to_node[id1][1]
                if orient1 == '-' and orient2 == '+':
                    src_real = read_to_node[id1][1]
                    dst_real = read_to_node[id2][0]
                    src_virt = read_to_node[id2][1]
                    dst_virt = read_to_node[id1][0]
                if orient1 == '-' and orient2 == '-':
                    src_real = read_to_node[id1][1]
                    dst_real = read_to_node[id2][1]
                    src_virt = read_to_node[id2][0]
                    dst_virt = read_to_node[id1][0]

                graph_nx.add_edge(src_real, dst_real)
                graph_nx.add_edge(src_virt,
                                dst_virt)  # In hifiasm GFA this might be redundant, but it is necessary for raven GFA

                edge_ids[(src_real, dst_real)] = edge_idx
                edge_ids[(src_virt, dst_virt)] = edge_idx + 1
                edge_idx += 2

                # -----------------------------------------------------------------------------------
                # This enforces similarity between the edge and its "virtual pair"
                # Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
                # When parsing CSV that was not necessarily so:
                # Sometimes reads would be slightly differently aligned from their RC pairs
                # Thus resulting in different overlap lengths
                # -----------------------------------------------------------------------------------

                try:
                    ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
                except ValueError:
                    print('Cannot convert CIGAR string into overlap length!')
                    raise ValueError

                overlap_lengths[(src_real, dst_real)] = ol_length
                overlap_lengths[(src_virt, dst_virt)] = ol_length

                prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
                prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length

        # Process any remaining A lines after the last S line
        if current_s_id is not None and pending_a_lines:
            self._process_pending_a_lines(pending_a_lines, current_s_id, node_idx - 2, 
                                         read_to_node2, node_to_read, read_headers, 
                                         training, use_liftover, liftover_objects, 
                                         m_suffix, p_suffix, read_strands, read_starts_M, 
                                         read_ends_M, read_starts_P, read_ends_P, 
                                         read_variants, read_chrs)
        
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')


        ### The following code is for adding the final assembly edges to the graph.
        ## In HiFiasm the final assembly has some edges that are NOT in the full graph.
        ## so we need to add them to make the graph complete.
        if rescue_hifiasm_final_edges:
            if self.diploid:
                hifiasm_asm_output_1 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap1.gfa')
                hifiasm_asm_output_2 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap2.gfa')
                final_assembly_edges_1, prefixes_1, orientations_1 = self.add_hifiasm_final_edges(hifiasm_asm_output_1)
                final_assembly_edges_2, prefixes_2, orientations_2 = self.add_hifiasm_final_edges(hifiasm_asm_output_2)
                final_assembly_edges = final_assembly_edges_1 + final_assembly_edges_2
                prefixes = prefixes_1 + prefixes_2
                orientations = orientations_1 + orientations_2
            else:
                hifiasm_asm_output = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.gfa')
                final_assembly_edges, prefixes, orientations = self.add_hifiasm_final_edges(hifiasm_asm_output)

            nx_edges = set(graph_nx.edges())
            nx_nodes = set(graph_nx.nodes())
            just_added_edges = set()
            new_nodes_to_read = {}
            new_edge = {edge: 0 for edge in graph_nx.edges()}
            new_node = {node: 0 for node in graph_nx.nodes()}
            # Initialize hifiasm_result attribute for all existing edges
            hifiasm_result = {edge: 0 for edge in graph_nx.edges()}
            
            # OPTIMIZATION: fastaq_seqs is already loaded, no need to load again
            print(f"Using pre-loaded sequences with {len(fastaq_seqs)} entries")

            # Initialize counters
            print(f"Processing {len(final_assembly_edges)} edges")
            total_edges = len(final_assembly_edges)
            skipped_missing_read_ids = 0
            skipped_missing_nodes = 0 
            skipped_self_loops = 0
            added_forward_edges = 0
            existing_forward_edges = 0
            double_new_edges = 0
            for i, edge in enumerate(final_assembly_edges):
                # Convert read IDs to node IDs using read_to_node
                ori = orientations[i]

                if edge[0] not in read_to_node2:
                    # Add node and sequence for the first read
                    #continue
                    # Check if read ID exists in fastaq_seqs before proceeding
                    extracted_id = self._extract_id_from_header(edge[0])
                    if extracted_id not in fastaq_seqs:
                        print(f"Warning: Read ID '{extracted_id}' not found in fastaq_seqs, skipping edge")
                        skipped_missing_read_ids += 1
                        continue
                    
                    # Check if read headers exist (for non-real data)
                    if not self.real and extracted_id not in read_headers:
                        print(f"Warning: Read ID '{extracted_id}' not found in read_headers, skipping edge")
                        skipped_missing_read_ids += 1
                        continue
                    
                    if ori[0] == 0:  # 0 means '+'
                        real_idx, virt_idx = node_idx, node_idx + 1 
                    else:
                        real_idx, virt_idx = node_idx + 1, node_idx
                    read_to_node2[edge[0]] = (real_idx, virt_idx)
                    for idx in (real_idx, virt_idx):
                        nx_nodes.add(idx)
                        graph_nx.add_node(idx)
                        node_to_read[idx] = edge[0]
                        new_nodes_to_read[idx] = edge[0]
                        new_node[idx] = 1
                        read_lengths[idx] = len(fastaq_seqs[extracted_id])
                        # Set default support value for new nodes
                        node_support[idx] = 1

                    node_idx += 2
                    if not self.real:
                        #### you have given here the read_ids (small read_ids) 
                        description = read_headers[extracted_id]
                        # OPTIMIZATION: Use optimized description parsing
                        desc_info = self._extract_description_info(description)
                        strand = 1 if desc_info['strand'] == '+' else -1
                        start = desc_info['start']
                        end = desc_info['end']
                        chromosome = desc_info['chromosome']
                        
                        # this should be the ones in the description header info!! So I am very close
                        # just use the descr header info and fill the dicts here???
                        if self.diploid:
                            variant = desc_info['variant']
                            read_variants[real_idx] = read_variants[virt_idx] = variant
                            # Set haplotype-specific coordinates based on variant
                            if variant == 'M':
                                # Define maternal coordinates
                                start_M = start
                                end_M = end
                                read_starts_M[real_idx] = read_starts_M[virt_idx] = start_M
                                read_ends_M[real_idx] = read_ends_M[virt_idx] = end_M
                                # Liftover M coordinates to P coordinates
                                if use_liftover:
                                    chr_id = f"chr{chromosome}"
                                    m_chr_name = f'{chr_id}_{m_suffix}'
                                    
                                    # Choose appropriate liftover object based on multi-chromosome mode
                                    if 'multi' in self.chrN:
                                        if chr_id in liftover_objects:
                                            m_to_p_lo = liftover_objects[chr_id]['m_to_p']
                                            start_M_to_P = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, start_M)
                                            end_M_to_P = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, end_M)
                                        else:
                                            start_M_to_P = end_M_to_P = -1
                                    else:
                                        start_M_to_P = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, start_M)
                                        end_M_to_P = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, end_M)
                                    
                                    read_starts_P[real_idx] = read_starts_P[virt_idx] = start_M_to_P if start_M_to_P != -1 else None
                                    read_ends_P[real_idx] = read_ends_P[virt_idx] = end_M_to_P if end_M_to_P != -1 else None
                                else:
                                    read_starts_P[real_idx] = read_starts_P[virt_idx] = None
                                    read_ends_P[real_idx] = read_ends_P[virt_idx] = None
                            else:  # variant == 'P'
                                # Define paternal coordinates
                                start_P = start
                                end_P = end
                                read_starts_P[real_idx] = read_starts_P[virt_idx] = start_P
                                read_ends_P[real_idx] = read_ends_P[virt_idx] = end_P
                                # Liftover P coordinates to M coordinates
                                if use_liftover:
                                    chr_id = f"chr{chromosome}"
                                    p_chr_name = f'{chr_id}_{p_suffix}'
                                    
                                    # Choose appropriate liftover object based on multi-chromosome mode
                                    if 'multi' in self.chrN:
                                        if chr_id in liftover_objects:
                                            p_to_m_lo = liftover_objects[chr_id]['p_to_m']
                                            start_P_to_M = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, start_P)
                                            end_P_to_M = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, end_P)
                                        else:
                                            start_P_to_M = end_P_to_M = -1
                                    else:
                                        start_P_to_M = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, start_P)
                                        end_P_to_M = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, end_P)
                                    
                                    read_starts_M[real_idx] = read_starts_M[virt_idx] = start_P_to_M if start_P_to_M != -1 else None
                                    read_ends_M[real_idx] = read_ends_M[virt_idx] = end_P_to_M if end_P_to_M != -1 else None
                                else:
                                    read_starts_M[real_idx] = read_starts_M[virt_idx] = None
                                    read_ends_M[real_idx] = read_ends_M[virt_idx] = None
                        else:
                            # For non-diploid case, use the old variable names for backward compatibility
                            read_starts_M[real_idx] = read_starts_M[virt_idx] = start
                            read_ends_M[real_idx] = read_ends_M[virt_idx] = end
                        read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                        read_chrs[real_idx] = read_chrs[virt_idx] = chromosome
                        #read_lengths[real_idx] = read_lengths[virt_idx] = abs(end-start)
                        #skipped_missing_read_ids += 1
                        #continue

                if edge[1] not in read_to_node2:
                    # Add node and sequence for the second read
                    #continue
                    # Check if read ID exists in fastaq_seqs before proceeding
                    extracted_id = self._extract_id_from_header(edge[1])
                    if extracted_id not in fastaq_seqs:
                        print(f"Warning: Read ID '{extracted_id}' not found in fastaq_seqs, skipping edge")
                        skipped_missing_read_ids += 1
                        continue
                    
                    # Check if read headers exist (for non-real data)
                    if not self.real and extracted_id not in read_headers:
                        print(f"Warning: Read ID '{extracted_id}' not found in read_headers, skipping edge")
                        skipped_missing_read_ids += 1
                        continue

                    if ori[1] == 0:  # 0 means '+'
                        real_idx, virt_idx = node_idx, node_idx + 1 
                    else:
                        real_idx, virt_idx = node_idx + 1, node_idx

                    read_to_node2[edge[1]] = (real_idx, virt_idx)
                    for idx in (real_idx, virt_idx):
                        nx_nodes.add(idx)
                        graph_nx.add_node(idx)
                        node_to_read[idx] = edge[1]
                        new_nodes_to_read[idx] = edge[1]
                        new_node[idx] = 1
                        read_lengths[idx] = len(fastaq_seqs[extracted_id])
                        # Set default support value for new nodes
                        node_support[idx] = 1
                        # Set default yak attributes for new nodes

                    node_idx += 2
                    if not self.real:
                        #### you have given here the read_ids (small read_ids) 
                        description = read_headers[extracted_id]
                        # OPTIMIZATION: Use optimized description parsing
                        desc_info = self._extract_description_info(description)
                        strand = 1 if desc_info['strand'] == '+' else -1
                        start = desc_info['start']
                        end = desc_info['end']
                        chromosome = desc_info['chromosome']
                        
                        # this should be the ones in the description header info!! So I am very close
                        # just use the descr header info and fill the dicts here???
                        if self.diploid:
                            variant = desc_info['variant']
                            read_variants[real_idx] = read_variants[virt_idx] = variant
                            # Set haplotype-specific coordinates based on variant
                            if variant == 'M':
                                # Define maternal coordinates
                                start_M = start
                                end_M = end
                                read_starts_M[real_idx] = read_starts_M[virt_idx] = start_M
                                read_ends_M[real_idx] = read_ends_M[virt_idx] = end_M
                                # Liftover M coordinates to P coordinates
                                if use_liftover:
                                    chr_id = f"chr{chromosome}"
                                    m_chr_name = f'{chr_id}_{m_suffix}'
                                    
                                    # Choose appropriate liftover object based on multi-chromosome mode
                                    if 'multi' in self.chrN:
                                        if chr_id in liftover_objects:
                                            m_to_p_lo = liftover_objects[chr_id]['m_to_p']
                                            start_M_to_P = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, start_M)
                                            end_M_to_P = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, end_M)
                                        else:
                                            start_M_to_P = end_M_to_P = -1
                                    else:
                                        start_M_to_P = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, start_M)
                                        end_M_to_P = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, end_M)
                                    
                                    read_starts_P[real_idx] = read_starts_P[virt_idx] = start_M_to_P if start_M_to_P != -1 else None
                                    read_ends_P[real_idx] = read_ends_P[virt_idx] = end_M_to_P if end_M_to_P != -1 else None
                                else:
                                    read_starts_P[real_idx] = read_starts_P[virt_idx] = None
                                    read_ends_P[real_idx] = read_ends_P[virt_idx] = None
                            else:  # variant == 'P'
                                # Define paternal coordinates
                                start_P = start
                                end_P = end
                                read_starts_P[real_idx] = read_starts_P[virt_idx] = start_P
                                read_ends_P[real_idx] = read_ends_P[virt_idx] = end_P
                                # Liftover P coordinates to M coordinates
                                if use_liftover:
                                    chr_id = f"chr{chromosome}"
                                    p_chr_name = f'{chr_id}_{p_suffix}'
                                    
                                    # Choose appropriate liftover object based on multi-chromosome mode
                                    if 'multi' in self.chrN:
                                        if chr_id in liftover_objects:
                                            p_to_m_lo = liftover_objects[chr_id]['p_to_m']
                                            start_P_to_M = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, start_P)
                                            end_P_to_M = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, end_P)
                                        else:
                                            start_P_to_M = end_P_to_M = -1
                                    else:
                                        start_P_to_M = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, start_P)
                                        end_P_to_M = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, end_P)
                                    
                                    read_starts_M[real_idx] = read_starts_M[virt_idx] = start_P_to_M if start_P_to_M != -1 else None
                                    read_ends_M[real_idx] = read_ends_M[virt_idx] = end_P_to_M if end_P_to_M != -1 else None
                                else:
                                    read_starts_M[real_idx] = read_starts_M[virt_idx] = None
                                    read_ends_M[real_idx] = read_ends_M[virt_idx] = None
                        else:
                            # For non-diploid case, use the old variable names for backward compatibility
                            read_starts_M[real_idx] = read_starts_M[virt_idx] = start
                            read_ends_M[real_idx] = read_ends_M[virt_idx] = end
                        read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                        read_chrs[real_idx] = read_chrs[virt_idx] = chromosome
                        #read_lengths[real_idx] = read_lengths[virt_idx] = abs(end-start)
                        #skipped_missing_read_ids += 1
                        #continue

                # Get forward node IDs
                src_node = read_to_node2[edge[0]][ori[0]]  # Forward or Backward node ID
                dst_node = read_to_node2[edge[1]][ori[1]]  # Forward or Backward node ID
                    
                if src_node == dst_node:
                    skipped_self_loops += 1
                    continue

                if (src_node, dst_node) in just_added_edges:
                    double_new_edges += 1
                    continue
                    
                # Add forward edge (e1,e2)
                if (src_node, dst_node) not in nx_edges:
                    #print(f"Adding forward edge ({src_node}, {dst_node})")
                    graph_nx.add_edge(src_node, dst_node)
                    new_edge[(src_node, dst_node)] = 1
                    just_added_edges.add((src_node, dst_node))
                    prefix_lengths[(src_node, dst_node)] = prefixes[i]
                    ol_length = read_lengths[src_node] - prefixes[i]

                    overlap_lengths[(src_node, dst_node)] = ol_length
                    # Mark this edge as part of HiFiasm's final assembly result
                    hifiasm_result[(src_node, dst_node)] = 1
                    added_forward_edges += 1
                    edge_ids[(src_node, dst_node)] = edge_idx
                    edge_idx += 1
                else:
                    # Edge already exists, mark it as part of HiFiasm's final assembly result
                    hifiasm_result[(src_node, dst_node)] = 1
                    existing_forward_edges += 1
                
                # Add reverse complement edge (e2^1, e1^1)
                rc_src = dst_node ^ 1  # e2^1
                rc_dst = src_node ^ 1  # e1^1
                
                if (rc_src, rc_dst) not in nx_edges:
                    #print(f"Adding RC edge ({rc_src}, {rc_dst})")
                    graph_nx.add_edge(rc_src, rc_dst)
                    new_edge[(rc_src, rc_dst)] = 1
                    just_added_edges.add((rc_src, rc_dst))
                    prefix_lengths[(rc_src, rc_dst)] = prefixes[i]
                    ol_length = read_lengths[rc_src] - prefixes[i]
                    overlap_lengths[(rc_src, rc_dst)] = ol_length
                    # Mark this edge as part of HiFiasm's final assembly result
                    hifiasm_result[(rc_src, rc_dst)] = 1
                    edge_ids[(rc_src, rc_dst)] = edge_idx
                    edge_idx += 1
                else:
                    # Edge already exists, mark it as part of HiFiasm's final assembly result
                    hifiasm_result[(rc_src, rc_dst)] = 1

                hifiasm_result[(src_node, dst_node)] = 1
                

            print(f"\nEdge Processing Summary:")
            print(f"Total edges checked: {total_edges}")
            print(f"Skipped edges:")
            print(f"  - Missing read IDs: {skipped_missing_read_ids}")
            print(f"  - Self loops: {skipped_self_loops}")
            print(f"Forward edges:")
            print(f"  - Added: {added_forward_edges}")
            print(f"  - Already existed: {existing_forward_edges}")
            print(f"  - Double new edges: {double_new_edges}")
            elapsed = (datetime.now() - time_start).seconds
            print(f'Elapsed time: {elapsed}s')
        
            # OPTIMIZATION: Only load sequences if no_seqs_flag is True or new nodes need sequences
            if no_seqs_flag:
                print(f'Loading sequences for all nodes...')
                for node_id in tqdm(node_to_read.keys(), ncols=120):
                    read_id = node_to_read[node_id]
                    extracted_id = self._extract_id_from_header(read_id)
                    seq = fastaq_seqs[extracted_id]
                    read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
                print(f'Loaded DNA sequences!')
            else:
                print(f'Loading sequences for new nodes...')
                for node_id in tqdm(new_nodes_to_read.keys(), ncols=120):
                    read_id = new_nodes_to_read[node_id]
                    extracted_id = self._extract_id_from_header(read_id)
                    seq = fastaq_seqs[extracted_id]
                    read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
                print(f'Added new DNA sequences!')

        print(f'Calculating similarities...')
        #overlap_similarities = self.calculate_similarities(edge_ids, read_seqs, overlap_lengths)
        overlap_similarities = {edge_id: 1 for edge_id in edge_ids}

        print(f'Done!')
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')

        nx.set_node_attributes(graph_nx, read_lengths, 'read_length')
        nx.set_node_attributes(graph_nx, variant_class, 'variant_class')

        nx.set_edge_attributes(graph_nx, prefix_lengths, 'prefix_length')
        nx.set_edge_attributes(graph_nx, overlap_lengths, 'overlap_length')
        #nx.set_edge_attributes(graph_nx, new_edge, 'new_edge')
        edge_attrs = ['prefix_length', 'overlap_length']

        if training:
            nx.set_node_attributes(graph_nx, read_strands, 'read_strand')
            nx.set_node_attributes(graph_nx, read_starts_M, 'read_start_M')
            nx.set_node_attributes(graph_nx, read_ends_M, 'read_end_M')
            nx.set_node_attributes(graph_nx, read_starts_P, 'read_start_P')
            nx.set_node_attributes(graph_nx, read_ends_P, 'read_end_P')
            nx.set_node_attributes(graph_nx, read_variants, 'read_variant')
            nx.set_node_attributes(graph_nx, read_chrs, 'read_chr')
            #node_attrs.extend(['read_strand', 'read_start', 'read_end', 'read_variant', 'read_chr'])

        
        # Check all nodes and fill in missing support values with neighborhood average
        for node in graph_nx.nodes():
            if node not in node_support:
                # Get 1-hop neighbors
                neighbors = list(graph_nx.predecessors(node)) + list(graph_nx.successors(node))
                if neighbors:
                    # Calculate average support from neighbors that have support values
                    neighbor_supports = [node_support[n] for n in neighbors if n in node_support]
                    if neighbor_supports:
                        node_support[node] = sum(neighbor_supports) / len(neighbor_supports)
                    else:
                        node_support[node] = 0  # No neighbors with support values
                else:
                     node_support[node] = 0  # No neighbors
        
        nx.set_node_attributes(graph_nx, node_support, 'support')
        nx.set_edge_attributes(graph_nx, overlap_similarities, 'overlap_similarity')
        #edge_attrs.append('overlap_similarity')

        # Set hifiasm_result attribute for all edges
        nx.set_edge_attributes(graph_nx, hifiasm_result, 'hifiasm_result')
        print(f"Sum of hifiasm_result values: {sum(hifiasm_result.values())}")

        # Create a dictionary of nodes and their direct successors
        successor_dict = {node: list(graph_nx.successors(node)) for node in graph_nx.nodes()}

        # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?
        if len(read_to_node2) != 0:
            read_to_node = read_to_node2

        # Print number of nodes and edges in graph
        print(f"Number of nodes in graph: {graph_nx.number_of_nodes()}")
 
        return graph_nx, read_seqs, node_to_read, read_to_node, successor_dict

    def liftover_convert_coordinate(self, liftover_obj, chromosome, position):
        """Convert a coordinate using liftover, handling errors gracefully."""
        if position is None:
            return -1
        try:
            new_pos = liftover_obj.convert_coordinate(chromosome, position)
            if new_pos and len(new_pos) > 0:
                return new_pos[0][1]  # Return the first mapped position
            return -1  # No mapping found
        except Exception as e:
            print(f"Liftover error for {chromosome}:{position} - {str(e)}")
            return -1  # Error in liftover
    
    def gt_soft(self, nx_graph):
        gt_bin = nx.get_edge_attributes(nx_graph, 'gt_bin')
        ambiguous = nx.get_node_attributes(nx_graph, 'ambigious')
        yak_attr = nx.get_node_attributes(nx_graph, 'yak_p')
        gt_malicious = nx.get_edge_attributes(nx_graph, 'gt_malicious')

        # Create a dictionary to store gt_bin_soft values
        gt_bin_soft = {}
        
        for edge in nx_graph.edges():
            # Start with the value from gt_bin
            gt_bin_soft[edge] = gt_bin.get(edge, 0)
            
            # Apply rules to modify the soft value

            # target is homozygous
            if gt_bin.get(edge, 0) == 0 and yak_attr.get(edge[1], 0) == 0 and gt_malicious.get(edge, 0) == 0:
                gt_bin_soft[edge] = 0.7

        # Set all the soft values at once
        nx.set_edge_attributes(nx_graph, gt_bin_soft, 'gt_bin_soft')

    def analyze_nodes(self, nx_graph):
        """
        Analyze the out degrees of nodes and edge correctness.
        """
        # Get relevant attributes
        gt_bin = nx.get_edge_attributes(nx_graph, 'gt_bin')
        gt_bin_soft = nx.get_edge_attributes(nx_graph, 'gt_bin_soft')
        ambiguous = nx.get_node_attributes(nx_graph, 'ambigious')
        yak_attr = nx.get_node_attributes(nx_graph, 'yak_p')
        
        # Count statistics
        total_nodes = nx_graph.number_of_nodes()
        zero_out_degree = 0
        only_wrong_edges = 0
        soft_values = []
        
        # Analyze each node
        for node in nx_graph.nodes():
            successors = list(nx_graph.successors(node))
            
            # Check if node has zero out degree
            if len(successors) == 0:
                zero_out_degree += 1
                continue
                
            # Check if all outgoing edges are wrong (gt_bin=0)
            outgoing_edges = [(node, succ) for succ in successors]
            all_wrong = all(gt_bin.get((node, succ), 0) == 0 for succ in successors)
            
            if all_wrong and len(successors) > 0:
                only_wrong_edges += 1
                # Collect the gt_bin_soft values for these edges
                for succ in successors:
                    edge = (node, succ)
                    if edge in gt_bin_soft:
                        soft_values.append(gt_bin_soft[edge])
        
        # Print results
        print("\nNode Analysis Results:")
        print(f"Total nodes: {total_nodes}")
        print(f"Nodes with 0 out degree: {zero_out_degree} ({zero_out_degree/total_nodes*100:.2f}%)")
        print(f"Nodes with only wrong outgoing edges: {only_wrong_edges} ({only_wrong_edges/total_nodes*100:.2f}%)")
        
        # Analyze gt_bin_soft values for nodes with only wrong outgoing edges
        if soft_values:
            print("\ngt_bin_soft values for nodes with only wrong outgoing edges:")
            value_counts = Counter(soft_values)
            for value, count in sorted(value_counts.items()):
                print(f"  Value {value}: {count} edges ({count/len(soft_values)*100:.2f}%)")
            print(f"  Mean value: {sum(soft_values)/len(soft_values):.4f}")
            print(f"  Min value: {min(soft_values)}")
            print(f"  Max value: {max(soft_values)}")

    def get_correct_edges_double_pos(self, edges, read_start_dict_M, read_end_dict_M, read_start_dict_P, read_end_dict_P, read_variant_dict, positive=True):
        """
        Check if edges are correct by examining overlaps in both maternal and paternal coordinate systems.
        An edge is considered correct if it has a valid overlap in either coordinate system.
        Incorrect edges are classified as forward or backward skips based on strand direction.
        
        Args:
            edges: Set of edges to check
            read_start_dict_M: Dictionary of read start positions in maternal coordinates
            read_end_dict_M: Dictionary of read end positions in maternal coordinates
            read_start_dict_P: Dictionary of read start positions in paternal coordinates
            read_end_dict_P: Dictionary of read end positions in paternal coordinates
            read_variant_dict: Dictionary mapping reads to their variant (M or P)
            positive: Boolean indicating strand direction
            
        Returns:
            correct: Set of edges that have valid overlaps
            forward_skips: Set of edges that are forward skips
            backward_skips: Set of edges that are backward skips
            other_incorrect: Set of edges that are incorrect but not classifiable as skips
        """
        # Initialize sets for different edge types
        correct = set()
        forward_skips = set()
        backward_skips = set()
        other_incorrect = set()
        
        for edge in edges:
            src, dst = edge
            is_correct = False
            
            # Check which coordinate systems have complete data for both reads
            maternal_complete = (src in read_start_dict_M and dst in read_start_dict_M and 
                               read_start_dict_M[src] is not None and read_start_dict_M[dst] is not None and
                               read_end_dict_M[src] is not None and read_end_dict_M[dst] is not None)
            
            paternal_complete = (src in read_start_dict_P and dst in read_start_dict_P and 
                               read_start_dict_P[src] is not None and read_start_dict_P[dst] is not None and
                               read_end_dict_P[src] is not None and read_end_dict_P[dst] is not None)
            
            classification_coords = None
            

            """print(f"src: {src}, dst: {dst}")
            print(f"Maternal coordinates:")
            print(f"  src: start={read_start_dict_M[src]}, end={read_end_dict_M[src]}")
            print(f"  dst: start={read_start_dict_M[dst]}, end={read_end_dict_M[dst]}")
            print(f"Paternal coordinates:")
            print(f"  src: start={read_start_dict_P[src]}, end={read_end_dict_P[src]}")
            print(f"  dst: start={read_start_dict_P[dst]}, end={read_end_dict_P[dst]}")"""

            # Check maternal coordinates if both nodes have them
            if maternal_complete:
                read_start_src_M = read_start_dict_M[src]
                read_end_src_M = read_end_dict_M[src]
                read_start_dst_M = read_start_dict_M[dst]
                read_end_dst_M = read_end_dict_M[dst]
                
                # Check if edge is correct in maternal coordinates
                if self.is_correct_edge(read_start_src_M, read_end_src_M,
                                       read_start_dst_M, read_end_dst_M, positive):
                    is_correct = True
                else:
                    # Store maternal coordinates for skip classification
                    classification_coords = (read_start_src_M, read_end_src_M, read_start_dst_M, read_end_dst_M)
            
            # Check paternal coordinates if both nodes have them and edge not already correct
            if not is_correct and paternal_complete:
                read_start_src_P = read_start_dict_P[src]
                read_end_src_P = read_end_dict_P[src]
                read_start_dst_P = read_start_dict_P[dst]
                read_end_dst_P = read_end_dict_P[dst]
                
                # Check if edge is correct in paternal coordinates
                if self.is_correct_edge(read_start_src_P, read_end_src_P,
                                       read_start_dst_P, read_end_dst_P, positive):
                    is_correct = True
                else:
                    # Use paternal coordinates for skip classification if maternal not complete or already incorrect
                    if not maternal_complete:
                        classification_coords = (read_start_src_P, read_end_src_P, read_start_dst_P, read_end_dst_P)
            
            # Add edge to appropriate set
            if is_correct:
                correct.add(edge)
            else:
                # Classify the type of skip if we have complete coordinates
                if classification_coords is not None:
                    src_start, src_end, dst_start, dst_end = classification_coords
                    
                    if positive:
                        # For positive strand:
                        # Forward skip: dst_start >= src_end (destination starts at or after source ends)
                        # Backward skip: dst_start <= src_start (destination starts at or before source starts)
                        if dst_start >= src_end:
                            forward_skips.add(edge)
                        elif dst_start <= src_start:
                            backward_skips.add(edge)
                        else:
                            # This shouldn't happen if is_correct_edge logic is consistent
                            other_incorrect.add(edge)
                    else:
                        # For negative strand:
                        # Forward skip: src_start >= dst_end (source starts at or after destination ends)
                        # Backward skip: src_start <= dst_start (source starts at or before destination starts)
                        if src_start >= dst_end:
                            forward_skips.add(edge)
                        elif src_start <= dst_start:
                            backward_skips.add(edge)
                        else:
                            # This shouldn't happen if is_correct_edge logic is consistent
                            other_incorrect.add(edge)
                else:
                    # No complete coordinate system available for classification
                    other_incorrect.add(edge)
        
        print(f"correct: {len(correct)}, forward_skips: {len(forward_skips)}, backward_skips: {len(backward_skips)}, other_incorrect: {len(other_incorrect)}")
        return correct, forward_skips, backward_skips, other_incorrect

    def find_trap_edges(self, graph, gt_bin):
        """
        Find trap edges - edges that lead to nodes with only wrong outgoing edges.
        A trap edge is an edge (u, v) where node v has at least one outgoing edge
        and ALL of its outgoing edges have gt_bin = 0.
        
        This function iterates until no new trap edges are found, since marking
        edges as wrong can create new trap edges.
        
        Args:
            graph: NetworkX graph
            gt_bin: Dictionary mapping edges to gt_bin values (0 = wrong, 1 = correct)
            
        Returns:
            Set of trap edges
        """
        trap_edges = set()
        iteration = 0
        
        while iteration < 80000:
            iteration += 1
            new_trap_edges = set()
            
            for edge in graph.edges():
                # Skip if already identified as trap edge
                if edge in trap_edges:
                    continue
                    
                src, dst = edge
                
                # Get all outgoing edges from the destination node
                outgoing_edges = list(graph.out_edges(dst))
                
                # Skip if destination node has no outgoing edges (not a trap)
                if len(outgoing_edges) == 0:
                    continue
                    
                # Check if ALL outgoing edges are wrong (gt_bin = 0 or already trap edge)
                all_wrong = all(
                    gt_bin.get(out_edge, 1) == 0 or out_edge in trap_edges 
                    for out_edge in outgoing_edges
                )
                
                if all_wrong:
                    new_trap_edges.add(edge)
            
            # If no new trap edges found, we're done
            if not new_trap_edges:
                print(f"Trap edge detection converged after {iteration} iterations")
                break
                
            # Add new trap edges to the total set
            trap_edges.update(new_trap_edges)
            print(f"Iteration {iteration}: Found {len(new_trap_edges)} new trap edges (total: {len(trap_edges)})")
                
        return trap_edges

    def add_hifiasm_final_edges_attribute(self, nx_graph):
        """
        Load final assembly edges from HiFiasm and add them as a new edge attribute 'hifiasm_edges'.
        This method takes an existing NetworkX graph and marks edges that appear in HiFiasm's final assembly.
        
        Args:
            nx_graph: NetworkX graph to add the attribute to
            
        Returns:
            NetworkX graph with the new 'hifiasm_edges' attribute added
        """
        print(f"Loading HiFiasm final assembly edges for attribute creation...")
        
        # Initialize the hifiasm_edges attribute for all existing edges
        hifiasm_edges = {edge: 0 for edge in nx_graph.edges()}
        
        # Get the paths for HiFiasm assembly outputs
        if self.diploid:
            hifiasm_asm_output_1 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap1.gfa')
            hifiasm_asm_output_2 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap2.gfa')
            
            # Load final assembly edges from both haplotypes
            final_assembly_edges_1, _, _ = self.add_hifiasm_final_edges(hifiasm_asm_output_1)
            final_assembly_edges_2, _, _ = self.add_hifiasm_final_edges(hifiasm_asm_output_2)
            final_assembly_edges = final_assembly_edges_1 + final_assembly_edges_2
        else:
            hifiasm_asm_output = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.gfa')
            final_assembly_edges, _, _ = self.add_hifiasm_final_edges(hifiasm_asm_output)
        
        # Load the existing read-to-node mapping from the saved pickle file
        read_to_node_path = os.path.join(self.read_to_node_path, f'{self.genome_str}.pkl')
        try:
            with open(read_to_node_path, 'rb') as f:
                read_to_node = pickle.load(f)
            print(f"Loaded existing read-to-node mapping with {len(read_to_node)} entries")
        except FileNotFoundError:
            print(f"Warning: Read-to-node mapping file not found at {read_to_node_path}")
            print("Cannot proceed without the mapping. Please ensure parse_gfa step has been run.")
            return nx_graph
        
        # Process each final assembly edge
        edges_found = 0
        edges_not_found = 0
        
        for edge in final_assembly_edges:
            src_read, dst_read = edge
            
            # Check if both reads exist in the mapping
            if src_read in read_to_node and dst_read in read_to_node:
                # Get the real node IDs (assuming forward orientation)
                src_node = read_to_node[src_read][0]  # Real node
                dst_node = read_to_node[dst_read][0]  # Real node
                
                # Check if the edge exists in the graph
                if (src_node, dst_node) in nx_graph.edges():
                    hifiasm_edges[(src_node, dst_node)] = 1
                    edges_found += 1
                
                # Also check the reverse complement edge
                rc_src = dst_node ^ 1
                rc_dst = src_node ^ 1
                if (rc_src, rc_dst) in nx_graph.edges():
                    hifiasm_edges[(rc_src, rc_dst)] = 1
                    edges_found += 1
            else:
                edges_not_found += 1
        
        # Set the attribute on the graph
        nx.set_edge_attributes(nx_graph, hifiasm_edges, 'hifiasm_edges')
        
        return nx_graph

    def infer_unitig_coordinates_from_overlaps(self, reads_info, maternal_reads, paternal_reads, use_liftover, liftover_objects, m_suffix, p_suffix, overlap_data=None):
        """
        Infer coordinates for a unitig using overlap information when possible,
        falling back to liftover only when necessary.
        
        This method uses the actual read lengths and overlap structure to calculate
        precise coordinates, rather than relying solely on liftover transformations.
        
        Args:
            reads_info: List of read information dictionaries (ordered as they appear in unitig)
            maternal_reads: List of maternal reads in the unitig
            paternal_reads: List of paternal reads in the unitig
            use_liftover: Boolean indicating whether liftover is available
            liftover_objects: Dictionary of liftover objects
            m_suffix: Maternal suffix for chromosome names
            p_suffix: Paternal suffix for chromosome names
            overlap_data: Optional dict of (src_node, dst_node) -> overlap_length for precise calculations
            
        Returns:
            Tuple of (start_M, end_M, start_P, end_P)
        """
        if not reads_info:
            return None, None, None, None
            
        chromosome = reads_info[0]['chromosome']
        
        # Step 1: Get direct coordinates for reads that have them
        maternal_coords = {}  # read_id -> (start, end)
        paternal_coords = {}  # read_id -> (start, end)
        
        for read in maternal_reads:
            maternal_coords[read['id']] = (read['start'], read['end'])
            
        for read in paternal_reads:
            paternal_coords[read['id']] = (read['start'], read['end'])
        
        # Step 2: If we have reads from both haplotypes, try coordinate inference
        # using overlap and length information
        if maternal_reads and paternal_reads and len(reads_info) > 1:
            # Try to infer missing coordinates using overlap structure
            maternal_coords = self._infer_missing_coordinates_in_read_chain(reads_info, maternal_coords, 'M', overlap_data)
            paternal_coords = self._infer_missing_coordinates_in_read_chain(reads_info, paternal_coords, 'P', overlap_data)
        
        # Step 3: Calculate unitig-level coordinates
        start_M = end_M = start_P = end_P = None
        
        if maternal_coords:
            all_starts = [coords[0] for coords in maternal_coords.values()]
            all_ends = [coords[1] for coords in maternal_coords.values()]
            start_M = min(all_starts)
            end_M = max(all_ends)
        
        if paternal_coords:
            all_starts = [coords[0] for coords in paternal_coords.values()]
            all_ends = [coords[1] for coords in paternal_coords.values()]
            start_P = min(all_starts)
            end_P = max(all_ends)
        
        # Step 4: Use liftover for completely missing haplotype coordinates
        if maternal_coords and not paternal_coords and use_liftover:
            start_P, end_P = self._liftover_coordinates(chromosome, start_M, end_M, 
                                                       'm_to_p', liftover_objects, m_suffix)
        elif paternal_coords and not maternal_coords and use_liftover:
            start_M, end_M = self._liftover_coordinates(chromosome, start_P, end_P, 
                                                       'p_to_m', liftover_objects, p_suffix)
        
        return start_M, end_M, start_P, end_P
    
    def _infer_missing_coordinates_in_read_chain(self, reads_info, known_coords, haplotype, overlap_data=None):
        """
        Infer coordinates for reads that are missing them by using overlap and length
        information from reads that do have coordinates.
        
        Args:
            reads_info: List of read information dictionaries (ordered)
            known_coords: Dict of read_id -> (start, end) for reads with known coordinates
            haplotype: 'M' or 'P' to filter reads by haplotype
            overlap_data: Optional dict of (src_node, dst_node) -> overlap_length
            
        Returns:
            Updated coordinates dictionary with inferred coordinates added
        """
        # Filter reads for this haplotype
        haplotype_reads = [r for r in reads_info if r['variant'] == haplotype]
        if len(haplotype_reads) <= 1:
            return known_coords
        
        inferred_coords = known_coords.copy()
        
        # Try forward inference: use a read with known coordinates to infer later reads
        for i, reference_read in enumerate(haplotype_reads):
            if reference_read['id'] in inferred_coords:
                # Found a reference read with coordinates, try to infer forward
                ref_start, ref_end = inferred_coords[reference_read['id']]
                current_position = ref_start  # Start from beginning of reference read
                
                # Infer coordinates for subsequent reads
                for j in range(i + 1, len(haplotype_reads)):
                    target_read = haplotype_reads[j]
                    if target_read['id'] not in inferred_coords:
                        # Calculate position based on accumulated length between reads
                        accumulated_length = self._calculate_accumulated_length_between_reads(
                            haplotype_reads, i, j, reads_info, overlap_data
                        )
                        
                        if accumulated_length is not None:
                            # Infer start position (accounting for accumulated sequence)
                            inferred_start = ref_start + accumulated_length
                            # Read length from the read info
                            read_length = target_read['end'] - target_read['start']
                            inferred_end = inferred_start + read_length
                            
                            inferred_coords[target_read['id']] = (inferred_start, inferred_end)
                            print(f"Inferred {haplotype} coordinates for read {target_read['id']}: "
                                  f"{inferred_start}-{inferred_end} (forward from {reference_read['id']}, offset +{accumulated_length})")
                
                break  # We found one reference, that's enough for forward inference
        
        # Try backward inference: use a read with known coordinates to infer earlier reads
        for i in range(len(haplotype_reads) - 1, -1, -1):
            reference_read = haplotype_reads[i]
            if reference_read['id'] in inferred_coords:
                # Found a reference read with coordinates, try to infer backward
                ref_start, ref_end = inferred_coords[reference_read['id']]
                
                # Infer coordinates for previous reads
                for j in range(i - 1, -1, -1):
                    target_read = haplotype_reads[j]
                    if target_read['id'] not in inferred_coords:
                        # Calculate position based on accumulated length between reads
                        accumulated_length = self._calculate_accumulated_length_between_reads(
                            haplotype_reads, j, i, reads_info, overlap_data
                        )
                        
                        if accumulated_length is not None:
                            # Infer start position (going backward)
                            read_length = target_read['end'] - target_read['start']
                            inferred_start = ref_start - accumulated_length
                            inferred_end = inferred_start + read_length
                            
                            inferred_coords[target_read['id']] = (inferred_start, inferred_end)
                            print(f"Inferred {haplotype} coordinates for read {target_read['id']}: "
                                  f"{inferred_start}-{inferred_end} (backward from {reference_read['id']}, offset -{accumulated_length})")
                
                break  # We found one reference, that's enough for backward inference
        
        return inferred_coords
    
    def _calculate_accumulated_length_between_reads(self, haplotype_reads, start_idx, end_idx, reads_info, overlap_data=None):
        """
        Calculate the accumulated sequence length between two reads in a haplotype chain.
        This sums read lengths and subtracts overlaps.
        
        Args:
            haplotype_reads: List of reads for this haplotype
            start_idx: Starting read index (inclusive)
            end_idx: Ending read index (exclusive)
            reads_info: Full read information
            overlap_data: Optional dict of (src_node, dst_node) -> overlap_length for precise calculations
            
        Returns:
            Accumulated length in bases, or None if cannot calculate
        """
        if start_idx >= end_idx:
            return 0
        
        accumulated = 0
        
        # Add lengths of reads from start_idx to end_idx-1
        for i in range(start_idx, end_idx):
            read = haplotype_reads[i]
            read_length = read['end'] - read['start']
            accumulated += read_length
        
        # Subtract overlaps between consecutive reads
        overlap_total = 0
        for i in range(start_idx, end_idx - 1):
            current_read = haplotype_reads[i]
            next_read = haplotype_reads[i + 1]
            
            # Try to get actual overlap if overlap_data is provided
            overlap_length = None
            if overlap_data:
                # This would require mapping read IDs to node IDs to look up overlaps
                # For now, use the estimated approach
                pass
            
            if overlap_length is None:
                # Use reasonable estimate based on typical HiFi read overlaps
                # This could be improved by analyzing the distribution of overlaps in the dataset
                estimated_overlap = min(1500, read_length * 0.1)  # 10% of read length, max 1.5kb
                overlap_length = estimated_overlap
            
            overlap_total += overlap_length
        
        accumulated -= overlap_total
        
        return accumulated if accumulated > 0 else None

    def _liftover_coordinates(self, chromosome, start_coord, end_coord, direction, liftover_objects, suffix):
        """
        Helper method to perform liftover coordinate conversion.
        
        Args:
            chromosome: Chromosome identifier
            start_coord: Start coordinate to convert
            end_coord: End coordinate to convert
            direction: 'm_to_p' or 'p_to_m'
            liftover_objects: Dictionary of liftover objects
            suffix: Chromosome suffix (M or P)
            
        Returns:
            Tuple of (converted_start, converted_end) or (None, None) if failed
        """
        if start_coord is None or end_coord is None:
            return None, None
            
        chr_id = f"chr{chromosome}"
        chr_name = f'{chr_id}_{suffix}'
        
        try:
            if 'multi' in self.chrN and chr_id in liftover_objects and liftover_objects[chr_id][direction] is not None:
                liftover_obj = liftover_objects[chr_id][direction]
            elif liftover_objects.get(direction) is not None:
                liftover_obj = liftover_objects[direction]
            else:
                return None, None
            
            converted_start = self.liftover_convert_coordinate(liftover_obj, chr_name, start_coord)
            converted_end = self.liftover_convert_coordinate(liftover_obj, chr_name, end_coord)
            
            # Convert -1 to None for failed liftovers
            converted_start = converted_start if converted_start != -1 else None
            converted_end = converted_end if converted_end != -1 else None
            
            return converted_start, converted_end
            
        except Exception as e:
            print(f"Liftover error for {chr_name}:{start_coord}-{end_coord}: {str(e)}")
            return None, None

    def _load_reads_data_optimized(self, reads_path):
        """Optimized method to load both headers and sequences in a single pass."""
        print(f"Loading reads data optimized from {reads_path}")
        filetype = self._get_filetype(reads_path)
        read_headers = {}
        read_sequences = {}
        
        def process_record(record):
            if self.real:
                # Extract ID from header like 'read=4294109,m54329U_211102_230231/135006270/ccs,pos_on_original_read=0-17097'
                header = record.description if record.description else record.id
                if header.startswith('read='):
                    extracted_id = header.split(',')[0].split('=')[1]
                else:
                    extracted_id = record.id
                read_headers[extracted_id] = record.description
                read_sequences[extracted_id] = record.seq
            else:
                read_headers[record.id] = record.description
                read_sequences[record.id] = record.seq
        
        if reads_path.endswith('.gz'):
            with gzip.open(reads_path, 'rt') as handle:
                for record in SeqIO.parse(handle, filetype):
                    process_record(record)
        else:
            for record in SeqIO.parse(reads_path, filetype):
                process_record(record)
        
        return read_headers, read_sequences

    def _process_gfa_streaming(self, gfa_path):
        """Process GFA file line by line instead of loading entire file into memory."""
        print(f"Processing GFA file streaming from {gfa_path}")
        
        with open(gfa_path, 'r') as ff:
            for line in ff:
                line = line.strip()
                if not line:
                    continue
                yield line.split()

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance."""
        if not hasattr(self, '_regex_patterns'):
            self._regex_patterns = {
                'strand': re.compile(r'strand=(\+|\-)'),
                'start': re.compile(r'start=(\d+)'),
                'end': re.compile(r'end=(\d+)'),
                'variant': re.compile(r'variant=([P|M])'),
                'chr': re.compile(r'chr=([^\s]+)'),
                'hifiasm_id': re.compile(r'(.*):\d-\d*')
            }
        return self._regex_patterns

    def _extract_description_info(self, description):
        """Extract information from description using pre-compiled regex patterns."""
        patterns = self._compile_regex_patterns()
        
        strand_match = patterns['strand'].search(description)
        start_match = patterns['start'].search(description)
        end_match = patterns['end'].search(description)
        chr_match = patterns['chr'].search(description)
        variant_match = patterns['variant'].search(description)
        
        return {
            'strand': strand_match.group(1) if strand_match else None,
            'start': int(start_match.group(1)) if start_match else None,
            'end': int(end_match.group(1)) if end_match else None,
            'chromosome': chr_match.group(1) if chr_match else None,
            'variant': variant_match.group(1) if variant_match else None
        }

    def _process_pending_a_lines(self, pending_a_lines, current_s_id, node_idx, 
                                read_to_node2, node_to_read, read_headers, 
                                training, use_liftover, liftover_objects, 
                                m_suffix, p_suffix, read_strands, read_starts_M, 
                                read_ends_M, read_starts_P, read_ends_P, 
                                read_variants, read_chrs):
        """Process pending A lines for unitig processing."""
        if not current_s_id.startswith('utg'):
            return
            
        real_idx = node_idx
        virt_idx = node_idx + 1
        
        # The issue here is that in some cases, one unitig can consist of more than one read
        # So this is the adapted version of the code that supports that
        # The only things of importance here are read_to_node2 dict (not overly used)
        # And id variable which I use for obtaining positions during training (for the labels)
        # I don't use it for anything else, which is good
        ids = []
        haplotypes = []  # Collect haplotypes from HG:A field
        
        for line in pending_a_lines:
            tag = line[0]
            utg_id = line[1]
            read_orientation = line[3]
            utg_to_read = line[4]
            ids.append((utg_to_read, read_orientation))
            read_to_node2[utg_to_read] = (real_idx, virt_idx)
            
            # Extract haplotype from HG:A field at index 8
            haplotype = line[8].split(':')[2] if len(line) > 8 and line[8].startswith('HG:A:') else None
            if haplotype:
                haplotypes.append(haplotype)

        id = ids
        node_to_read[real_idx] = id
        node_to_read[virt_idx] = id
        
        # Determine unitig haplotype assignment
        has_m = 'm' in haplotypes
        has_p = 'p' in haplotypes
        
        # Create ambiguous flag: 1 if both m and p appear, 0 otherwise
        is_ambiguous = 1 if (has_m and has_p) else 0
        
        # Determine haplotype assignment
        if has_m and not has_p:
            # All maternal, assign to m
            unitig_haplotype = 'm'
        elif has_p and not has_m:
            # All paternal, assign to p  
            unitig_haplotype = 'p'
        else:
            # Mixed or neither, assign to 0
            unitig_haplotype = '0'
        
        # Create yak_m and yak_p attributes
        if unitig_haplotype == 'm':
            yak_m_val = 1
            yak_p_val = -1
        elif unitig_haplotype == 'p':
            yak_m_val = -1
            yak_p_val = 1
        else:
            yak_m_val = 0
            yak_p_val = 0
        
        # Store the attributes for both real and virtual nodes
        """yak_m[real_idx] = yak_m_val
        yak_m[virt_idx] = yak_m_val
        yak_p[real_idx] = yak_p_val
        yak_p[virt_idx] = yak_p_val
        ambiguous[real_idx] = is_ambiguous
        ambiguous[virt_idx] = is_ambiguous"""
        
        if training:
            #print(f"id: {id}")
            if type(id) != list:
                print("unexpected woooow")
                exit()
            else:
                # Collect reads information first
                reads_info = []
                for id_r, id_o in id:
                    description = read_headers[id_r]
                    # OPTIMIZATION: Use optimized description parsing
                    desc_info = self._extract_description_info(description)
                    strand_fasta = 1 if desc_info['strand'] == '+' else -1
                    strand_gfa = 1 if id_o == '+' else -1
                    strand = strand_fasta * strand_gfa
                    
                    chromosome = desc_info['chromosome']
                    variant = desc_info['variant']
                    start = desc_info['start']
                    end = desc_info['end']
                    
                    reads_info.append({
                        'id': id_r,
                        'orientation': id_o,
                        'strand': strand,
                        'chromosome': chromosome,
                        'variant': variant,
                        'start': start,
                        'end': end
                    })
                
                # Separate reads by haplotype
                maternal_reads = [r for r in reads_info if r['variant'] == 'M']
                paternal_reads = [r for r in reads_info if r['variant'] == 'P']
                
                # Calculate unitig-level attributes
                strand = 1 if sum(r['strand'] for r in reads_info) >= 0 else -1
                chromosome = Counter(r['chromosome'] for r in reads_info).most_common()[0][0]
                variant = Counter(r['variant'] for r in reads_info).most_common()[0][0]
                
                # Use the new coordinate inference method
                start_M, end_M, start_P, end_P = self.infer_unitig_coordinates_from_overlaps(
                    reads_info, maternal_reads, paternal_reads, use_liftover, 
                    liftover_objects, m_suffix, p_suffix
                )
                
                # Initialize coordinates
                start_M = end_M = start_P = end_P = None
                
                # Calculate maternal coordinates
                if maternal_reads:
                    maternal_starts = [r['start'] for r in maternal_reads]
                    maternal_ends = [r['end'] for r in maternal_reads]
                    start_M = min(maternal_starts)
                    end_M = max(maternal_ends)
                    
                    # If we have maternal reads but no paternal reads, use liftover for P coordinates
                    if not paternal_reads and use_liftover:
                        chr_id = f"chr{chromosome}"
                        m_chr_name = f'{chr_id}_{m_suffix}'
                        
                        if 'multi' in self.chrN and chr_id in liftover_objects and liftover_objects[chr_id]['m_to_p'] is not None:
                            m_to_p_lo = liftover_objects[chr_id]['m_to_p']
                            start_P = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, start_M)
                            end_P = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, end_M)
                        elif liftover_objects.get('m_to_p') is not None:
                            start_P = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, start_M)
                            end_P = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, end_M)
                        
                        # Convert -1 to None for failed liftovers
                        start_P = start_P if start_P != -1 else None
                        end_P = end_P if end_P != -1 else None
                
                # Calculate paternal coordinates
                if paternal_reads:
                    paternal_starts = [r['start'] for r in paternal_reads]
                    paternal_ends = [r['end'] for r in paternal_reads]
                    start_P = min(paternal_starts)
                    end_P = max(paternal_ends)
                    
                    # If we have paternal reads but no maternal reads, use liftover for M coordinates
                    if not maternal_reads and use_liftover:
                        chr_id = f"chr{chromosome}"
                        p_chr_name = f'{chr_id}_{p_suffix}'
                        
                        if 'multi' in self.chrN and chr_id in liftover_objects and liftover_objects[chr_id]['p_to_m'] is not None:
                            p_to_m_lo = liftover_objects[chr_id]['p_to_m']
                            start_M = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, start_P)
                            end_M = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, end_P)
                        elif liftover_objects.get('p_to_m') is not None:
                            start_M = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, start_P)
                            end_M = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, end_P)
                        
                        # Convert -1 to None for failed liftovers
                        start_M = start_M if start_M != -1 else None
                        end_M = end_M if end_M != -1 else None
                
                # If we have reads from both haplotypes, we can use coordinate inference
                # to improve accuracy by leveraging the overlap structure
                if maternal_reads and paternal_reads and len(reads_info) > 1:
                    # For unitigs with mixed haplotypes, the coordinates should span
                    # the same genomic region. We can use the more reliable haplotype
                    # as the reference and validate/adjust the other.
                    
                    # Calculate the length ratio between haplotypes to detect potential issues
                    maternal_length = end_M - start_M if (start_M is not None and end_M is not None) else 0
                    paternal_length = end_P - start_P if (start_P is not None and end_P is not None) else 0
                    
                    if maternal_length > 0 and paternal_length > 0:
                        length_ratio = abs(maternal_length - paternal_length) / max(maternal_length, paternal_length)
                        
                        # If the lengths differ significantly, prefer the haplotype with more reads
                        if length_ratio > 0.1:  # 10% difference threshold
                            if len(maternal_reads) > len(paternal_reads):
                                # Use maternal as reference, adjust paternal if needed
                                if use_liftover:
                                    chr_id = f"chr{chromosome}"
                                    m_chr_name = f'{chr_id}_{m_suffix}'
                                    
                                    if 'multi' in self.chrN and chr_id in liftover_objects and liftover_objects[chr_id]['m_to_p'] is not None:
                                        m_to_p_lo = liftover_objects[chr_id]['m_to_p']
                                        start_P_ref = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, start_M)
                                        end_P_ref = self.liftover_convert_coordinate(m_to_p_lo, m_chr_name, end_M)
                                    elif liftover_objects.get('m_to_p') is not None:
                                        start_P_ref = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, start_M)
                                        end_P_ref = self.liftover_convert_coordinate(liftover_objects['m_to_p'], m_chr_name, end_M)
                                    
                                    if start_P_ref != -1 and end_P_ref != -1:
                                        start_P = start_P_ref
                                        end_P = end_P_ref
                            
                            elif len(paternal_reads) > len(maternal_reads):
                                # Use paternal as reference, adjust maternal if needed
                                if use_liftover:
                                    chr_id = f"chr{chromosome}"
                                    p_chr_name = f'{chr_id}_{p_suffix}'
                                    
                                    if 'multi' in self.chrN and chr_id in liftover_objects and liftover_objects[chr_id]['p_to_m'] is not None:
                                        p_to_m_lo = liftover_objects[chr_id]['p_to_m']
                                        start_M_ref = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, start_P)
                                        end_M_ref = self.liftover_convert_coordinate(p_to_m_lo, p_chr_name, end_P)
                                    elif liftover_objects.get('p_to_m') is not None:
                                        start_M_ref = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, start_P)
                                        end_M_ref = self.liftover_convert_coordinate(liftover_objects['p_to_m'], p_chr_name, end_P)
                                    
                                    if start_M_ref != -1 and end_M_ref != -1:
                                        start_M = start_M_ref
                                        end_M = end_M_ref

                read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                read_starts_M[real_idx] = read_starts_M[virt_idx] = start_M
                read_ends_M[real_idx] = read_ends_M[virt_idx] = end_M
                read_starts_P[real_idx] = read_starts_P[virt_idx] = start_P
                read_ends_P[real_idx] = read_ends_P[virt_idx] = end_P
                read_variants[real_idx] = read_variants[virt_idx] = variant
                read_chrs[real_idx] = read_chrs[virt_idx] = chromosome

    def check_attributes(self, nx_graph):
        # Ensure all expected edge attributes are present for DGL conversion  
        for attr in self.edge_attrs:
            edge_attr_dict = nx.get_edge_attributes(nx_graph, attr)
            if len(edge_attr_dict) == 0:
                print(f"Warning: Edge attribute '{attr}' not found in graph, creating empty attribute")
                # Create minimal placeholder values so attribute exists and is callable
                placeholder_values = {edge: 0 for edge in nx_graph.edges()}
                nx.set_edge_attributes(nx_graph, placeholder_values, attr)
            elif len(edge_attr_dict) != nx_graph.number_of_edges():
                print(f"Warning: Edge attribute '{attr}' incomplete ({len(edge_attr_dict)}/{nx_graph.number_of_edges()} edges), filling missing edges with 0")
                # Some edges are missing this attribute, fill them with 0
                for edge in nx_graph.edges():
                    if edge not in edge_attr_dict:
                        edge_attr_dict[edge] = 0
                nx.set_edge_attributes(nx_graph, edge_attr_dict, attr)

    def get_hifiasm_result_attr(self, nx_graph):
        """
        Efficiently create the hifiasm_result edge attribute for an existing NetworkX graph.
        This method loads HiFiasm final assembly edges and marks them in the graph.
        
        Args:
            nx_graph: NetworkX graph to add the hifiasm_result attribute to
            
        Returns:
            NetworkX graph with the hifiasm_result attribute added
        """
        print(f"Creating hifiasm_result attribute for graph with {nx_graph.number_of_edges()} edges...")
        
        # Initialize hifiasm_result attribute for all existing edges
        hifiasm_result = {edge: 0 for edge in nx_graph.edges()}
        
        # Load the existing read-to-node mapping from the saved pickle file
        read_to_node_path = os.path.join(self.read_to_node_path, f'{self.genome_str}.pkl')
        try:
            with open(read_to_node_path, 'rb') as f:
                read_to_node = pickle.load(f)
            print(f"Loaded read-to-node mapping with {len(read_to_node)} entries")
        except FileNotFoundError:
            print(f"Warning: Read-to-node mapping file not found at {read_to_node_path}")
            print("Cannot proceed without the mapping. Please ensure parse_gfa step has been run.")
            return nx_graph
        
        # Load HiFiasm final assembly edges
        try:
            if self.diploid:
                hifiasm_asm_output_1 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap1.gfa')
                hifiasm_asm_output_2 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap2.gfa')
                final_assembly_edges_1, _, orientations_1 = self.add_hifiasm_final_edges(hifiasm_asm_output_1)
                final_assembly_edges_2, _, orientations_2 = self.add_hifiasm_final_edges(hifiasm_asm_output_2)
                final_assembly_edges = final_assembly_edges_1 + final_assembly_edges_2
                orientations = orientations_1 + orientations_2
            else:
                hifiasm_asm_output = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.gfa')
                final_assembly_edges, _, orientations = self.add_hifiasm_final_edges(hifiasm_asm_output)
        except Exception as e:
            print(f"Error loading HiFiasm assembly files: {e}")
            # Set all edges to 0 and return
            nx.set_edge_attributes(nx_graph, hifiasm_result, 'hifiasm_result')
            return nx_graph
        
        # Process each final assembly edge and mark corresponding graph edges
        edges_found = 0
        edges_not_found = 0
        
        for i, edge in enumerate(final_assembly_edges):
            src_read, dst_read = edge
            ori = orientations[i]
            
            # Check if both reads exist in the mapping
            if src_read in read_to_node and dst_read in read_to_node:
                # Get node IDs based on orientation
                src_node = read_to_node[src_read][ori[0]]  # ori[0]: 0 for '+', 1 for '-'
                dst_node = read_to_node[dst_read][ori[1]]  # ori[1]: 0 for '+', 1 for '-'
                
                # Check if the edge exists in the graph and mark it
                if (src_node, dst_node) in nx_graph.edges():
                    hifiasm_result[(src_node, dst_node)] = 1
                    edges_found += 1
                
                # Also mark the reverse complement edge
                rc_src = dst_node ^ 1  # XOR with 1 to get reverse complement
                rc_dst = src_node ^ 1
                if (rc_src, rc_dst) in nx_graph.edges():
                    hifiasm_result[(rc_src, rc_dst)] = 1
                    edges_found += 1
            else:
                edges_not_found += 1
        
        # Set the attribute on the graph
        nx.set_edge_attributes(nx_graph, hifiasm_result, 'hifiasm_result')
        
        print(f"HiFiasm result attribute creation complete:")
        print(f"  - Final assembly edges processed: {len(final_assembly_edges)}")
        print(f"  - Graph edges marked as hifiasm_result=1: {edges_found}")
        print(f"  - Final assembly edges not found in graph: {edges_not_found}")
        print(f"  - Total hifiasm_result=1 edges: {sum(hifiasm_result.values())}")
        
        return nx_graph

    def save_to_dgl_and_pyg_by_chromosome(self, nx_graph, graph_number):
        """
        Split the graph by chromosome and save each chromosome as separate DGL and PyG graphs.
        Uses the 'read_chr' node attribute to determine chromosome assignment.
        """
        print()
        print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
        print(f"Total edges in graph: {nx_graph.number_of_edges()}")
        
        # Get chromosome assignments for nodes
        read_chr = nx.get_node_attributes(nx_graph, 'read_chr')
        if not read_chr:
            print("Warning: No 'read_chr' attribute found in graph. Cannot split by chromosome.")
            return
        
        # Group nodes by chromosome
        chr_to_nodes = defaultdict(set)
        nodes_without_chr = 0
        for node, chr_id in read_chr.items():
            if chr_id is None or chr_id == '':
                nodes_without_chr += 1
                continue
            # Normalize chromosome ID (remove 'chr' prefix if present)
            if chr_id.startswith('chr'):
                chr_id = chr_id[3:]
            chr_to_nodes[chr_id].add(node)
        
        if nodes_without_chr > 0:
            print(f"Warning: {nodes_without_chr} nodes have no chromosome assignment")
        
        print(f"Found {len(chr_to_nodes)} chromosomes: {sorted(chr_to_nodes.keys())}")
        
        # Print chromosome statistics
        print("\nChromosome node distribution:")
        for chr_id in sorted(chr_to_nodes.keys()):
            node_count = len(chr_to_nodes[chr_id])
            print(f"  Chromosome {chr_id}: {node_count} nodes")
        
        # Ensure all expected node attributes are present for DGL conversion
        for attr in self.node_attrs_single:
            if len(nx.get_node_attributes(nx_graph, attr)) == 0:
                print(f"Warning: Node attribute '{attr}' not found in graph, creating empty attribute")
                placeholder_values = {node: 0 for node in nx_graph.nodes()}
                nx.set_node_attributes(nx_graph, placeholder_values, attr)
        
        # Check which edge attributes are actually present and filter out missing ones
        available_edge_attrs = []
        for attr in self.edge_attrs_single:
            edge_attr_dict = nx.get_edge_attributes(nx_graph, attr)
            if len(edge_attr_dict) > 0:
                available_edge_attrs.append(attr)
            else:
                print(f"Warning: Edge attribute '{attr}' not found in graph, skipping it")
        
        print(f"Available edge attributes for DGL conversion: {available_edge_attrs}")
        
        # Process each chromosome separately
        total_saved_graphs = 0
        total_saved_nodes = 0
        total_saved_edges = 0
        
        # Statistics collection for gt_bin and importance_weight
        gt_bin_counts = {'0': 0, '1': 0}
        importance_weights = []
        
        for chr_id, nodes in chr_to_nodes.items():
            print(f"\nProcessing chromosome {chr_id} with {len(nodes)} nodes...")
            
            # Create subgraph for this chromosome
            chr_subgraph = nx_graph.subgraph(nodes).copy()
            
            # Get edges that are entirely within this chromosome
            chr_edges = set()
            cross_chr_edges = 0
            for edge in chr_subgraph.edges():
                src, dst = edge
                if src in nodes and dst in nodes:
                    chr_edges.add(edge)
                else:
                    cross_chr_edges += 1
            
            # Create final subgraph with only the edges within this chromosome
            final_chr_graph = nx.DiGraph()
            final_chr_graph.add_nodes_from(chr_subgraph.nodes(data=True))
            final_chr_graph.add_edges_from(chr_subgraph.edges(data=True))
            
            # Filter to only include edges where both endpoints are in this chromosome
            edges_to_remove = []
            for edge in final_chr_graph.edges():
                src, dst = edge
                if src not in nodes or dst not in nodes:
                    edges_to_remove.append(edge)
            
            for edge in edges_to_remove:
                final_chr_graph.remove_edge(*edge)
            
            print(f"Chromosome {chr_id} subgraph: {final_chr_graph.number_of_nodes()} nodes, {final_chr_graph.number_of_edges()} edges")
            if cross_chr_edges > 0:
                print(f"  Removed {cross_chr_edges} cross-chromosome edges")
            
            # Skip if subgraph is empty
            if final_chr_graph.number_of_nodes() == 0:
                print(f"Skipping chromosome {chr_id} - no nodes")
                continue
            
            # Collect statistics for this chromosome's edges
            if 'gt_bin' in available_edge_attrs:
                gt_bin_attr = nx.get_edge_attributes(final_chr_graph, 'gt_bin')
                for edge, gt_bin_val in gt_bin_attr.items():
                    gt_bin_str = str(gt_bin_val)
                    if gt_bin_str in gt_bin_counts:
                        gt_bin_counts[gt_bin_str] += 1
                    else:
                        gt_bin_counts[gt_bin_str] = 1
                
                # Verification: Ensure all incorrect edges in this chromosome are marked as gt_bin=0
                cross_strand_attr = nx.get_edge_attributes(final_chr_graph, 'cross_strand')
                skip_attr = nx.get_edge_attributes(final_chr_graph, 'skip')
                unknown_attr = nx.get_edge_attributes(final_chr_graph, 'unknown')
                
                incorrect_edges_in_chr = set()
                for edge in final_chr_graph.edges():
                    if (cross_strand_attr.get(edge, 0) == 1 or 
                        skip_attr.get(edge, 0) == 1 or 
                        unknown_attr.get(edge, 0) == 1):
                        incorrect_edges_in_chr.add(edge)
                
                incorrect_edges_with_wrong_gt_bin = [edge for edge in incorrect_edges_in_chr if gt_bin_attr.get(edge, 1) != 0]
                if incorrect_edges_with_wrong_gt_bin:
                    print(f"WARNING: Chromosome {chr_id} has {len(incorrect_edges_with_wrong_gt_bin)} incorrect edges with gt_bin=1!")
                    print("Fixing these edges to gt_bin=0...")
                    for edge in incorrect_edges_with_wrong_gt_bin:
                        gt_bin_attr[edge] = 0
                        gt_bin_counts['1'] -= 1
                        gt_bin_counts['0'] += 1
                    # Update the graph attributes
                    nx.set_edge_attributes(final_chr_graph, gt_bin_attr, 'gt_bin')
                else:
                    print(f"✓ Chromosome {chr_id}: All incorrect edges properly marked as gt_bin=0")
            
            if 'importance_weight' in available_edge_attrs:
                importance_weight_attr = nx.get_edge_attributes(final_chr_graph, 'importance_weight')
                importance_weights.extend(list(importance_weight_attr.values()))
            
            # Convert to DGL using only available attributes
            try:
                graph_dgl = dgl.from_networkx(final_chr_graph, 
                                             node_attrs=self.node_attrs_single, 
                                             edge_attrs=available_edge_attrs)
                
                # Verify that attributes are accessible
                print(f"Verifying DGL graph attributes for chromosome {chr_id}...")
                print(f"Available node data keys: {list(graph_dgl.ndata.keys())}")
                print(f"Available edge data keys: {list(graph_dgl.edata.keys())}")
                
                # Test accessibility of each expected attribute
                for attr in self.node_attrs_single:
                    if attr in graph_dgl.ndata:
                        print(f"✓ Node attribute '{attr}' is accessible")
                    else:
                        print(f"✗ Node attribute '{attr}' is NOT accessible")
                        
                for attr in available_edge_attrs:
                    if attr in graph_dgl.edata:
                        print(f"✓ Edge attribute '{attr}' is accessible")
                    else:
                        print(f"✗ Edge attribute '{attr}' is NOT accessible")
                
                # Save with chromosome-specific naming
                chr_genome_str = f'{self.genome}_chr{chr_id}_{graph_number}'
                dgl_save_path = os.path.join(self.dgl_single_chrom_path, f'{chr_genome_str}.dgl')
                dgl.save_graphs(dgl_save_path, graph_dgl)
                
                print(f"Saved DGL graph for chromosome {chr_id} as {chr_genome_str}.dgl")
                print(f"- Node attributes: {self.node_attrs_single}")
                print(f"- Edge attributes: {available_edge_attrs}")
                
                total_saved_graphs += 1
                total_saved_nodes += final_chr_graph.number_of_nodes()
                total_saved_edges += final_chr_graph.number_of_edges()
                
            except Exception as e:
                print(f"Error converting chromosome {chr_id} to DGL: {str(e)}")
                continue
        
        print(f"\nCompleted chromosome-wise graph splitting and saving.")
        print(f"Summary:")
        print(f"  - Total graphs saved: {total_saved_graphs}")
        print(f"  - Total nodes across all graphs: {total_saved_nodes}")
        print(f"  - Total edges across all graphs: {total_saved_edges}")
        print(f"  - Average nodes per graph: {total_saved_nodes/total_saved_graphs:.1f}" if total_saved_graphs > 0 else "  - No graphs saved")
        print(f"  - Average edges per graph: {total_saved_edges/total_saved_graphs:.1f}" if total_saved_graphs > 0 else "  - No graphs saved")
        
        # Print gt_bin distribution statistics
        if gt_bin_counts:
            print(f"  - gt_bin distribution:")
            for gt_bin_val, count in sorted(gt_bin_counts.items()):
                percentage = (count / total_saved_edges * 100) if total_saved_edges > 0 else 0
                print(f"    gt_bin {gt_bin_val}: {count} edges ({percentage:.1f}%)")
        
        # Print importance_weight distribution statistics
        if importance_weights:
            from collections import Counter
            weight_counts = Counter(importance_weights)
            print(f"  - importance_weight distribution:")
            for weight_val, count in sorted(weight_counts.items()):
                percentage = (count / total_saved_edges * 100) if total_saved_edges > 0 else 0
                print(f"    importance_weight {weight_val}: {count} edges ({percentage:.1f}%)")
        
        print(f"All attributes should now be callable by name in the DGL graphs.")

def main():
    parser = argparse.ArgumentParser(description="Generate dataset based on configuration")
    parser.add_argument('--ref', type=str, default='/mnt/sod2-project/csb4/wgs/martin/genome_references', help='Path to references root dir')
    parser.add_argument('--data_path', type=str, default='/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset', help='Path to dataset folder')   
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    dataset_path = args.data_path
    ref_base_path = args.ref
    # Read the configuration file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    full_dataset = create_full_dataset_dict(config)
    dataset_object = DatasetCreator(args.ref, args.data_path, args.config)

    # Process each chromosome
    for chrN, amount in full_dataset.items():
        for i in range(amount):
            gen_steps(dataset_object, chrN, i, config['gen_steps'], ref_base_path)

def full_2_utg(dataset_object, nx_graph, chrN, i):
    import utg_builder
    graph_path = os.path.join(dataset_object.nx_full_graphs_path, f'{dataset_object.genome_str}.pkl')
    reads_fasta = os.path.join(dataset_object.reduced_reads_raw_path, f'{dataset_object.genome_str}.fasta')
    utgnode_to_node_path = os.path.join(dataset_object.utg_node_to_node_path, f'{dataset_object.genome_str}.pkl')

    output_fasta = os.path.join(dataset_object.reduced_reads_path, f'{dataset_object.genome_str}.fasta')
    output_graph = os.path.join(dataset_object.nx_utg_graphs_path, f'{dataset_object.genome_str}.pkl')
    output_gfa_path = os.path.join(dataset_object.utg_gfa_path, f'{dataset_object.genome_str}.gfa')
    print("hello start utg builder")
    utg_builder.load_and_simplify_graph(
        graph_path,
        reads_fasta_path=reads_fasta,
        utgnode_to_node_path=utgnode_to_node_path,
        output_fasta_path=output_fasta,
        output_graph_path=output_graph,
        output_gfa_path=output_gfa_path,
    )



def gen_steps(dataset_object, chrN_, i, gen_step_config, ref_base_path):

    split_chrN = chrN_.split(".")
    chrN = split_chrN[1]
    genome = split_chrN[0]
    chr_id = f'{chrN}_{i}'
    ref_base = (f'{ref_base_path}/{genome}')
    #if i < 33:
    #    return
        
    dataset_object.load_chromosome(genome, chr_id, ref_base)
    print(f'Processing {dataset_object.genome_str}...')

    if gen_step_config['sample_reads']:
        dataset_object.simulate_reads()

    if gen_step_config['create_graphs']:
        dataset_object.create_graphs()
        print(f"Created gfa graph {chrN}_{i}")

    if gen_step_config['parse_gfa']:
        nx_graph = dataset_object.parse_gfa()
        dataset_object.pickle_save(nx_graph, dataset_object.nx_full_graphs_path)
        print(f"Saved nx graph {chrN}_{i}")

    if gen_step_config['full_2_utg']:
        nx_graph = dataset_object.load_full_graph()
        #dataset_object.get_hifiasm_result_attr(nx_graph)
        # Print information about hifiasm edges
        hifiasm_results = nx.get_edge_attributes(nx_graph, 'hifiasm_result')
        total = sum(hifiasm_results.values())
        #print(f"hifiasm_result is in the graph: {hifiasm_results}")
        print(f"Sum of hifiasm_result values: {total}")
        full_2_utg(dataset_object, nx_graph, chrN, i)

        print(f"Saved utg graph {chrN}_{i}")

    if dataset_object.diploid and gen_step_config['create_yak']:
        
        print(f"Creating yak files {chrN}_{i}")
        dataset_object.gen_yak_files()
        print(f"Done with yak files {chrN}_{i}")

    if dataset_object.diploid and gen_step_config['diploid_features']:
        nx_graph = dataset_object.load_utg_graph()

        print(f"Loaded nx graph {chrN}_{i}")
        dataset_object.check_attributes(nx_graph)
        dataset_object.add_trio_binning_labels(nx_graph)
        dataset_object.pickle_save(nx_graph, dataset_object.nx_utg_graphs_path)

    if not dataset_object.real and gen_step_config['ground_truth']:
        nx_graph = dataset_object.load_utg_graph()
        print(f"Loaded nx graph {chrN}_{i}")
        #dataset_object.add_hifiasm_final_edges_attribute(nx_graph)
        dataset_object.create_gt(nx_graph)
        dataset_object.pickle_save(nx_graph, dataset_object.nx_utg_graphs_path)

        """nx_graph = dataset_object.load_utg_graph()
        print(f"Loaded nx graph {chrN}_{i}")
        dataset_object.gt_soft(nx_graph)
        dataset_object.analyze_nodes(nx_graph)
        print(f"Done with ground truth creation {chrN}_{i}")
        dataset_object.pickle_save(nx_graph, dataset_object.nx_utg_graphs_path)"""

    if gen_step_config['ml_graphs']:

        nx_graph = dataset_object.load_utg_graph()

        # Check if chromosome splitting is enabled in config
        # Check if read_chr attribute exists in graph nodes
        split_by_chromosome = False # 'read_chr' in next(iter(nx_graph.nodes(data=True)))[1]
        print(f"Split by chromosome: {split_by_chromosome}")
        if split_by_chromosome:
            print(f"Saving graphs split by chromosome for {chrN}_{i}")
            dataset_object.save_to_dgl_and_pyg_by_chromosome(nx_graph, i)
        else:
            print(f"Saving single combined graph for {chrN}_{i}")
            dataset_object.save_to_dgl_and_pyg(nx_graph)
        
        print(f"Saved DGL and PYG graphs of {chrN}_{i}")

    print("Done for one chromosome!")

if __name__ == "__main__":
    main()
