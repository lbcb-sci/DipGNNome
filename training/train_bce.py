import torch
import torch.nn as nn
import wandb
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
import os
from datetime import datetime
import utils
import argparse
import yaml
import logging

from SymGatedGCN import SymGatedGCNModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedBCEWSymmetryLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCEWSymmetryLoss, self).__init__()
        # Use BCEWithLogitsLoss with proper pos_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.alpha = 0.1
    def forward(self, predictions, rev_predictions, labels, importance_weights=None):
        labels = labels.float()
        BCE_org = self.criterion(predictions, labels)
        BCE_rev = self.criterion(rev_predictions, labels)
        abs_diff = torch.abs(predictions - rev_predictions)
        loss = (BCE_org + BCE_rev + self.alpha * abs_diff)
        
        # Apply importance weights if provided
        if importance_weights is not None:
            loss = loss * importance_weights
        
        loss = loss.mean()
        return loss


def prepare_subgraph(g, sub_g, device, gt_score='gt_bin', importance_weight=False):
    # Move graph to device before accessing its data
    sub_g = sub_g.to(device)

    # Edge features - get edge IDs and access attributes from original graph
    edge_ids = sub_g.edata['_ID'].to(device)
    ol_len = g.edata['overlap_length'][edge_ids].float()
    ol_len /= 10000
    #ol_sim = g.edata['overlap_similarity'][edge_ids].float()
    e = ol_len.unsqueeze(-1)

    #e = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    # Node features - get node IDs and access attributes from original graph
    node_ids = sub_g.ndata['_ID'].to(device)

    # Node degree features from subgraph
    pe_in = sub_g.in_degrees().float().unsqueeze(1)
    pe_out = sub_g.out_degrees().float().unsqueeze(1)

    # Access node attributes from original graph using node IDs
    read_length = g.ndata['read_length'][node_ids].to(device)/100000
    support = g.ndata['support'][node_ids].to(device)

    # Binary labels - access from original graph using edge IDs
    y_bin = g.edata[gt_score][edge_ids].to(device)

    # Importance weights - access from original graph using edge IDs if enabled
    importance_weights = None
    if importance_weight and 'importance_weight' in g.edata:
        importance_weights = g.edata['importance_weight'][edge_ids].to(device)

    """
    y_strand_change = g.edata['strand_change'][edge_ids].to(device)
    y_cross_chr = g.edata['cross_chr'][edge_ids].to(device)
    y_skip_forward = g.edata['skip_forward'][edge_ids].to(device)
    y_skip_backward = g.edata['skip_backward'][edge_ids].to(device)
    y_all = torch.max(torch.stack([y_strand_change, y_cross_chr, y_skip_forward, y_skip_backward]), dim=0)[0]
    """

    # Combine node features - ensure all tensors have same dimensions
    pe = torch.cat((pe_in, pe_out, support.unsqueeze(1), read_length.unsqueeze(1)), dim=1).to(device)

    return sub_g, e, pe, y_bin, importance_weights


def train(model, data_path, train_selection, valid_selection, device, config, diploid=False, symmetry=False, mode='default', loss_weight=0.01, gt_score='gt_bin', importance_weight=False):
    compute_metrics_every_n_train = 1
    compute_metrics_every_n_valid = 1
    overfit = not bool(valid_selection)

    criterion = WeightedBCEWSymmetryLoss(pos_weight=torch.tensor([loss_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    time_start = datetime.now()

    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name) as run:
        try:
            wandb.watch(model, criterion, log='all', log_freq=1000)
        except Exception as e:
            print(f"Warning: Could not watch model with wandb: {e}")
            print("Continuing training without wandb model watching...")

        best_f1_score = 0
        for epoch in range(config['num_epochs']):
            print(f'===> TRAINING EPOCH {epoch}')
            model.train()
            random.shuffle(train_selection)
            train_loss = []
            pred_mean, pred_std = [], []
            train_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

            for idx, graph_name in enumerate(train_selection):
                #if idx < 12:
                #    continue
                graph_path = os.path.join(data_path, graph_name + '.dgl')
                if not os.path.exists(graph_path):
                    logger.warning(f"Graph file not found: {graph_path}, skipping...")
                    continue
                
                try:
                    g = dgl.load_graphs(graph_path)[0][0]
                except Exception as e:
                    logger.error(f"Error loading graph {graph_path}: {e}, skipping...")
                    continue

                num_clusters = utils.get_num_nodes_per_cluster(g, config)
                g = g.long()
                full_graph = bool(num_clusters <= 1)
                if not full_graph:
                    d = dgl.metis_partition(g, num_clusters, extra_cached_hops=config['k_extra_hops'])
                    sub_gs = list(d.values())
                    random.shuffle(sub_gs)
                else:
                    # Create a subgraph that includes all nodes/edges
                    all_nodes = torch.arange(g.number_of_nodes())
                    sub_gs = [dgl.node_subgraph(g, all_nodes)]
                g = g.to(device)
                for sub_g in sub_gs:
                    sub_g, e, x, y_bin, importance_weights = prepare_subgraph(g, sub_g, device, gt_score=gt_score, importance_weight=importance_weight)

                    # Forward predictions
                    edge_predictions, _ = model(sub_g, x, e)
                    edge_predictions = edge_predictions.squeeze(-1)

                    # Reverse predictions
                    sub_g_reverse = dgl.reverse(sub_g, copy_ndata=True, copy_edata=True)
                    rev_edge_predictions, _ = model(sub_g_reverse, x, e)
                    rev_edge_predictions = rev_edge_predictions.squeeze(-1)

                    sigmoid_predictions = torch.sigmoid(edge_predictions)  # Apply sigmoid to get [0, 1] range
                    pred_mean.append(edge_predictions.mean().item())
                    pred_std.append(edge_predictions.std().item())

                    optimizer.zero_grad()
                    loss = criterion(edge_predictions, rev_edge_predictions, y_bin, importance_weights=importance_weights)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())

                    if epoch % compute_metrics_every_n_train == 0:
                        predictions = (sigmoid_predictions > 0.5).float()
                        # Treat class 1 as positive class for metrics
                        tp = ((predictions == 1) & (y_bin == 1)).sum().item()  # Correctly predicted 1
                        fp = ((predictions == 1) & (y_bin == 0)).sum().item()  # Incorrectly predicted 1
                        fn = ((predictions == 0) & (y_bin == 1)).sum().item()  # Missed class 1
                        tn = ((predictions == 0) & (y_bin == 0)).sum().item()  # Correctly predicted 0
                        train_metrics['tp'] += tp
                        train_metrics['fp'] += fp
                        train_metrics['fn'] += fn
                        train_metrics['tn'] += tn

            # Calculate training metrics
            if train_metrics['tp'] + train_metrics['fp'] > 0:
                precision = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fp'])
            else:
                precision = 0

            if train_metrics['tp'] + train_metrics['fn'] > 0:
                recall = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fn'])
            else:
                recall = 0

            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Calculate false positive rate and false negative rate
            if train_metrics['fp'] + train_metrics['tn'] > 0:
                fpr = train_metrics['fp'] / (train_metrics['fp'] + train_metrics['tn'])
            else:
                fpr = 0

            if train_metrics['fn'] + train_metrics['tp'] > 0:
                fnr = train_metrics['fn'] / (train_metrics['fn'] + train_metrics['tp'])
            else:
                fnr = 0

            # Validation loop
            if not overfit:
                print('===> VALIDATION')
                model.eval()
                valid_loss = []
                valid_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

                with torch.no_grad():
                    for idx, graph_name in enumerate(valid_selection):
                        graph_path = os.path.join(data_path, graph_name + '.dgl')
                        if not os.path.exists(graph_path):
                            logger.warning(f"Graph file not found: {graph_path}, skipping...")
                            continue
                        
                        try:
                            g = dgl.load_graphs(graph_path)[0][0]
                        except Exception as e:
                            logger.error(f"Error loading graph {graph_path}: {e}, skipping...")
                            continue

                        num_clusters = utils.get_num_nodes_per_cluster(g, config)
                        g = g.long()
                        full_graph = bool(num_clusters <= 1)
                        if not full_graph:
                            d = dgl.metis_partition(g, num_clusters, extra_cached_hops=config['k_extra_hops'])
                            sub_gs = list(d.values())
                        else:
                            # Create a subgraph that includes all nodes/edges
                            all_nodes = torch.arange(g.number_of_nodes())
                            sub_gs = [dgl.node_subgraph(g, all_nodes)]

                        g = g.to(device)
                        for sub_g in sub_gs:
                            sub_g, e, x, y_bin, importance_weights = prepare_subgraph(g, sub_g, device, gt_score=gt_score, importance_weight=importance_weight)

                            # Forward predictions
                            edge_predictions, _ = model(sub_g, x, e)
                            edge_predictions = edge_predictions.squeeze(-1)

                            # Reverse predictions
                            sub_g_reverse = dgl.reverse(sub_g, copy_ndata=True, copy_edata=True)
                            rev_edge_predictions, _ = model(sub_g_reverse, x, e)
                            rev_edge_predictions = rev_edge_predictions.squeeze(-1)

                            sigmoid_predictions = torch.sigmoid(edge_predictions)  # Apply sigmoid to get [0, 1] range

                            loss = criterion(edge_predictions, rev_edge_predictions, y_bin, importance_weights=importance_weights)
                            valid_loss.append(loss.item())

                            if epoch % compute_metrics_every_n_valid == 0:
                                predictions = (sigmoid_predictions > 0.5).float()
                                # Treat class 1 as positive class for metrics
                                tp = ((predictions == 1) & (y_bin == 1)).sum().item()  # Correctly predicted 1
                                fp = ((predictions == 1) & (y_bin == 0)).sum().item()  # Incorrectly predicted 1
                                fn = ((predictions == 0) & (y_bin == 1)).sum().item()  # Missed class 1
                                tn = ((predictions == 0) & (y_bin == 0)).sum().item()  # Correctly predicted 0
                                valid_metrics['tp'] += tp
                                valid_metrics['fp'] += fp
                                valid_metrics['fn'] += fn
                                valid_metrics['tn'] += tn

                print(f'Validation metrics:')
                print(f'  True positives: {valid_metrics["tp"]}')
                print(f'  False positives: {valid_metrics["fp"]}')
                print(f'  False negatives: {valid_metrics["fn"]}')
                print(f'  True negatives: {valid_metrics["tn"]}')

                # Calculate validation metrics
                if valid_metrics['tp'] + valid_metrics['fp'] > 0:
                    val_precision = valid_metrics['tp'] / (valid_metrics['tp'] + valid_metrics['fp'])
                else:
                    val_precision = 0

                if valid_metrics['tp'] + valid_metrics['fn'] > 0:
                    val_recall = valid_metrics['tp'] / (valid_metrics['tp'] + valid_metrics['fn'])
                else:
                    val_recall = 0

                val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

                # Calculate validation false positive rate and false negative rate
                if valid_metrics['fp'] + valid_metrics['tn'] > 0:
                    val_fpr = valid_metrics['fp'] / (valid_metrics['fp'] + valid_metrics['tn'])
                else:
                    val_fpr = 0

                if valid_metrics['fn'] + valid_metrics['tp'] > 0:
                    val_fnr = valid_metrics['fn'] / (valid_metrics['fn'] + valid_metrics['tp'])
                else:
                    val_fnr = 0

                # Update learning rate scheduler
                scheduler.step(sum(valid_loss) / len(valid_loss) if valid_loss else float('inf'))

            # Log metrics
            metrics_dict = {
                'loss': sum(train_loss) / len(train_loss) if train_loss else 0,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'fnr': fnr,
                'epoch': epoch
            }

            if not overfit:
                metrics_dict.update({
                    'val_loss': sum(valid_loss) / len(valid_loss) if valid_loss else 0,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_fpr': val_fpr,
                    'val_fnr': val_fnr
                })

            # Log metrics to wandb
            try:
                wandb.log(metrics_dict)
            except Exception as e:
                print(f"Warning: Could not log metrics to wandb: {e}")
                print("Continuing training without wandb logging...")

            # Save best model based on F1 score
            current_f1 = val_f1_score if not overfit else f1_score
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                model_path = os.path.join(save_model_path, f'{args.run_name}_best.pt')
                torch.save(model.state_dict(), model_path)
                print(f"Saved model with F1 score: {best_f1_score:.4f}")

            # Save model every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_model_path, f'{args.run_name}_epoch_{epoch}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch}")

            print(f"Epoch {epoch} completed in {datetime.now() - time_start}")
            time_start = datetime.now()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--load_checkpoint", type=str, default='', help="dataset path")
    parser.add_argument("--save_model_path", type=str, default='/mnt/sod2-project/csb4/wgs/martin/trained_models', help="dataset path")
    parser.add_argument("--data_path", type=str, default='data', help="dataset path")
    parser.add_argument("--run_name", type=str, default='test', help="dataset path")
    parser.add_argument("--device", type=str, default='cpu', help="dataset path")
    parser.add_argument("--data_config", type=str, default='data_debug.yml', help="dataset path")
    parser.add_argument("--hyper_config", type=str, default='configs/config.yml', help="dataset path")
    parser.add_argument('--diploid', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--wandb", type=str, default='debug', help="dataset path")
    parser.add_argument('--symmetry', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--aux', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--bce', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--sigmoid', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--hap_switch', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--seed", type=int, default=0, help="dataset path")
    parser.add_argument('--mode', type=str, default='default')
    parser.add_argument('--gt_score', type=str, default='gt_bin')
    parser.add_argument('--importance_weight', action='store_true', default=False, help="Enable importance weighting based on edge attribute")

    args = parser.parse_args()

    wandb_project = args.wandb
    run_name = args.run_name
    load_checkpoint = args.load_checkpoint
    save_model_path = args.save_model_path
    data_path = args.data_path
    device = args.device
    data_config = args.data_config
    diploid = args.diploid
    symmetry = args.symmetry
    aux = args.aux
    hyper_config = args.hyper_config
    hap_switch = args.hap_switch
    mode = args.mode
    sigmoid = args.sigmoid
    gt_score = args.gt_score
    importance_weight = args.importance_weight

    full_dataset, valid_selection, train_selection = utils.create_dataset_dicts(data_config=data_config)
    valid_data = utils.get_numbered_graphs(valid_selection)
    train_data = utils.get_numbered_graphs(train_selection, starting_counts=valid_selection)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    with open(hyper_config) as file:
        config = yaml.safe_load(file)['training']

    utils.set_seed(args.seed)

    add_features = False
    if mode.startswith('ghost'):
        mode_split = mode.split('-')
        add_features = bool(int(mode_split[2]))

    if add_features:
        node_features = config['node_features'] + 1
        edge_features = config['edge_features'] + 1
    else:
        node_features = config['node_features']
        edge_features = config['edge_features']

    model = SymGatedGCNModel(
        node_features,
        edge_features,
        config['hidden_features'],
        config['hidden_edge_features'],
        config['num_gnn_layers'],
        config['hidden_edge_scores'],
        config['nb_pos_enc'],
        config['nr_classes'],
        dropout=config['dropout'],
        strand_change_head=config['strand_change_head'],
        norm=config['norm'] # Default to 'layer' if not specified
    )

    # Add statistics gathering
    total_edges = 0
    total_nodes = 0
    skip_forward_edges = 0
    skip_backward_edges = 0
    strand_change_edges = 0
    cross_chr_edges = 0
    gt_bin_edges = 0
    weighted_positive_edges = 0
    weighted_negative_edges = 0
    
    # Statistics for training and validation splits
    train_stats = {
        'total_edges': 0,
        'total_nodes': 0,
        'gt_bin_edges': 0,
        'weighted_positive_edges': 0,
        'weighted_negative_edges': 0,
        'importance_weights': []
    }
    
    valid_stats = {
        'total_edges': 0,
        'total_nodes': 0,
        'gt_bin_edges': 0,
        'weighted_positive_edges': 0,
        'weighted_negative_edges': 0,
        'importance_weights': []
    }

    # Analyze training graphs
    print("=== TRAINING SET STATISTICS ===")
    for graph_name in train_data:
        graph_path = os.path.join(data_path, graph_name + '.dgl')
        if not os.path.exists(graph_path):
            logger.warning(f"Graph file not found: {graph_path}, skipping...")
            continue
        
        try:
            g = dgl.load_graphs(graph_path)[0][0]
        except Exception as e:
            logger.error(f"Error loading graph {graph_path}: {e}, skipping...")
            continue

        train_stats['total_edges'] += g.num_edges()
        train_stats['total_nodes'] += g.num_nodes()
        total_edges += g.num_edges()
        total_nodes += g.num_nodes()
        
        """if 'skip_forward' in g.edata:
            skip_forward_edges += g.edata['skip_forward'].sum().item()
        if 'skip_backward' in g.edata:
            skip_backward_edges += g.edata['skip_backward'].sum().item()
        if 'strand_change' in g.edata:
            strand_change_edges += g.edata['strand_change'].sum().item()
        if 'cross_chr' in g.edata:
            cross_chr_edges += g.edata['cross_chr'].sum().item()"""
        if gt_score in g.edata:
            train_stats['gt_bin_edges'] += g.edata[gt_score].sum().item()
            gt_bin_edges += g.edata[gt_score].sum().item()
            
            # Calculate weighted statistics if importance weighting is enabled
            if importance_weight and 'importance_weight' in g.edata:
                importance_weights = g.edata['importance_weight']
                train_stats['importance_weights'].extend(importance_weights.tolist())
                
                positive_mask = g.edata[gt_score] == 1
                negative_mask = g.edata[gt_score] == 0
                
                train_stats['weighted_positive_edges'] += (importance_weights * positive_mask).sum().item()
                train_stats['weighted_negative_edges'] += (importance_weights * negative_mask).sum().item()
                weighted_positive_edges += (importance_weights * positive_mask).sum().item()
                weighted_negative_edges += (importance_weights * negative_mask).sum().item()

    # Analyze validation graphs
    print("=== VALIDATION SET STATISTICS ===")
    for graph_name in valid_data:
        graph_path = os.path.join(data_path, graph_name + '.dgl')
        if not os.path.exists(graph_path):
            logger.warning(f"Graph file not found: {graph_path}, skipping...")
            continue
        
        try:
            g = dgl.load_graphs(graph_path)[0][0]
        except Exception as e:
            logger.error(f"Error loading graph {graph_path}: {e}, skipping...")
            continue

        valid_stats['total_edges'] += g.num_edges()
        valid_stats['total_nodes'] += g.num_nodes()
        total_edges += g.num_edges()
        total_nodes += g.num_nodes()
        
        if gt_score in g.edata:
            valid_stats['gt_bin_edges'] += g.edata[gt_score].sum().item()
            gt_bin_edges += g.edata[gt_score].sum().item()
            
            # Calculate weighted statistics if importance weighting is enabled
            if importance_weight and 'importance_weight' in g.edata:
                importance_weights = g.edata['importance_weight']
                valid_stats['importance_weights'].extend(importance_weights.tolist())
                
                positive_mask = g.edata[gt_score] == 1
                negative_mask = g.edata[gt_score] == 0
                
                valid_stats['weighted_positive_edges'] += (importance_weights * positive_mask).sum().item()
                valid_stats['weighted_negative_edges'] += (importance_weights * negative_mask).sum().item()
                weighted_positive_edges += (importance_weights * positive_mask).sum().item()
                weighted_negative_edges += (importance_weights * negative_mask).sum().item()

    # Print comprehensive statistics
    print(f"\n=== COMPREHENSIVE DATASET STATISTICS ===")
    print(f"Ground truth score attribute: {gt_score}")
    
    print(f"\n--- TRAINING SET ---")
    print(f"Total nodes: {train_stats['total_nodes']:,}")
    print(f"Total edges: {train_stats['total_edges']:,}")
    if train_stats['total_nodes'] > 0:
        print(f"Average degree: {2 * train_stats['total_edges'] / train_stats['total_nodes']:.2f}")
    print(f"Edges with {gt_score}=1: {train_stats['gt_bin_edges']:,} ({(train_stats['gt_bin_edges']/train_stats['total_edges'])*100:.2f}%)")
    
    if importance_weight and train_stats['importance_weights']:
        print(f"Weighted positive edges: {train_stats['weighted_positive_edges']:.2f}")
        print(f"Weighted negative edges: {train_stats['weighted_negative_edges']:.2f}")
        print(f"Weighted positive class ratio: {train_stats['weighted_positive_edges']/(train_stats['weighted_positive_edges'] + train_stats['weighted_negative_edges']):.4f}")
        
        # Edge weight statistics
        weights = torch.tensor(train_stats['importance_weights'])
        print(f"Importance weights - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
        
        # Simple weight distribution counts
        weights_lt_1 = (weights < 1).sum().item()
        weights_eq_1 = (weights == 1).sum().item()
        weights_gt_1 = (weights > 1).sum().item()
        print(f"Weight distribution - <1: {weights_lt_1}, ==1: {weights_eq_1}, >1: {weights_gt_1}")
    
    print(f"\n--- VALIDATION SET ---")
    print(f"Total nodes: {valid_stats['total_nodes']:,}")
    print(f"Total edges: {valid_stats['total_edges']:,}")
    if valid_stats['total_nodes'] > 0:
        print(f"Average degree: {2 * valid_stats['total_edges'] / valid_stats['total_nodes']:.2f}")
    print(f"Edges with {gt_score}=1: {valid_stats['gt_bin_edges']:,} ({(valid_stats['gt_bin_edges']/valid_stats['total_edges'])*100:.2f}%)")
    
    if importance_weight and valid_stats['importance_weights']:
        print(f"Weighted positive edges: {valid_stats['weighted_positive_edges']:.2f}")
        print(f"Weighted negative edges: {valid_stats['weighted_negative_edges']:.2f}")
        print(f"Weighted positive class ratio: {valid_stats['weighted_positive_edges']/(valid_stats['weighted_positive_edges'] + valid_stats['weighted_negative_edges']):.4f}")
        
        # Edge weight statistics
        weights = torch.tensor(valid_stats['importance_weights'])
        print(f"Importance weights - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
        
        # Simple weight distribution counts
        weights_lt_1 = (weights < 1).sum().item()
        weights_eq_1 = (weights == 1).sum().item()
        weights_gt_1 = (weights > 1).sum().item()
        print(f"Weight distribution - <1: {weights_lt_1}, ==1: {weights_eq_1}, >1: {weights_gt_1}")
    
    print(f"\n--- COMBINED DATASET ---")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")
    if total_nodes > 0:
        print(f"Average degree: {2 * total_edges / total_nodes:.2f}")
    print(f"Edges with {gt_score}=1: {gt_bin_edges:,} ({(gt_bin_edges/total_edges)*100:.2f}%)")

    #all_bad_edges = skip_forward_edges + skip_backward_edges + strand_change_edges + cross_chr_edges

    # Calculate loss weight considering importance weights if available
    if importance_weight and weighted_positive_edges > 0 and weighted_negative_edges > 0:
        pos_weight = weighted_negative_edges / weighted_positive_edges  # weighted_negatives / weighted_positives
        print(f"Weighted positive edges: {weighted_positive_edges:.2f}")
        print(f"Weighted negative edges: {weighted_negative_edges:.2f}")
        print(f"Weighted positive class ratio: {weighted_positive_edges/(weighted_positive_edges + weighted_negative_edges):.4f}")
    else:
        pos_weight = (total_edges - gt_bin_edges) / gt_bin_edges  # num_negatives / num_positives
        print(f"Unweighted positive class ratio: {gt_bin_edges/total_edges:.4f}")
    
    loss_weight = pos_weight
    print(f"Loss weight (pos_weight): {loss_weight}")
    print(f"Expected initial bias for balanced predictions: {-torch.log(torch.tensor(pos_weight)):.4f}")

    # Log importance weighting status
    if importance_weight:
        print(f"Importance weighting enabled: {importance_weight}")
        # Check if importance_weight attribute exists in at least one graph
        importance_weight_found = False
        for graph_name in train_data[:5]:  # Check first 5 graphs
            graph_path = os.path.join(data_path, graph_name + '.dgl')
            if os.path.exists(graph_path):
                try:
                    g = dgl.load_graphs(graph_path)[0][0]
                    if 'importance_weight' in g.edata:
                        importance_weight_found = True
                        break
                except Exception:
                    continue
        
        if importance_weight_found:
            print("✓ importance_weight attribute found in graph data")
        else:
            print("⚠ Warning: importance_weight attribute not found in graph data. Importance weighting will be disabled.")
            importance_weight = False

    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    train(model, data_path, train_data, valid_data, device, config, diploid, symmetry, mode=mode, loss_weight=loss_weight, gt_score=gt_score, importance_weight=importance_weight)

